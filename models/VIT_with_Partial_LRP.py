# file: partial_lrp_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import torchvision.models as tv_models
from torchvision.models.vision_transformer import ViT_B_16_Weights


# ---------------------------
# Utility: epsilon-rule LRP for Linear (expects W as [in, out])
# ---------------------------
def lrp_linear(a: torch.Tensor, W: torch.Tensor, R_out: torch.Tensor,
               b: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    LRP (epsilon rule) for z = a @ W + b
    Args:
        a:     [B,C] or [B,N,C]          # input activations
        W:     [C_in, C_out]             # weights (in × out)
        R_out: [B,C_out] or [B,N,C_out]  # relevance at the output
        b:     [C_out] or None
        eps:   stabilizer
    Returns:
        R_in:  [B,C] or [B,N,C]
    """
    def _core(a2, Rout2):
        # z = aW（不含 bias）
        z = a2 @ W
        if b is not None:
            z = z + b
        denom = z + eps * torch.sign(z)
        s = Rout2 / (denom + 1e-12)
        c = s @ W.t()
        Rin = a2 * c
        return torch.nan_to_num(Rin, nan=0.0, posinf=0.0, neginf=0.0)

    if a.dim() == 2:
        return _core(a, R_out)
    elif a.dim() == 3:
        B, N, Cin = a.shape
        Cout = W.shape[1]
        a2 = a.view(-1, Cin)
        Rout2 = R_out.view(-1, Cout)
        Rin = _core(a2, Rout2).view(B, N, Cin)
        return Rin
    else:
        raise ValueError("a must be [B,C] or [B,N,C]")


# ---------------------------
# Wrapper: capture internals from MultiheadAttention
# ---------------------------
class MHAWrapper(nn.Module):
    """
    Wrap nn.MultiheadAttention (batch_first=True) to expose:
      - saved_X_in: [B,N,C]
      - saved_out:  [B,N,C]
      - saved_attn_probs: [B,H,N,N]
      - W_o (in×out): [C,C]
      - W_v (in×out): [C,C]
      - saved_V: [B,H,N,d]
    """
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        assert mha.batch_first, "Expect MultiheadAttention with batch_first=True."
        self.mha = mha

        # Keep weights in "in×out" orientation for LRP
        W_o = mha.out_proj.weight.t().detach()                    # [C,C]
        W_qkv = mha.in_proj_weight                                # [3C, C] (q,k,v as out×in)
        C = W_qkv.shape[1]
        W_q = W_qkv[:C, :].t().detach()                          # [in=C, out=C]
        W_k = W_qkv[C:2*C, :].t().detach()                        # [in=C, out=C]
        W_v = W_qkv[2*C:3*C, :].t().detach()                      # [in=C, out=C]
        b_v = (mha.in_proj_bias[2*C:3*C].detach()
               if mha.in_proj_bias is not None else None)         # [C] or None

        # Register as buffers so they track device with model.to(device)
        self.register_buffer('W_o', W_o)
        self.register_buffer('W_q', W_q)
        self.register_buffer('W_k', W_k)
        self.register_buffer('W_v', W_v)
        if b_v is not None:
            self.register_buffer('b_v', b_v)
        else:
            self.b_v = None

        # Buffers
        self.saved_X_in = None
        self.saved_out = None
        self.saved_attn_probs = None
        self.saved_V = None

    # 注意：簽名要和 nn.MultiheadAttention 一致
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # 無論外部要不要 weights，內部一律設 need_weights=True、average_attn_weights=False 以取得 per-head
        out, attn_w = self.mha(
            query, key, value, # query, key, value 通常在 ViT_B_16 都是 X
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,   # 要 per-head
            is_causal=is_causal,
        )

        # 保存中間量（使用 detach() 避免梯度累積）
        self.saved_X_in = query.detach()                          # [B,N,C]
        self.saved_out  = out.detach()                            # [B,N,C]
        self.saved_attn_probs = attn_w.detach()                   # [B,H,N,N]

        # V = X @ W_v + b_v，並 reshape 成 [B,H,N,d]
        # v = query @ self.W_v + (self.b_v if self.b_v is not None else 0.0)  # [B,N,C]
        v = value @ self.W_v + (self.b_v if self.b_v is not None else 0.0) # [B,N,C]
        H = self.mha.num_heads
        d = v.shape[-1] // H # C // H = d
        self.saved_V = v.view(v.shape[0], v.shape[1], H, d).permute(0, 2, 1, 3).detach()

        # torchvision ViT 會做 x, _ = self_attention(..., need_weights=False)
        # 因此這裡無論 need_weights 為何都回傳 (out, weights/None) 的 tuple
        return out, (attn_w if need_weights else None)


# ---------------------------
# Analyzer: Partial LRP for torchvision ViT
# ---------------------------
class VIT_with_Partial_LRP(nn.Module):
    """
    Minimal working Partial LRP for torchvision ViT (vit_b_16, etc.).
    - Performs head-level relevance in attention (Split -> A^T -> partial -> W^V back to X)
    - MLP & residual/LN are simplified (not propagated in this MVP).
    """
    def __init__(self,
                 vit_model: Optional[nn.Module] = None,
                 in_channels: int = 3,
                 out_channels: int = 10,
                 input_size: Tuple[int, int] = (224, 224),
                 topk_heads: Optional[int] = None,
                 head_weighting: str = 'normalize',
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.topk_heads = topk_heads
        self.head_weighting = head_weighting
        self.input_size = input_size

        # 構建或採用外部提供的 ViT backbone（不強制 eval，由訓練流程控制）
        if vit_model is None:
            image_size = input_size[0]
            try:
                backbone = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1, image_size=image_size)
            except TypeError:
                backbone = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # 調整分類頭
            in_features = backbone.heads.head.in_features
            backbone.heads.head = nn.Linear(in_features, out_channels)
            self.model = backbone
        else:
            self.model = vit_model

        # 若輸入通道不是 3，加入 1x1 adapter 轉到 3 通道
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        self.cache: List[Dict[str, Any]] = []
        self._wrap_attentions()
        self._hook_tokens()
        self._hook_mlp_and_ln()  # // NEW
        self._block_counter = 0  # // NEW：追蹤目前是第幾個 block（由 attn hook 推進）
        self._pending_ln1 = []   # ← NEW: ln1 的輸入先暫存這裡（FIFO）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return self.model(x)

    # ---- helpers to navigate module tree ----
    def _get_parent_by_name(self, name: str) -> nn.Module:
        parts = name.split('.')
        obj = self.model
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj

    def _wrap_attentions(self):
        """Replace nn.MultiheadAttention modules with MHAWrapper + hook them."""
        # collect first to avoid modifying while iterating
        targets = [(name, m) for name, m in self.model.named_modules() if isinstance(m, nn.MultiheadAttention)]
        self.attn_wrappers: List[MHAWrapper] = []
        for name, m in targets:
            parent = self._get_parent_by_name(name)
            attr = name.split('.')[-1]
            wrapper = MHAWrapper(m)
            setattr(parent, attr, wrapper)                    # replace in the model
            wrapper.register_forward_hook(self._attn_forward_hook)
            self.attn_wrappers.append(wrapper)

    def _attn_forward_hook(self, module: MHAWrapper, inputs, output):
        # 先從等待佇列取對應的 ln1_in（若沒有就放 None）
        ln1_from_queue = self._pending_ln1.pop(0) if len(self._pending_ln1) > 0 else None

        self.cache.append(dict(
            attn_probs = module.saved_attn_probs,   # [B,H,N,N]
            V          = module.saved_V,            # [B,H,N,d]
            W_o        = module.W_o,                # [C,C]
            W_v        = module.W_v,                # [C,C]
            X_in       = module.saved_X_in,         # [B,N,C]   = ln1(x1) 之後的 token
            out        = module.saved_out,          # [B,N,C]
            ln1_in     = ln1_from_queue,            # ← NEW: 這次 block 的殘差分支 x1
            ln2_in     = None,
            fc1_in     = None, fc1_out=None, act_out=None, fc2_out=None,
            W_fc1      = None, b_fc1=None,
            W_fc2      = None, b_fc2=None,
        ))


    def _hook_tokens(self):
        """Grab final encoder tokens (before classifier)."""
        self.last_tokens = None

        def _save_tokens(_, __, out):
            self.last_tokens = out

        # torchvision ViT: encoder.ln 是最後一個 LayerNorm
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "ln"):
            self.model.encoder.ln.register_forward_hook(_save_tokens)
            print("Debug: Hook registered on encoder.ln")
        else:
            print("Debug: encoder.ln not found, trying alternative structures...")
            # 嘗試其他可能的結構
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "norm"):
                self.model.encoder.norm.register_forward_hook(_save_tokens)
                print("Debug: Hook registered on encoder.norm")
            elif hasattr(self.model, "norm"):
                self.model.norm.register_forward_hook(_save_tokens)
                print("Debug: Hook registered on model.norm")
            else:
                print("Debug: No suitable LayerNorm found for hooking")
                # 列出所有模組以幫助調試
                print("Available modules:")
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        print(f"  - {name}: {type(module)}")

    # ---------------------------
    # Public API
    # ---------------------------
    @torch.no_grad()
    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                return_map: str = 'patch', upsample_to_input: bool = False, 
                return_head_importance: bool = False) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return_map: 'patch' -> [B, grid_h*grid_w], 'token' -> [B,N,C], 'image' -> [B, H, W]
        return_head_importance: bool -> 是否回傳每層 head 重要性分數
        """
        # 清理所有累積的張量，避免記憶體洩漏
        self._clear_memory_cache()
        device = next(self.model.parameters()).device
        x = x.to(device)

        # 儲存每層 head 重要性分數
        head_importance_scores = [] if return_head_importance else None

        logits = self.model(x)                          # triggers hooks
        B, num_classes = logits.shape

        # target selection
        if target_class is None:
            target_idx = logits.argmax(dim=1)
        else:
            target_idx = torch.full((B,), int(target_class), device=device, dtype=torch.long)

        # Need encoder output tokens
        if self.last_tokens is None:
            return self._fallback_grad_relmap(x, target_idx, return_map)

        # classifier weights (PyTorch Linear: y = x @ W^T + b)
        # torchvision ViT 使用 model.heads.head
        try:
            W_cls = self.model.heads.head.weight.t().detach()   # [C, num_classes] in×out
            b_cls = self.model.heads.head.bias.detach()         # [num_classes]
        except AttributeError:
            print("Error: Cannot find model.heads.head, trying alternative structures...")
            # 嘗試其他可能的結構
            if hasattr(self.model, 'head'):
                W_cls = self.model.head.weight.t().detach()
                b_cls = self.model.head.bias.detach()
            else:
                print("Error: Cannot find classifier head in model")
                return self._fallback_grad_relmap(x, target_idx, return_map)
        h_cls = self.last_tokens[:, 0, :]                   # [B,C]

        # initialize relevance at the target logit
        R_out_cls = torch.zeros_like(logits)
        R_out_cls[torch.arange(B), target_idx] = logits[torch.arange(B), target_idx]
        
        # 檢查初始 relevance 是否為零
        if torch.all(R_out_cls == 0):
            # 如果 logits 都是 0（未初始化模型），使用均勻分布作為 relevance
            if torch.all(logits == 0):
                # 對於未初始化的模型，給每個類別分配相等的 relevance
                R_out_cls[torch.arange(B), target_idx] = 1.0 / num_classes
            # 如果 logits 都是負值，使用 softmax 機率
            elif logits.max() < 0:
                probs = torch.softmax(logits, dim=1)
                R_out_cls[torch.arange(B), target_idx] = probs[torch.arange(B), target_idx]

        # LRP back to CLS vector
        R_cls = lrp_linear(h_cls, W_cls, R_out_cls, b=b_cls, eps=self.eps)  # [B,C]
        R_tokens = torch.zeros_like(self.last_tokens)                       # [B,N,C]
        R_tokens[:, 0, :] = R_cls
        
        # Debug: 檢查 R_cls 的值
        print(f"Debug: R_cls shape: {R_cls.shape}, range: [{R_cls.min():.4f}, {R_cls.max():.4f}]")
        print(f"Debug: R_cls sum: {R_cls.sum():.4f}")

        # walk encoder blocks in reverse (attention branch only in MVP)
        for i, blk in enumerate(reversed(self.cache)):
            # 便利變數
            A    = blk['attn_probs']    # [B,H,N,N]
            W_o  = blk['W_o']           # [C,C]
            W_v  = blk['W_v']           # [C,C]
            X_in = blk['X_in']          # [B,N,C]  (ln1(x1))
            out  = blk['out']           # [B,N,C]  (attn_out)

            # ========= (I) 第二個殘差: x2 + mlp_out =========
            # 當前 R_tokens 是 block 輸出 y2 的關聯度，把它分給 x2 與 mlp_out
            ln2_in = blk['ln2_in']      # x2
            mlp_out = blk.get('fc2_out', None)
            if ln2_in is not None and mlp_out is not None:
                denom = (ln2_in.abs() + mlp_out.abs() + 1e-6)
                R_skip2   = R_tokens * (ln2_in.abs() / denom)     # 回到殘差支路 x2
                R_mlp_out = R_tokens * (mlp_out.abs() / denom)    # 回到 mlp 輸出
            else:
                R_skip2, R_mlp_out = R_tokens, None  # 若沒 hook 到，全部當作殘差

            # ========= (II) 回推 MLP: fc2 ← GELU ← fc1 =========
            # 先經 fc2 反推到 act_out（a=act_out, W=W_fc2, R_out=R_mlp_out）
            if R_mlp_out is not None and blk['W_fc2'] is not None and blk['act_out'] is not None:
                R_act = lrp_linear(blk['act_out'], blk['W_fc2'], R_mlp_out,
                                b=blk['b_fc2'], eps=self.eps)      # [B,N,Hid]
            else:
                R_act = None

            # GELU 先視為身份映射：R_fc1_out = R_act
            # R_fc1_out = R_act # pass through 感覺之後可以在 config 設定 true/false
            R_fc1_out = self.lrp_gelu_deeplift(blk['fc1_out'], R_act, eps=self.eps)


            # 再經 fc1 反推到 ln2(x2)（a=fc1_in, W=W_fc1, R_out=R_fc1_out）
            if R_fc1_out is not None and blk['W_fc1'] is not None and blk['fc1_in'] is not None:
                R_ln2_in = lrp_linear(blk['fc1_in'], blk['W_fc1'], R_fc1_out,
                                    b=blk['b_fc1'], eps=self.eps)   # [B,N,C]
            else:
                R_ln2_in = torch.zeros_like(R_skip2)

            # 把 MLP 支路回來的關聯度，加回第二殘差的「跳接」R_skip2
            R_after_mlp = R_skip2 + R_ln2_in  # 這是 y1 的關聯度（第一殘差之前）

            # ========= (III) 第一個殘差: x1 + attn_out =========
            x1 = blk['ln1_in']   # 殘差支路 x1
            if x1 is not None:
                denom1 = (x1.abs() + out.abs() + 1e-6)
                R_x1      = R_after_mlp * (x1.abs()  / denom1)   # 分給殘差 x1
                R_attnout = R_after_mlp * (out.abs() / denom1)   # 分給 attn 輸出
            else:
                R_x1, R_attnout = torch.zeros_like(R_after_mlp), R_after_mlp

            # ========= (IV) 回推 Attention: W_o ← A^T ← W_v =========
            # (1) out ← W_o：把 R_attnout 經 ε-LRP 回到 concat-heads
            R_concat = lrp_linear(out, W_o, R_attnout, b=None, eps=self.eps)   # [B,N,C]
            B_, N, C = R_concat.shape
            H = A.shape[1]
            d = C // H
            R_heads = R_concat.view(B_, N, H, d).permute(0, 2, 1, 3)           # [B,H,N,d]

            # (2) 用 A^T 路由：bhji · bhjd -> bhid
            R_V = torch.einsum('bhji,bhjd->bhid', A, R_heads)                   # [B,H,N,d]

            # (3) Partial：top-k 與 normalize（建議先 mask 再 normalize）
            # head_scores = R_V.abs().sum(dim=(2, 3))                             # [B,H]
            head_scores = torch.clamp(R_V, min=0).sum(dim=(2,3))                # [B,H]

            
            # 檢查並處理 NaN
            head_scores = torch.where(torch.isnan(head_scores), torch.zeros_like(head_scores), head_scores)
            
            # 儲存 head 重要性分數（在 mask 之前）
            if return_head_importance:
                # 計算當前層索引：reversed 第 i 個，對應原始層 (len(cache)-1-i)
                current_layer = len(self.cache) - 1 - i
                head_importance_scores.append({
                    'layer': current_layer,
                    'head_scores': head_scores.detach().cpu(),  # [B,H]
                    'topk_heads': self.topk_heads,
                    'head_weighting': self.head_weighting
                })
            
            if self.topk_heads is not None and self.topk_heads < H:
                idx  = head_scores.topk(self.topk_heads, dim=1).indices
                mask = torch.zeros_like(head_scores, dtype=R_V.dtype).scatter(1, idx, 1.0)
                R_V  = R_V * mask[:, :, None, None]
                head_scores = head_scores * mask  # // 建議：先 mask 再 normalize
            if self.head_weighting == 'normalize':
                w = head_scores / (head_scores.sum(dim=1, keepdim=True) + 1e-6)
                R_V = R_V * w[:, :, None, None]

            # (4) V ← W_v：把關聯度回到 ln1(x1)（先視 LN 為身份）
            R_V_concat  = R_V.permute(0, 2, 1, 3).contiguous().view(B_, N, C)   # [B,N,C]
            R_ln1_in    = lrp_linear(X_in, W_v, R_V_concat, b=None, eps=self.eps)

            # ========= (V) 匯總成該 block 輸入的關聯度 =========
            # 先視 LayerNorm 為身份：直接把 R_ln1_in 加到 R_x1
            R_tokens = R_x1 + R_ln1_in
            
            # 檢查並處理 NaN
            R_tokens = torch.where(torch.isnan(R_tokens), torch.zeros_like(R_tokens), R_tokens)

        # produce maps
        if return_map == 'token':
            result = R_tokens  # [B,N,C]  (includes CLS at index 0)
        else:
            # patch grid (sum over channel dim)
            # tokens: [CLS, p1, p2, ..., pM]
            grid_h = x.shape[-2] // self._patch_size()
            grid_w = x.shape[-1] // self._patch_size()
            
            # 使用正相關度聚合，避免正負抵銷導致熱圖失焦
            R_patch = R_tokens[:, 1:, :].clamp(min=0).sum(dim=2)        # [B, M]

            # # 嘗試 1：保留所有關聯度 (Positive + Negative)
            # R_patch = R_tokens[:, 1:, :].sum(dim=2)
            # # 這會顯示所有影響，但可能難以解釋。

            # # 嘗試 2：使用絕對值 (Absolute value)
            # R_patch = R_tokens[:, 1:, :].abs().sum(dim=2)
            # # 這顯示了哪些區域影響了決策（無論是支持還是反對）。
            
            if return_map == 'patch':
                result = R_patch
            elif return_map == 'image':
                R_grid = R_patch.view(B, grid_h, grid_w).unsqueeze(1)   # [B,1,gh,gw]
                R_img  = F.interpolate(R_grid, size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                result = R_img if upsample_to_input else R_grid.squeeze(1)
            else:
                # default: patch map
                result = R_patch

        # 回傳結果與 head 重要性（如果要求）
        if return_head_importance:
            return {
                'relevance_map': result,
                'head_importance': head_importance_scores,
                'target_class': target_idx.cpu().tolist() if target_class is None else target_class
            }
        else:
            return result

    @staticmethod
    def relu_like_safe_div(num, den, eps=1e-6):
        return num / (den.abs() + eps)

    @staticmethod
    def lrp_gelu_deeplift(x, R_out, eps=1e-6):
        # y = GELU(x) = x * Φ(x) 近似（或用 torch.nn.functional.gelu）
        if R_out is None:
            return None
        y = F.gelu(x)
        alpha = y / (x.abs() + eps)
        alpha = torch.where(x.abs() < 1e-6, torch.full_like(alpha, 0.5), alpha)
        return R_out * alpha

    # ---- helpers ----
    def _patch_size(self) -> int:
        # torchvision ViT uses conv_proj with kernel_size = stride = patch_size
        if hasattr(self.model, "conv_proj"):
            ks = self.model.conv_proj.kernel_size
            return ks[0] if isinstance(ks, tuple) else int(ks)
        return 16

    def _fallback_grad_relmap(self, x: torch.Tensor, target_idx: torch.Tensor, return_map: str):
        # simple gradient×input fallback (not true LRP) to avoid crash if hooks fail
        # 確保模型在訓練模式以允許梯度計算
        was_training = self.model.training
        self.model.train()
        
        # 確保輸入需要梯度
        x = x.detach().clone().requires_grad_(True)
        
        # 確保模型參數需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 前向傳播
        logits = self.model(x)
        sel = logits[torch.arange(x.shape[0]), target_idx]
        
        # 確保 sel 需要梯度
        if not sel.requires_grad:
            sel = sel.requires_grad_(True)
            
        # 清除之前的梯度
        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
            
        # 反向傳播
        sel.sum().backward()
        rel = (x.grad * x).abs().sum(dim=1)  # [B,H,W]
        
        # 恢復原始訓練狀態
        if not was_training:
            self.model.eval()
            
        if return_map == 'image':
            return rel
        # pack as patch-like if requested
        if return_map == 'patch':
            gh = x.shape[-2] // self._patch_size()
            gw = x.shape[-1] // self._patch_size()
            return F.adaptive_avg_pool2d(rel.unsqueeze(1), (gh, gw)).view(x.shape[0], gh*gw)
        return rel

    # convenience wrapper to match your earlier API
    def compute_lrp_relevance(self, x: torch.Tensor, target_class: Optional[int] = None,
                              return_intermediate: bool = False, return_map: str = 'patch',
                              upsample_to_input: bool = False, return_head_importance: bool = False) -> torch.Tensor:
        R = self.explain(x, target_class=target_class, return_map=return_map,
                         upsample_to_input=upsample_to_input, return_head_importance=return_head_importance)
        return {'final': R} if return_intermediate else R

    def analyze_head_importance(self, x: torch.Tensor, target_class: Optional[int] = None, 
                               save_plots: bool = True, save_dir: str = "./head_analysis") -> Dict:
        """
        分析並視覺化每層 head 的重要性
        Returns:
            Dict containing head importance analysis and plots
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        
        result = self.explain(x, target_class=target_class, return_map='patch', 
                             return_head_importance=True)
        
        if isinstance(result, dict):
            relevance_map = result['relevance_map']
            head_importance = result['head_importance']
            target_class = result['target_class']
        else:
            raise ValueError("Expected dict result with head_importance=True")
        
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
        
        analysis = {
            'target_class': target_class,
            'relevance_map': relevance_map,
            'head_analysis': []
        }
        
        # 分析每層的 head 重要性
        for layer_info in head_importance:
            layer_idx = layer_info['layer']
            head_scores = layer_info['head_scores']  # [B, H]
            topk_heads = layer_info['topk_heads']
            head_weighting = layer_info['head_weighting']
            
            # 計算統計資訊
            mean_scores = head_scores.mean(dim=0)  # [H]
            std_scores = head_scores.std(dim=0)    # [H]
            top_heads = mean_scores.topk(min(5, len(mean_scores))).indices.tolist()
            
            layer_analysis = {
                'layer': layer_idx,
                'num_heads': len(mean_scores),
                'mean_scores': mean_scores.tolist(),
                'std_scores': std_scores.tolist(),
                'top_heads': top_heads,
                'topk_heads': topk_heads,
                'head_weighting': head_weighting
            }
            
            # 繪製 head 重要性圖
            if save_plots:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # 左圖：head 重要性分數
                heads = range(len(mean_scores))
                ax1.bar(heads, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
                ax1.set_xlabel('Head Index')
                ax1.set_ylabel('Importance Score')
                ax1.set_title(f'Layer {layer_idx}: Head Importance Scores')
                ax1.grid(True, alpha=0.3)
                
                # 標示 top-k heads（如果有設定）
                if topk_heads is not None:
                    topk_indices = mean_scores.topk(topk_heads).indices
                    for idx in topk_indices:
                        ax1.bar(idx, mean_scores[idx], color='red', alpha=0.8)
                
                # 右圖：head 重要性排序
                sorted_indices = mean_scores.argsort(descending=True)
                sorted_scores = mean_scores[sorted_indices]
                ax2.bar(range(len(sorted_scores)), sorted_scores, alpha=0.7)
                ax2.set_xlabel('Head Rank')
                ax2.set_ylabel('Importance Score')
                ax2.set_title(f'Layer {layer_idx}: Head Importance Ranking')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/layer_{layer_idx}_head_importance.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            analysis['head_analysis'].append(layer_analysis)
        
        # 繪製跨層比較圖
        if save_plots and len(head_importance) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            layers = [info['layer'] for info in head_importance]
            max_heads = max([len(info['head_scores'].mean(dim=0)) for info in head_importance])
            
            # 為每層繪製 head 重要性
            for i, layer_info in enumerate(head_importance):
                mean_scores = layer_info['head_scores'].mean(dim=0)
                ax.plot(range(len(mean_scores)), mean_scores, 
                       marker='o', label=f'Layer {layer_info["layer"]}', alpha=0.7)
            
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Mean Importance Score')
            ax.set_title('Head Importance Across Layers')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/cross_layer_head_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        return analysis

    
    # --- new --- #

    def _hook_mlp_and_ln(self):  # // NEW
        # 建立從 block idx → cache 位置 的映射（用 block_idx 對齊）
        self._ln1_modules = []
        self._ln2_modules = []
        self._fc1_modules = []
        self._act_modules = []
        self._fc2_modules = []

        # 掃描 torchvision ViT 的結構，尋找每個 block 的 ln1/ln2 和 mlp.fc1/act/fc2
        for name, m in self.model.named_modules():
            # ln_1 / ln_2
            if name.endswith('.ln_1') and isinstance(m, nn.LayerNorm):
                m.register_forward_hook(self._ln1_hook)  # ln1 輸入是殘差分支 x1
                self._ln1_modules.append(m)
            if name.endswith('.ln_2') and isinstance(m, nn.LayerNorm):
                m.register_forward_hook(self._ln2_hook)  # ln2 輸入是殘差分支 x2
                self._ln2_modules.append(m)
            # MLP 兩層 & GELU
            if name.endswith('.mlp.fc1') and isinstance(m, nn.Linear):
                m.register_forward_hook(self._fc1_hook)
                self._fc1_modules.append(m)
            if name.endswith('.mlp.act') and isinstance(m, nn.GELU):
                m.register_forward_hook(self._act_hook)
                self._act_modules.append(m)
            if name.endswith('.mlp.fc2') and isinstance(m, nn.Linear):
                m.register_forward_hook(self._fc2_hook)
                self._fc2_modules.append(m)

    # def _find_latest_cache(self):  # // NEW：回到最近一次 attn 加入的那個 block
    #     assert len(self.cache) > 0, "cache empty before ln/mlp hook"
    #     return self.cache[-1]

    def _find_latest_cache(self):
        if len(self.cache) == 0:
            return None
        return self.cache[-1]

    def _clear_memory_cache(self):
        """清理所有累積的張量，避免記憶體洩漏"""
        # 清理 cache
        self.cache.clear()
        
        # 清理 pending ln1 列表
        self._pending_ln1.clear()
        
        # 清理 MHAWrapper 中保存的張量
        for wrapper in self.attn_wrappers:
            wrapper.saved_X_in = None
            wrapper.saved_out = None
            wrapper.saved_attn_probs = None
            wrapper.saved_V = None
        
        # 清理 last_tokens
        self.last_tokens = None
        
        # 強制垃圾回收
        import gc
        gc.collect()
        
        # 如果使用 CUDA，清理 CUDA 快取
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ln1_hook(self, module, inp, out):
        x1 = inp[0].detach()          # ln1 的輸入（尚未做 LN）
        self._pending_ln1.append(x1)  # 先排隊，等 MHA hook 來取

    def _ln2_hook(self, module, inp, out):
        blk = self._find_latest_cache()
        if blk is not None:
            blk['ln2_in'] = inp[0].detach()

    def _fc1_hook(self, module: nn.Linear, inp, out):
        blk = self._find_latest_cache()
        if blk is not None:
            blk['fc1_in'] = inp[0].detach()
            blk['fc1_out'] = out.detach()
            blk['W_fc1'] = module.weight.t().detach()
            blk['b_fc1'] = (module.bias.detach() if module.bias is not None else None)

    def _act_hook(self, module: nn.GELU, inp, out):
        blk = self._find_latest_cache()
        if blk is not None:
            blk['act_out'] = out.detach()

    def _fc2_hook(self, module: nn.Linear, inp, out):
        blk = self._find_latest_cache()
        if blk is not None:
            blk['fc2_out'] = out.detach()
            blk['W_fc2'] = module.weight.t().detach()
            blk['b_fc2'] = (module.bias.detach() if module.bias is not None else None)



# ---------------------------
# Quick self-test (optional)
# ---------------------------
if __name__ == "__main__":
    import torchvision.models as tv_models
    from torchvision.models.vision_transformer import ViT_B_16_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = None  # or ViT_B_16_Weights.IMAGENET1K_V1
    model = tv_models.vit_b_16(weights=weights).to(device).eval()

    analyzer = VIT_with_Partial_LRP(model, topk_heads=2, head_weighting='normalize', eps=1e-6)

    x = torch.randn(1, 3, 224, 224, device=device)
    R_patch = analyzer.compute_lrp_relevance(x, return_map='patch')     # [1, 196]
    print("Patch relevance:", R_patch.shape)

    R_img = analyzer.compute_lrp_relevance(x, return_map='image', upsample_to_input=True)  # [1,224,224]
    print("Image relevance:", R_img.shape)
