# file: partial_lrp_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


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
    if a.dim() == 2:
        a2 = a                              # [B,C_in]
        z  = a2 @ W + (b if b is not None else 0.0)   # [B,C_out]
        denom = z + eps * torch.sign(z)
        s  = R_out / denom                               # [B,C_out]
        c  = s @ W.t()                                   # [B,C_in]
        return a2 * c                                    # [B,C_in]

    elif a.dim() == 3:
        B, N, Cin = a.shape
        Cout = W.shape[1]
        a2 = a.reshape(-1, Cin)                          # [B*N, Cin]
        Rout2 = R_out.reshape(-1, Cout)                  # [B*N, Cout]
        z  = a2 @ W + (b if b is not None else 0.0)      # [B*N, Cout]
        denom = z + eps * torch.sign(z)
        s  = Rout2 / denom
        c  = s @ W.t()
        Rin = (a2 * c).view(B, N, Cin)
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
        self.W_o = mha.out_proj.weight.t()                        # [C,C]
        W_qkv = mha.in_proj_weight                                # [3C, C] (q,k,v as out×in)
        C = W_qkv.shape[1]
        self.W_v = W_qkv[2*C:3*C, :].t()                          # [in=C, out=C]
        self.b_v = (mha.in_proj_bias[2*C:3*C]
                    if mha.in_proj_bias is not None else None)    # [C] or None

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
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,   # 要 per-head
            is_causal=is_causal,
        )

        # 保存中間量
        self.saved_X_in = query                          # [B,N,C]
        self.saved_out  = out                            # [B,N,C]
        self.saved_attn_probs = attn_w.detach()          # [B,H,N,N]

        # V = X @ W_v + b_v，並 reshape 成 [B,H,N,d]
        v = query @ self.W_v + (self.b_v if self.b_v is not None else 0.0)  # [B,N,C]
        H = self.mha.num_heads
        d = v.shape[-1] // H
        self.saved_V = v.view(v.shape[0], v.shape[1], H, d).permute(0, 2, 1, 3).detach()

        # torchvision ViT 會做 x, _ = self_attention(..., need_weights=False)
        # 因此這裡無論 need_weights 為何都回傳 (out, weights/None) 的 tuple
        return out, (attn_w if need_weights else None)


# ---------------------------
# Analyzer: Partial LRP for torchvision ViT
# ---------------------------
class VIT_with_Partial_LRP:
    """
    Minimal working Partial LRP for torchvision ViT (vit_b_16, etc.).
    - Performs head-level relevance in attention (Split -> A^T -> partial -> W^V back to X)
    - MLP & residual/LN are simplified (not propagated in this MVP).
    """
    def __init__(self, vit_model: nn.Module, topk_heads: Optional[int] = None,
                 head_weighting: str = 'normalize', eps: float = 1e-6):
        self.model = vit_model.eval()
        self.eps = eps
        self.topk_heads = topk_heads
        self.head_weighting = head_weighting

        self.cache: List[Dict[str, Any]] = []
        self._wrap_attentions()
        self._hook_tokens()
        self._hook_mlp_and_ln()  # // NEW
        self._block_counter = 0  # // NEW：追蹤目前是第幾個 block（由 attn hook 推進）
        self._pending_ln1 = []   # ← NEW: ln1 的輸入先暫存這裡（FIFO）

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

    # def _attn_forward_hook(self, module: MHAWrapper, inputs, output):
    #     self._block_counter += 1  # // NEW：每遇到一個 attention 就 +1，視為一個 block
    #     self.cache.append(dict(
    #         block_idx=self._block_counter,          # // NEW
    #         attn_probs=module.saved_attn_probs,     # [B,H,N,N]
    #         V=module.saved_V,                       # [B,H,N,d]
    #         W_o=module.W_o,                         # [C,C]
    #         W_v=module.W_v,                         # [C,C]
    #         X_in=module.saved_X_in,                 # [B,N,C] (這是 ln1(x1) 後的 token)
    #         out=module.saved_out,                   # [B,N,C] (attn 輸出)
    #         # 下列欄位稍後由其他 hook 填充：
    #         ln1_in=None,    # [B,N,C]  （殘差分支 x1）
    #         ln2_in=None,    # [B,N,C]  （殘差分支 x2）
    #         fc1_in=None,    # [B,N,C]  （= ln2(x2)）
    #         fc1_out=None,   # [B,N,hidden]
    #         act_out=None,   # [B,N,hidden] GELU 輸出
    #         fc2_out=None,   # [B,N,C] (mlp 輸出)
    #         W_fc1=None, b_fc1=None,
    #         W_fc2=None, b_fc2=None,
    #     ))

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

    # ---------------------------
    # Public API
    # ---------------------------
    @torch.no_grad()
    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                return_map: str = 'patch', upsample_to_input: bool = False) -> torch.Tensor:
        """
        x: [B,3,H,W]
        return_map: 'patch' -> [B, grid_h*grid_w], 'token' -> [B,N,C], 'image' -> [B, H, W]
        """
        self.cache.clear()
        device = next(self.model.parameters()).device
        x = x.to(device)

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
        W_cls = self.model.heads.head.weight.t().detach()   # [C, num_classes] in×out
        b_cls = self.model.heads.head.bias.detach()         # [num_classes]
        h_cls = self.last_tokens[:, 0, :]                   # [B,C]

        # initialize relevance at the target logit
        R_out_cls = torch.zeros_like(logits)
        R_out_cls[torch.arange(B), target_idx] = logits[torch.arange(B), target_idx]

        # LRP back to CLS vector
        R_cls = lrp_linear(h_cls, W_cls, R_out_cls, b=b_cls, eps=self.eps)  # [B,C]
        R_tokens = torch.zeros_like(self.last_tokens)                       # [B,N,C]
        R_tokens[:, 0, :] = R_cls

        # walk encoder blocks in reverse (attention branch only in MVP)
        for blk in reversed(self.cache):
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
            R_fc1_out = R_act

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
            head_scores = R_V.abs().sum(dim=(2, 3))                             # [B,H]
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

        # produce maps
        if return_map == 'token':
            return R_tokens  # [B,N,C]  (includes CLS at index 0)

        # patch grid (sum over channel dim)
        # tokens: [CLS, p1, p2, ..., pM]
        grid_h = x.shape[-2] // self._patch_size()
        grid_w = x.shape[-1] // self._patch_size()
        R_patch = R_tokens[:, 1:, :].sum(dim=2)                     # [B, M]
        if return_map == 'patch':
            return R_patch

        if return_map == 'image':
            R_grid = R_patch.view(B, grid_h, grid_w).unsqueeze(1)   # [B,1,gh,gw]
            R_img  = F.interpolate(R_grid, size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            return R_img if upsample_to_input else R_grid.squeeze(1)

        # default: patch map
        return R_patch

    # ---- helpers ----
    def _patch_size(self) -> int:
        # torchvision ViT uses conv_proj with kernel_size = stride = patch_size
        if hasattr(self.model, "conv_proj"):
            ks = self.model.conv_proj.kernel_size
            return ks[0] if isinstance(ks, tuple) else int(ks)
        return 16

    def _fallback_grad_relmap(self, x: torch.Tensor, target_idx: torch.Tensor, return_map: str):
        # simple gradient×input fallback (not true LRP) to avoid crash if hooks fail
        x = x.detach().clone().requires_grad_(True)
        logits = self.model(x)
        sel = logits[torch.arange(x.shape[0]), target_idx]
        self.model.zero_grad(set_to_none=True)
        sel.sum().backward()
        rel = (x.grad * x).abs().sum(dim=1)  # [B,H,W]
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
                              upsample_to_input: bool = False) -> torch.Tensor:
        R = self.explain(x, target_class=target_class, return_map=return_map,
                         upsample_to_input=upsample_to_input)
        return {'final': R} if return_intermediate else R

    
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
