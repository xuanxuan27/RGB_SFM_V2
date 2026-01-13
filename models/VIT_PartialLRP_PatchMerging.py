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

        # a2 = a.view(-1, Cin)
        # Rout2 = R_out.view(-1, Cout)
        # Rin = _core(a2, Rout2).view(B, N, Cin)

        a2 = a.reshape(-1, Cin)
        Rout2 = R_out.reshape(-1, Cout)
        Rin = _core(a2, Rout2).reshape(B, N, Cin)
        return Rin
    else:
        raise ValueError("a must be [B,C] or [B,N,C]")


# ---------------------------
# Patch Merging Module (類似 PVT 的 OverlapPatchEmbed)
# ---------------------------
class PatchMerging(nn.Module):
    """
    類似 PVT 的 patch merging，用於在 ViT 層之間進行下採樣。
    輸入: (B, N, C) token 序列
    輸出: (B, N', C') token 序列，其中 N' < N（下採樣）
    """
    def __init__(self, in_dim: int, out_dim: int, 
                 kernel_size: int = 7, stride: int = 4, padding: int = 3,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 使用 Conv2d 進行 patch merging
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.norm = norm_layer if norm_layer is not None else nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, N, C) token 序列，其中 N = H * W
            H, W: 空間維度（用於 reshape）
        Returns:
            out: (B, N', C') token 序列
            H', W': 新的空間維度
        """
        B, N, C = x.shape
        
        # 將 token 序列轉換為 feature map: (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 進行 patch merging (下採樣)
        x = self.proj(x)  # (B, out_dim, H', W')
        
        # 獲取新的空間維度
        H_new, W_new = x.shape[-2:]
        N_new = H_new * W_new
        
        # 轉回 token 序列: (B, out_dim, H', W') -> (B, N', out_dim)
        x = x.flatten(2).transpose(1, 2)  # (B, N', out_dim)
        
        # LayerNorm
        x = self.norm(x)
        
        return x, H_new, W_new


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
        # Optional head mask: None or tensor shaped [H] or [B,H] with {0,1} values
        self.head_mask: Optional[torch.Tensor] = None

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

        # 若啟用 head masking，改以 attn_w 與 saved_V 重建輸出（只在推論時生效）
        if self.head_mask is not None:
            B = query.shape[0]
            # head_mask: [H] 或 [B,H] -> 轉成 [B,H,1,1]
            if self.head_mask.dim() == 1:
                mask = self.head_mask.view(1, H, 1, 1).to(query.device)
                mask = mask.expand(B, H, 1, 1)
            elif self.head_mask.dim() == 2:
                mask = self.head_mask.view(B, H, 1, 1).to(query.device)
            else:
                raise ValueError("head_mask must be shape [H] or [B,H]")

            # per-head output: [B,H,N,d] = einsum('bhij,bhjd->bhid', attn, V)
            head_out = torch.einsum('bhij,bhjd->bhid', attn_w, self.saved_V.to(query.device))
            head_out = head_out * mask
            # concat heads -> [B,N,C]
            concat = head_out.permute(0, 2, 1, 3).contiguous().view(B, head_out.shape[2], H * d)
            # out proj（使用原 mha 的 out_proj）
            out_re = concat @ self.mha.out_proj.weight.t() + self.mha.out_proj.bias
            out = out_re
            self.saved_out = out.detach()

        # torchvision ViT 會做 x, _ = self_attention(..., need_weights=False)
        # 因此這裡無論 need_weights 為何都回傳 (out, weights/None) 的 tuple
        return out, (attn_w if need_weights else None)


# ---------------------------
# Analyzer: Partial LRP for torchvision ViT
# ---------------------------
class VIT_PartialLRP_PatchMerging(nn.Module):
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
                 eps: float = 1e-6,
                 # Patch Merging 相關參數
                 enable_patch_merging: bool = True,  # 預設啟用
                 patch_merging_layers: Optional[List[int]] = None,  # 預設為 [3, 6, 9]（每 3 層一次）
                 patch_merging_kernel_size: int = 3,
                 patch_merging_stride: int = 2,
                 patch_merging_padding: int = 1):
        super().__init__()
        self.eps = eps
        self.topk_heads = topk_heads
        self.head_weighting = head_weighting
        self.input_size = input_size
        
        # Patch Merging 設定
        self.enable_patch_merging = enable_patch_merging
        # 預設 merge 位置：對於 12 層的 vit_b_16，建議更保守的 merge 策略
        # 選項 1: [4, 8] - 只 merge 2 次，保持更多空間信息
        #   - 早期層（0-4）：保持高解析度 14×14，學習低層特徵
        #   - 中期層（5-8）：第一次下採樣到 7×7，學習中層特徵
        #   - 後期層（9-11）：第二次下採樣到 3×3，學習高層特徵（仍有足夠信息）
        # 選項 2: [6] - 只 merge 1 次，更保守
        #   - 早期層（0-6）：保持高解析度 14×14
        #   - 後期層（7-11）：下採樣到 7×7
        if patch_merging_layers is None:
            # 預設：只在第 4、8 層之後 merge（更保守，避免過度下採樣）
            # 這樣可以保持足夠的空間信息用於後續層的處理
            self.patch_merging_layers = [4, 8]
        else:
            self.patch_merging_layers = patch_merging_layers
        self.patch_merging_kernel_size = patch_merging_kernel_size
        self.patch_merging_stride = patch_merging_stride
        self.patch_merging_padding = patch_merging_padding

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

        # 初始化 Patch Merging 模組（如果啟用）
        self.patch_merging_modules = nn.ModuleDict()
        if self.enable_patch_merging and len(self.patch_merging_layers) > 0:
            self._init_patch_merging()

        self.cache: List[Dict[str, Any]] = []
        self.layer_activations: List[torch.Tensor] = []  # 儲存每層的 activation
        self._wrap_attentions()
        self._hook_tokens()
        self._hook_mlp_and_ln()  # // NEW
        self._hook_layer_activations()  # Hook 每層的 activation
        self._block_counter = 0  # // NEW：追蹤目前是第幾個 block（由 attn hook 推進）
        self._pending_ln1 = []   # ← NEW: ln1 的輸入先暫存這裡（FIFO）
        
        # 追蹤當前的空間維度（用於 patch merging）
        self.current_H = None
        self.current_W = None
        
        # 用於在 patch merging forward 中保存每層的 activation（patch merging 之後）
        self.layer_activations_with_merging = []

    def _init_patch_merging(self):
        """初始化 Patch Merging 模組"""
        # 獲取 encoder 的層數和 embed_dim
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            num_layers = len(self.model.encoder.layers)
            # 獲取 embed_dim（從第一個 encoder layer 的 attention 取得）
            if num_layers > 0:
                first_layer = self.model.encoder.layers[0]
                if hasattr(first_layer, 'self_attention'):
                    embed_dim = first_layer.self_attention.in_proj_weight.shape[1]
                elif hasattr(first_layer, 'ln_1'):
                    embed_dim = first_layer.ln_1.normalized_shape[0]
                else:
                    # 嘗試從 patch embedding 取得
                    if hasattr(self.model, 'conv_proj'):
                        embed_dim = self.model.conv_proj.out_channels
                    else:
                        embed_dim = 768  # 預設值
            else:
                embed_dim = 768
        else:
            num_layers = 12  # 預設值
            embed_dim = 768
        
        # 為每個指定的層位置創建 patch merging 模組
        for layer_idx in self.patch_merging_layers:
            if 0 <= layer_idx < num_layers:
                # 計算輸出維度（可以保持相同或增加）
                out_dim = embed_dim  # 可以改為 embed_dim * 2 來增加通道數
                module_name = f'merge_after_layer_{layer_idx}'
                self.patch_merging_modules[module_name] = PatchMerging(
                    in_dim=embed_dim,
                    out_dim=out_dim,
                    kernel_size=self.patch_merging_kernel_size,
                    stride=self.patch_merging_stride,
                    padding=self.patch_merging_padding
                )
    
    def _get_patch_grid_size(self, num_patches: int) -> Tuple[int, int]:
        """從 patch 數量推斷空間維度"""
        # 假設是正方形
        grid_size = int(num_patches ** 0.5)
        return grid_size, grid_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # 如果啟用 patch merging，需要自定義 forward
        if self.enable_patch_merging and len(self.patch_merging_modules) > 0:
            return self._forward_with_patch_merging(x)
        else:
            return self.model(x)
    
    def _forward_with_patch_merging(self, x: torch.Tensor) -> torch.Tensor:
        """
        帶有 patch merging 的自定義 forward
        在指定的層之間插入 patch merging 操作
        """
        # 獲取 encoder
        if not hasattr(self.model, 'encoder'):
            return self.model(x)
        
        encoder = self.model.encoder
        
        # Patch embedding
        if hasattr(self.model, 'conv_proj'):
            x = self.model.conv_proj(x)  # (B, C, H, W)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
            if hasattr(self.model, 'class_token'):
                # 添加 class token
                cls_token = self.model.class_token.expand(B, -1, -1)  # (B, 1, C)
                x = torch.cat([cls_token, x], dim=1)  # (B, 1+N, C)
            if hasattr(self.model, 'pos_embed'):
                x = x + self.model.pos_embed
        else:
            # 使用 patch_embed（其他 ViT 變體）
            x, H, W = self.model.patch_embed(x)
            if hasattr(self.model, 'pos_embed'):
                x = x + self.model.pos_embed
        
        # 更新當前空間維度
        if hasattr(self.model, 'conv_proj'):
            self.current_H = H
            self.current_W = W
        else:
            # 從 patch 數量推斷
            num_patches = x.shape[1] - 1  # 減去 CLS token
            self.current_H, self.current_W = self._get_patch_grid_size(num_patches)
        
        # 清空 patch merging 專用的 activation 列表
        self.layer_activations_with_merging = []
        
        # 通過 encoder layers
        for layer_idx, layer in enumerate(encoder.layers):
            # 通過當前層
            x = layer(x)
            
            # 檢查是否需要在這一層之後進行 patch merging
            merge_key = f'merge_after_layer_{layer_idx}'
            if merge_key in self.patch_merging_modules:
                # 分離 CLS token 和 patch tokens
                if x.shape[1] > 1:  # 有 CLS token
                    cls_token = x[:, 0:1, :]  # (B, 1, C)
                    patch_tokens = x[:, 1:, :]  # (B, N, C)
                    
                    # 對 patch tokens 進行 patch merging
                    merged_patches, H_new, W_new = self.patch_merging_modules[merge_key](
                        patch_tokens, self.current_H, self.current_W
                    )
                    
                    # 合併回 CLS token
                    x = torch.cat([cls_token, merged_patches], dim=1)  # (B, 1+N', C')
                    
                    # 更新空間維度
                    self.current_H = H_new
                    self.current_W = W_new
                else:
                    # 沒有 CLS token，直接對所有 tokens 進行 patch merging
                    x, H_new, W_new = self.patch_merging_modules[merge_key](
                        x, self.current_H, self.current_W
                    )
                    self.current_H = H_new
                    self.current_W = W_new
            
            # 保存這一層的 activation（patch merging 之後的結果）
            self.layer_activations_with_merging.append(x.detach())
        
        # 通過最後的 LayerNorm
        if hasattr(encoder, 'ln'):
            x = encoder.ln(x)
        elif hasattr(encoder, 'norm'):
            x = encoder.norm(x)
        
        # 分類頭
        if hasattr(self.model, 'heads'):
            x = self.model.heads(x)
        elif hasattr(self.model, 'head'):
            x = self.model.head(x)
        
        return x
    
    # ---- Head masking control (affects forward only) ----
    def clear_head_mask(self):
        """
        清除所有注意力層的 head mask（恢復原始 forward 行為）
        """
        for w in getattr(self, 'attn_wrappers', []):
            w.head_mask = None

    def set_head_mask(self, layer_to_heads: Dict[int, List[int]], num_heads: Optional[int] = None,
                      per_batch: Optional[int] = None, keep_only: bool = True):
        """
        設定每層要保留的 head 清單，其他 head 將被遮蔽為 0（只影響 forward，不影響 LRP 與 explain）
        Args:
            layer_to_heads: {layer_idx: [h1, h2, ...]}
            num_heads: 若提供，則自動建立 [H] 向量；未提供則從對應 wrapper 推斷
            per_batch: 若提供，將建立 [B,H] 面罩（目前保留接口，預設使用 [H]）
            keep_only: True 時僅保留指定 heads；False 時移除指定 heads、保留其他 heads
        """
        for layer_idx, wrapper in enumerate(getattr(self, 'attn_wrappers', [])):
            H = num_heads if num_heads is not None else wrapper.mha.num_heads
            if keep_only:
                mask = torch.zeros(H, dtype=torch.float32)
                if layer_idx in layer_to_heads and len(layer_to_heads[layer_idx]) > 0:
                    keep = [h for h in layer_to_heads[layer_idx] if 0 <= int(h) < H]
                    if len(keep) > 0:
                        mask[torch.tensor(keep, dtype=torch.long)] = 1.0
                else:
                    mask = torch.ones(H, dtype=torch.float32)
            else:
                mask = torch.ones(H, dtype=torch.float32)
                if layer_idx in layer_to_heads and len(layer_to_heads[layer_idx]) > 0:
                    remove = [h for h in layer_to_heads[layer_idx] if 0 <= int(h) < H]
                    if len(remove) > 0:
                        mask[torch.tensor(remove, dtype=torch.long)] = 0.0
            wrapper.head_mask = mask

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
        head_module = None
        if hasattr(self.model, 'heads') and hasattr(self.model.heads, 'head'):
            head_module = self.model.heads.head
        elif hasattr(self.model, 'head'):
            head_module = self.model.head
        if head_module is None:
            print("Error: Cannot find classifier head in model")
            return self._fallback_grad_relmap(x, target_idx, return_map)

        W_cls = head_module.weight.t().detach()
        b_cls = head_module.bias.detach()
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
        # print(f"Debug: R_cls shape: {R_cls.shape}, range: [{R_cls.min():.4f}, {R_cls.max():.4f}]")
        # print(f"Debug: R_cls sum: {R_cls.sum():.4f}")

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
            # Try to get patch_shape from model first (more reliable)
            if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_shape"):
                patch_shape = self.model.patch_embed.patch_shape
                if isinstance(patch_shape, (tuple, list)) and len(patch_shape) == 2:
                    grid_h, grid_w = int(patch_shape[0]), int(patch_shape[1])
                else:
                    grid_h = grid_w = int(patch_shape) if isinstance(patch_shape, (int, float)) else None
            else:
                grid_h = grid_w = None
            
            # Fallback to calculating from input size and patch_size
            if grid_h is None or grid_w is None:
                patch_size = self._patch_size()
                grid_h = x.shape[-2] // patch_size
                grid_w = x.shape[-1] // patch_size
            
            # # 使用正相關度聚合，避免正負抵銷導致熱圖失焦
            R_patch = R_tokens[:, 1:, :].clamp(min=0).sum(dim=2)        # [B, M]

            # 嘗試 1：保留所有關聯度 (Positive + Negative)
            # R_patch = R_tokens[:, 1:, :].sum(dim=2)
            # 這會顯示所有影響，但可能難以解釋。

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
        # VIT_SmallPatch uses patch_embed.proj
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "proj"):
            ks = self.model.patch_embed.proj.kernel_size
            return ks[0] if isinstance(ks, tuple) else int(ks)
        # VIT_SmallPatch also has patch_size attribute
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            return ps if isinstance(ps, int) else (ps[0] if isinstance(ps, (tuple, list)) else int(ps))
        # VIT_SmallPatch model may have patch_size directly
        if hasattr(self.model, "patch_size"):
            ps = self.model.patch_size
            return ps if isinstance(ps, int) else (ps[0] if isinstance(ps, (tuple, list)) else int(ps))
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

    def explain_per_layer(self, x: torch.Tensor, target_class: Optional[int] = None,
                          return_map: str = 'patch', upsample_to_input: bool = False) -> Dict:
        """
        計算每一層的 relevance map
        Returns:
            Dict containing relevance maps for each layer
        """
        # 清理所有累積的張量，避免記憶體洩漏
        self._clear_memory_cache()
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
            print("Warning: last_tokens is None, using fallback method")
            fallback_result = self._fallback_grad_relmap(x, target_idx, return_map)
            return {'layer_0': fallback_result}

        # classifier weights (PyTorch Linear: y = x @ W^T + b)
        try:
            W_cls = self.model.heads.head.weight.t().detach()   # [C, num_classes] in×out
            b_cls = self.model.heads.head.bias.detach()         # [num_classes]
        except AttributeError:
            print("Error: Cannot find model.heads.head, trying alternative structures...")
            if hasattr(self.model, 'head'):
                W_cls = self.model.head.weight.t().detach()
                b_cls = self.model.head.bias.detach()
            else:
                print("Error: Cannot find classifier head in model")
                fallback_result = self._fallback_grad_relmap(x, target_idx, return_map)
                return {'layer_0': fallback_result}
        
        h_cls = self.last_tokens[:, 0, :]                   # [B,C]

        # initialize relevance at the target logit
        R_out_cls = torch.zeros_like(logits)
        R_out_cls[torch.arange(B), target_idx] = logits[torch.arange(B), target_idx]
        
        # 檢查初始 relevance 是否為零
        if torch.all(R_out_cls == 0):
            if torch.all(logits == 0):
                R_out_cls[torch.arange(B), target_idx] = 1.0 / num_classes
            elif logits.max() < 0:
                probs = torch.softmax(logits, dim=1)
                R_out_cls[torch.arange(B), target_idx] = probs[torch.arange(B), target_idx]

        # LRP back to CLS vector
        R_cls = lrp_linear(h_cls, W_cls, R_out_cls, b=b_cls, eps=self.eps)  # [B,C]
        R_tokens = torch.zeros_like(self.last_tokens)                       # [B,N,C]
        R_tokens[:, 0, :] = R_cls

        # 儲存每一層的結果
        layer_results = {}
        
        # 先儲存初始狀態（分類器層）
        if return_map == 'token':
            layer_results['classifier'] = R_tokens.clone()
        else:
            # patch grid (sum over channel dim)
            grid_h = x.shape[-2] // self._patch_size()
            grid_w = x.shape[-1] // self._patch_size()
            # R_patch = R_tokens[:, 1:, :].clamp(min=0).sum(dim=2)        # [B, M]
            R_patch = R_tokens[:, 1:, :].sum(dim=2)
            if return_map == 'patch':
                layer_results['classifier'] = R_patch.clone()
            elif return_map == 'image':
                R_grid = R_patch.view(B, grid_h, grid_w).unsqueeze(1)   # [B,1,gh,gw]
                R_img  = F.interpolate(R_grid, size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                layer_results['classifier'] = R_img.clone() if upsample_to_input else R_grid.squeeze(1).clone()

        # walk encoder blocks in reverse (attention branch only in MVP)
        for i, blk in enumerate(reversed(self.cache)):
            # 便利變數
            A    = blk['attn_probs']    # [B,H,N,N]
            W_o  = blk['W_o']           # [C,C]
            W_v  = blk['W_v']           # [C,C]
            X_in = blk['X_in']          # [B,N,C]  (ln1(x1))
            out  = blk['out']           # [B,N,C]  (attn_out)

            # ========= (I) 第二個殘差: x2 + mlp_out =========
            ln2_in = blk['ln2_in']      # x2
            mlp_out = blk.get('fc2_out', None)
            if ln2_in is not None and mlp_out is not None:
                denom = (ln2_in.abs() + mlp_out.abs() + 1e-6)
                R_skip2   = R_tokens * (ln2_in.abs() / denom)     # 回到殘差支路 x2
                R_mlp_out = R_tokens * (mlp_out.abs() / denom)    # 回到 mlp 輸出
            else:
                R_skip2, R_mlp_out = R_tokens, None

            # ========= (II) 回推 MLP: fc2 ← GELU ← fc1 =========
            if R_mlp_out is not None and blk['W_fc2'] is not None and blk['act_out'] is not None:
                R_act = lrp_linear(blk['act_out'], blk['W_fc2'], R_mlp_out,
                                b=blk['b_fc2'], eps=self.eps)      # [B,N,Hid]
            else:
                R_act = None

            R_fc1_out = self.lrp_gelu_deeplift(blk['fc1_out'], R_act, eps=self.eps)

            if R_fc1_out is not None and blk['W_fc1'] is not None and blk['fc1_in'] is not None:
                R_ln2_in = lrp_linear(blk['fc1_in'], blk['W_fc1'], R_fc1_out,
                                    b=blk['b_fc1'], eps=self.eps)   # [B,N,C]
            else:
                R_ln2_in = torch.zeros_like(R_skip2)

            R_after_mlp = R_skip2 + R_ln2_in

            # ========= (III) 第一個殘差: x1 + attn_out =========
            x1 = blk['ln1_in']   # 殘差支路 x1
            if x1 is not None:
                denom1 = (x1.abs() + out.abs() + 1e-6)
                R_x1      = R_after_mlp * (x1.abs()  / denom1)   # 分給殘差 x1
                R_attnout = R_after_mlp * (out.abs() / denom1)   # 分給 attn 輸出
            else:
                R_x1, R_attnout = torch.zeros_like(R_after_mlp), R_after_mlp

            # ========= (IV) 回推 Attention: W_o ← A^T ← W_v =========
            R_concat = lrp_linear(out, W_o, R_attnout, b=None, eps=self.eps)   # [B,N,C]
            B_, N, C = R_concat.shape
            H = A.shape[1]
            d = C // H
            R_heads = R_concat.view(B_, N, H, d).permute(0, 2, 1, 3)           # [B,H,N,d]

            R_V = torch.einsum('bhji,bhjd->bhid', A, R_heads)                   # [B,H,N,d]

            head_scores = torch.clamp(R_V, min=0).sum(dim=(2,3))                # [B,H]
            head_scores = torch.where(torch.isnan(head_scores), torch.zeros_like(head_scores), head_scores)
            
            if self.topk_heads is not None and self.topk_heads < H:
                idx  = head_scores.topk(self.topk_heads, dim=1).indices
                mask = torch.zeros_like(head_scores, dtype=R_V.dtype).scatter(1, idx, 1.0)
                R_V  = R_V * mask[:, :, None, None]
                head_scores = head_scores * mask
            if self.head_weighting == 'normalize':
                w = head_scores / (head_scores.sum(dim=1, keepdim=True) + 1e-6)
                R_V = R_V * w[:, :, None, None]

            R_V_concat  = R_V.permute(0, 2, 1, 3).contiguous().view(B_, N, C)   # [B,N,C]
            R_ln1_in    = lrp_linear(X_in, W_v, R_V_concat, b=None, eps=self.eps)

            # ========= (V) 匯總成該 block 輸入的關聯度 =========
            R_tokens = R_x1 + R_ln1_in
            R_tokens = torch.where(torch.isnan(R_tokens), torch.zeros_like(R_tokens), R_tokens)

            # 儲存當前層的結果
            current_layer = len(self.cache) - 1 - i
            
            if return_map == 'token':
                layer_results[f'layer_{current_layer}'] = R_tokens.clone()
            else:
                # patch grid (sum over channel dim)
                grid_h = x.shape[-2] // self._patch_size()
                grid_w = x.shape[-1] // self._patch_size()
                # R_patch = R_tokens[:, 1:, :].clamp(min=0).sum(dim=2)        # [B, M]
                R_patch = R_tokens[:, 1:, :].sum(dim=2)
                if return_map == 'patch':
                    layer_results[f'layer_{current_layer}'] = R_patch.clone()
                elif return_map == 'image':
                    R_grid = R_patch.view(B, grid_h, grid_w).unsqueeze(1)   # [B,1,gh,gw]
                    R_img  = F.interpolate(R_grid, size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                    layer_results[f'layer_{current_layer}'] = R_img.clone() if upsample_to_input else R_grid.squeeze(1).clone()

        return layer_results

    def get_per_layer_activations(self, x: torch.Tensor, return_map: str = 'image', 
                                   upsample_to_input: bool = False) -> Dict:
        """
        獲取每層的 activation value（不是 relevance）
        Args:
            x: [B, 3, H, W] 輸入圖像
            return_map: 'token' -> [B,N,C], 'patch' -> [B, num_patches], 'image' -> [B, H, W]
            upsample_to_input: 是否上採樣到輸入圖像大小
        Returns:
            Dict: {layer_name: activation_map}
        """
        self._clear_memory_cache()
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        # Forward pass 觸發 hooks
        # 使用 self(x) 而不是 self.model(x)，這樣才會走 forward 中的 patch merging 邏輯
        _ = self(x)
        
        # 獲取每層的 activation
        layer_results = {}
        
        # 如果啟用了 patch merging，使用 patch merging 之後的 activation
        # 否則使用標準的 layer_activations
        if self.enable_patch_merging and len(self.patch_merging_modules) > 0:
            activations_to_use = self.layer_activations_with_merging
            if len(activations_to_use) == 0:
                # 如果 patch merging 的 activation 列表為空，回退到標準的
                print(f"Warning: layer_activations_with_merging is empty, using layer_activations instead")
                activations_to_use = self.layer_activations
            else:
                # 調試：打印每層的 token 數量
                print(f"Using patch merging activations. Layer token counts:")
                for i, act in enumerate(activations_to_use):
                    num_tokens = act.shape[1] - 1  # 減去 CLS token
                    print(f"  Layer {i}: {num_tokens} patches")
        else:
            activations_to_use = self.layer_activations
        
        # 第一層是 input relevance（需要單獨計算）
        # 這裡我們先處理 encoder layers 的 activation
        for i, act in enumerate(activations_to_use):
            # act shape: [B, N, C]
            layer_name = f'layer_{i}'
            
            if return_map == 'token':
                layer_results[layer_name] = act.clone()
            else:
                # 計算 patch-level activation（對 channel 維度做聚合）
                # 原本方法：使用 L2 norm
                # act_patch = act[:, 1:, :].norm(dim=2)  # [B, num_patches] 排除 CLS token
                
                # 新方法：絕對值再取平均
                act_patch = act[:, 1:, :].abs().mean(dim=2)  # [B, num_patches] 排除 CLS token
                
                if return_map == 'patch':
                    layer_results[layer_name] = act_patch
                elif return_map == 'image':
                    # 轉換成 image map
                    # 根據實際的 patch 數量計算空間維度（支援 patch merging）
                    num_patches = act_patch.shape[1]  # [B, num_patches]
                    # 假設是正方形（對於 ViT 通常如此）
                    grid_size = int(num_patches ** 0.5)
                    # 如果不是完全平方數，使用最接近的整數
                    if grid_size * grid_size != num_patches:
                        # 嘗試找到最接近的因數分解
                        best_diff = float('inf')
                        best_h, best_w = grid_size, grid_size
                        for h in range(1, int(num_patches ** 0.5) + 1):
                            if num_patches % h == 0:
                                w = num_patches // h
                                diff = abs(h - w)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_h, best_w = h, w
                        grid_h, grid_w = best_h, best_w
                    else:
                        grid_h, grid_w = grid_size, grid_size
                    
                    act_grid = act_patch.view(x.shape[0], grid_h, grid_w).unsqueeze(1)  # [B, 1, gh, gw]
                    if upsample_to_input:
                        act_img = F.interpolate(act_grid, size=x.shape[-2:], 
                                              mode='bilinear', align_corners=False).squeeze(1)
                        layer_results[layer_name] = act_img
                    else:
                        layer_results[layer_name] = act_grid.squeeze(1)
        
        return layer_results

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

    def build_head_mask_from_importance(self, head_importance, topk: int = 2) -> Dict[int, List[int]]:
        """
        根據 analyze_head_importance 的結果，自動挑選最重要 head
        並生成 {layer: [heads]} 格式
        
        Args:
            head_importance: analyze_head_importance 返回的結果字典
            topk: 每層要保留的 top-k heads 數量
        
        Returns:
            Dict[int, List[int]]: {layer_idx: [head_idx1, head_idx2, ...]}
        """
        layer_to_heads = {}
        
        # head_importance 可能是 analyze_head_importance 的完整結果，或只是 head_analysis 列表
        if isinstance(head_importance, dict):
            head_analysis = head_importance.get('head_analysis', [])
        elif isinstance(head_importance, list):
            head_analysis = head_importance
        else:
            raise ValueError("head_importance 必須是 dict 或 list")
        
        for layer_info in head_analysis:
            layer = layer_info['layer']
            scores = layer_info['mean_scores']  # list of length H
            
            # 轉換為 tensor 以便使用 topk
            if isinstance(scores, list):
                scores_tensor = torch.tensor(scores)
            else:
                scores_tensor = scores
            
            k = min(topk, len(scores_tensor))
            if k > 0:
                idx = scores_tensor.topk(k).indices.tolist()
                layer_to_heads[layer] = idx
            else:
                layer_to_heads[layer] = []
        
        return layer_to_heads

    def select_and_mask_important_heads(self, x: torch.Tensor, topk: int = 2, 
                                       target_class: Optional[int] = None,
                                       save_plots: bool = False) -> Dict[int, List[int]]:
        """
        一鍵啟動「只用重要 head forward」
        
        1) 先跑一張圖片抓 head importance
        2) 自動挑選 topk heads
        3) 設定 forward mask，只保留這些 head
        
        Args:
            x: 輸入圖像 [B, 3, H, W]
            topk: 每層要保留的 top-k heads 數量
            target_class: 目標類別（None 表示使用預測類別）
            save_plots: 是否保存 head importance 圖表
        
        Returns:
            Dict[int, List[int]]: {layer_idx: [head_idx1, head_idx2, ...]} 使用的 head mapping
        """
        # Step 1: 分析 head importance
        analysis = self.analyze_head_importance(x, target_class=target_class, save_plots=save_plots)
        
        # Step 2: 從分析結果建立 head mask
        mapping = self.build_head_mask_from_importance(analysis, topk=topk)
        
        # Step 3: 設定 forward mask
        self.set_head_mask(mapping, keep_only=True)
        
        return mapping

    
    # --- new --- #

    def _hook_mlp_and_ln(self):  # // NEW
        # 建立從 block idx → cache 位置 的映射（用 block_idx 對齊）
        self._ln1_modules = []
        self._ln2_modules = []
        self._fc1_modules = []
        self._act_modules = []
        self._fc2_modules = []

        # 掃描 torchvision ViT 的結構，尋找每個 block 的 ln1/ln2 和 mlp.fc1/act/fc2
        ln1_suffixes = ('.ln_1', '.norm1')
        ln2_suffixes = ('.ln_2', '.norm2')

        for name, m in self.model.named_modules():
            # ln_1 / ln_2 或 norm1 / norm2
            if any(name.endswith(sfx) for sfx in ln1_suffixes) and isinstance(m, nn.LayerNorm):
                m.register_forward_hook(self._ln1_hook)  # ln1 輸入是殘差分支 x1
                self._ln1_modules.append(m)
            if any(name.endswith(sfx) for sfx in ln2_suffixes) and isinstance(m, nn.LayerNorm):
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

    def _hook_layer_activations(self):
        """Hook 每個 encoder block 的輸出（activation）"""
        self.layer_activations = []
        
        def _save_activation(module, inp, out):
            # 每個 block 的輸出是 [B, N, C]，保存它
            self.layer_activations.append(out.detach())
        
        # 尋找所有 encoder blocks
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            for i, layer in enumerate(self.model.encoder.layers):
                # Hook 在每個 block 的最後（MLP 之後的輸出）
                layer.register_forward_hook(_save_activation)
        elif hasattr(self.model, "blocks"):
            for i, layer in enumerate(self.model.blocks):
                layer.register_forward_hook(_save_activation)
        else:
            print("Warning: 無法找到 encoder layers 或 blocks 以擷取 activation")

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
        
        # 清理 layer activations
        self.layer_activations.clear()
        self.layer_activations_with_merging.clear()
        
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

    analyzer = VIT_PartialLRP_PatchMerging(model, topk_heads=2, head_weighting='normalize', eps=1e-6)

    x = torch.randn(1, 3, 224, 224, device=device)
    R_patch = analyzer.compute_lrp_relevance(x, return_map='patch')     # [1, 196]
    print("Patch relevance:", R_patch.shape)

    R_img = analyzer.compute_lrp_relevance(x, return_map='image', upsample_to_input=True)  # [1,224,224]
    print("Image relevance:", R_img.shape)
