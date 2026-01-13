# file: partial_lrp_vit_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import torchvision.models as tv_models
from torchvision.models.vision_transformer import ViT_B_16_Weights

# (前面的 lrp_linear, MHAWrapper 保持不變，直接沿用)
# ... [請保留原本的 lrp_linear 和 MHAWrapper 程式碼] ...

class VIT_with_Partial_LRP(nn.Module):
    """
    Modified Manual LRP for ViT.
    Features:
      - Supports 'num_registers' (to exclude register tokens from heatmap).
      - Supports 'restrict_heads' (manual head filtering).
    """
    def __init__(self,
                 vit_model: Optional[nn.Module] = None,
                 in_channels: int = 3,
                 out_channels: int = 1000,
                 input_size: Tuple[int, int] = (224, 224),
                 topk_heads: Optional[int] = None,
                 head_weighting: str = 'normalize',
                 num_registers: int = 0,  # <--- [NEW] 新增 Register token 數量
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.topk_heads = topk_heads
        self.head_weighting = head_weighting
        self.input_size = input_size
        self.num_registers = num_registers # <--- [NEW]

        # 構建或採用外部提供的 ViT backbone
        if vit_model is None:
            # (預設載入 torchvision 模型邏輯保持不變)
            try:
                backbone = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            except TypeError:
                backbone = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.model = backbone
        else:
            self.model = vit_model

        # Channel Adapter
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # Cache & Hooks
        self.cache: List[Dict[str, Any]] = []
        self.layer_activations: List[torch.Tensor] = []
        self._wrap_attentions()
        self._hook_tokens()
        self._hook_mlp_and_ln()
        self._hook_layer_activations()
        self._block_counter = 0
        self._pending_ln1 = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return self.model(x)

    # ... [中間的 hook 函式、clear_head_mask 等保持不變] ...
    # ... [_wrap_attentions, _attn_forward_hook, _hook_tokens 等保持不變] ...
    # ... [_hook_mlp_and_ln, _hook_layer_activations 等保持不變] ...

    # ---------------------------
    # Public API (Modified)
    # ---------------------------
    @torch.no_grad()
    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                return_map: str = 'patch', upsample_to_input: bool = False, 
                return_head_importance: bool = False, 
                restrict_heads: Optional[Dict[int, List[int]]] = None) -> torch.Tensor: # <--- [Refined]
        """
        Args:
            x: [B, 3, H, W]
            return_map: 'patch' (grid), 'token' (raw tokens), 'image' (upsampled)
            restrict_heads: {layer_idx: [head_idx1, head_idx2]}
                            Only allow relevance to flow through these heads.
        """
        self._clear_memory_cache()
        device = next(self.model.parameters()).device
        x = x.to(device)

        # 1. Forward Pass
        logits = self.model(x)
        B, num_classes = logits.shape

        # 2. Target Selection
        if target_class is None:
            target_idx = logits.argmax(dim=1)
        else:
            target_idx = torch.full((B,), int(target_class), device=device, dtype=torch.long)

        # 3. Initialize Relevance (CLS token)
        if self.last_tokens is None:
            return self._fallback_grad_relmap(x, target_idx, return_map)

        # (抓取 Classifier 權重邏輯不變)
        head_module = self.model.heads.head if hasattr(self.model, 'heads') else self.model.head
        W_cls = head_module.weight.t().detach()
        b_cls = head_module.bias.detach()
        h_cls = self.last_tokens[:, 0, :] # CLS is always at 0

        R_out_cls = torch.zeros_like(logits)
        R_out_cls[torch.arange(B), target_idx] = logits[torch.arange(B), target_idx]

        # Handle zero/negative logits
        if torch.all(R_out_cls == 0):
             if logits.max() < 0:
                probs = torch.softmax(logits, dim=1)
                R_out_cls[torch.arange(B), target_idx] = probs[torch.arange(B), target_idx]
             else:
                R_out_cls[torch.arange(B), target_idx] = 1.0

        # Backprop to CLS token
        R_cls = self.lrp_linear_static(h_cls, W_cls, R_out_cls, b=b_cls, eps=self.eps)
        
        # Initialize R_tokens [B, N, C]
        R_tokens = torch.zeros_like(self.last_tokens)
        R_tokens[:, 0, :] = R_cls

        # 4. Backward Propagation (Layer by Layer)
        head_importance_scores = [] if return_head_importance else None

        for i, blk in enumerate(reversed(self.cache)):
            current_layer = len(self.cache) - 1 - i
            
            # Unpack block cache
            A, W_o, W_v = blk['attn_probs'], blk['W_o'], blk['W_v']
            X_in, out = blk['X_in'], blk['out']
            
            # --- (I) & (II) MLP & Residual 2 Backprop ---
            # (這部分邏輯保持不變，為了節省版面省略細節，直接照抄原本的)
            # ... [省略 MLP LRP 程式碼，請確保複製過來] ...
            # 假設這裡計算出了 R_after_mlp (第一殘差前的 Relevance)
            
            # 這裡簡化代替 MLP 部分程式碼供參考：
            ln2_in = blk['ln2_in']
            mlp_out = blk.get('fc2_out', None)
            # ... (略) ...
            # 暫時假設 MLP 傳播完畢，得到 R_after_mlp
            # 實際使用請填入完整 MLP 回推邏輯
            # ----------------------------------------------------------------
            # 為確保程式碼完整，以下是 MLP 的簡略回推（請使用原檔的詳細邏輯）
            # 簡單版: 若 hook 失敗則全給殘差
            if ln2_in is not None and mlp_out is not None:
                denom = (ln2_in.abs() + mlp_out.abs() + 1e-6)
                R_skip2 = R_tokens * (ln2_in.abs() / denom)
                R_mlp_out = R_tokens * (mlp_out.abs() / denom)
                
                # MLP Backprop
                if blk['act_out'] is not None:
                     R_act = self.lrp_linear_static(blk['act_out'], blk['W_fc2'], R_mlp_out, b=blk['b_fc2'], eps=self.eps)
                     R_fc1_out = self.lrp_gelu_stable_static(blk['fc1_out'], R_act, eps=self.eps)
                     R_ln2_in = self.lrp_linear_static(blk['fc1_in'], blk['W_fc1'], R_fc1_out, b=blk['b_fc1'], eps=self.eps)
                else:
                     R_ln2_in = torch.zeros_like(R_skip2)
                R_after_mlp = R_skip2 + R_ln2_in
            else:
                R_after_mlp = R_tokens

            # --- (III) Residual 1 Backprop ---
            x1 = blk['ln1_in']
            if x1 is not None:
                denom1 = (x1.abs() + out.abs() + 1e-6)
                R_x1 = R_after_mlp * (x1.abs() / denom1)
                R_attnout = R_after_mlp * (out.abs() / denom1)
            else:
                R_x1, R_attnout = torch.zeros_like(R_after_mlp), R_after_mlp

            # --- (IV) Attention Backprop (Core Logic) ---
            # 1. Linear Backprop: out <- concat
            R_concat = self.lrp_linear_static(out, W_o, R_attnout, eps=self.eps)
            B_, N, C = R_concat.shape
            H = A.shape[1]
            d = C // H
            R_heads = R_concat.view(B_, N, H, d).permute(0, 2, 1, 3) # [B,H,N,d]

            # 2. Attention Map Routing (A^T)
            R_V = torch.einsum('bhji,bhjd->bhid', A, R_heads) # [B,H,N,d]

            # ==========================================
            # [NEW] Head Filtering (Restrict Heads)
            # ==========================================
            if restrict_heads is not None and current_layer in restrict_heads:
                active_heads = restrict_heads[current_layer]
                mask = torch.zeros(H, device=R_V.device, dtype=R_V.dtype)
                valid_heads = [h for h in active_heads if 0 <= h < H]
                if len(valid_heads) > 0:
                    mask[valid_heads] = 1.0
                
                # Apply mask to R_V. This stops relevance flowing through unwanted heads.
                R_V = R_V * mask.view(1, H, 1, 1)
            # ==========================================

            # 3. Head Scoring & Normalization
            head_scores = torch.clamp(R_V, min=0).sum(dim=(2,3)) # [B,H]
            head_scores = torch.nan_to_num(head_scores)

            if return_head_importance:
                head_importance_scores.append({
                    'layer': current_layer,
                    'head_scores': head_scores.detach().cpu()
                })

            if self.head_weighting == 'normalize':
                w = head_scores / (head_scores.sum(dim=1, keepdim=True) + 1e-6)
                R_V = R_V * w[:, :, None, None]

            # 4. Linear Backprop: V <- X_in
            R_V_concat = R_V.permute(0, 2, 1, 3).contiguous().view(B_, N, C)
            R_ln1_in = self.lrp_linear_static(X_in, W_v, R_V_concat, eps=self.eps)

            # --- (V) Sum Relevance ---
            R_tokens = R_x1 + R_ln1_in
            R_tokens = torch.nan_to_num(R_tokens)

        # 5. Output Formatting (Register Aware)
        # R_tokens shape: [B, N, C]
        # N = 1 (CLS) + num_patches + num_registers
        
        # [NEW] Calculate pure patch count
        total_tokens = R_tokens.shape[1]
        num_patches = total_tokens - 1 - self.num_registers
        
        if return_map == 'token':
            # Return raw tokens (including registers if any)
            return R_tokens
            
        else:
            # Slicing: Skip CLS (0), take patches, Skip Registers (end)
            # shape: [B, num_patches, C]
            R_patch_tokens = R_tokens[:, 1 : 1 + num_patches, :]
            
            # Aggregate Channel Relevance
            R_patch = R_patch_tokens.abs().sum(dim=2) # [B, num_patches]

            if return_map == 'patch':
                return R_patch
            
            elif return_map == 'image':
                # Reshape to grid
                grid_size = int(num_patches**0.5)
                # Check if square
                if grid_size * grid_size != num_patches:
                    print(f"Warning: Patch count {num_patches} is not a perfect square. "
                          f"Cannot reshape to image map. Returning patch vector.")
                    return R_patch

                R_grid = R_patch.view(B, grid_size, grid_size).unsqueeze(1) # [B,1,H,W]
                
                if upsample_to_input:
                    R_img = F.interpolate(R_grid, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    return R_img.squeeze(1)
                else:
                    return R_grid.squeeze(1)
                    
        return R_tokens # default fallback

    # ... [Helper methods: static methods for lrp, etc. copy from original] ...
    @staticmethod
    def lrp_linear_static(a, W, R_out, b=None, eps=1e-6):
        # (將原來的 lrp_linear 邏輯搬進來或是外部引用)
        z = a @ W
        if b is not None: z = z + b
        denom = z + eps * torch.sign(z)
        s = R_out / (denom + 1e-12)
        c = s @ W.t()
        R_in = a * c
        return torch.nan_to_num(R_in)

    @staticmethod
    def lrp_gelu_stable_static(x, R_out, eps=1e-6):
        # (將原來的 lrp_gelu_stable 邏輯搬進來)
        if R_out is None: return None
        Phi = 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))
        phi = torch.exp(-0.5 * x**2) / 2.5066282746310002
        dydx = Phi + x * phi
        dydx = torch.clamp(dydx, min=eps)
        R_in = R_out * dydx
        return torch.nan_to_num(R_in)
    
    # ... [其他 helper methods 如 _clear_memory_cache 保持不變] ...