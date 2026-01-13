import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

# --------------------------------------------------------
# 1. 定義一個可解釋的 Attention 層 (用來替換原本的層)
# --------------------------------------------------------
class LRP_Attention(nn.Module):
    """
    這是一個「白盒子」的 Attention，用來替換 torchvision 的 nn.MultiheadAttention。
    它的行為跟原本一模一樣，但會把 Attention Map 暴露出來讓我們抓梯度。
    """
    def __init__(self, original_attn: nn.MultiheadAttention):
        super().__init__()
        # 繼承原本的權重
        self.num_heads = original_attn.num_heads
        self.in_proj_weight = original_attn.in_proj_weight
        self.in_proj_bias = original_attn.in_proj_bias
        self.out_proj = original_attn.out_proj
        self.embed_dim = original_attn.embed_dim
        self.batch_first = original_attn.batch_first

        # 用來存 LRP 需要的資料
        self.attn_map = None
        self.attn_grad = None

    def save_attn_grad(self, grad):
        self.attn_grad = grad

    def forward(self, x, key=None, value=None, key_padding_mask=None, need_weights=True, attn_mask=None):
        # 假設是 Self-Attention，輸入通常都是 x
        # x shape: [Batch, Seq, Dim] (if batch_first=True)
        is_batched = x.dim() == 3
        
        # 簡單處理 batch_first (Torchvision ViT 預設 batch_first=True)
        if self.batch_first and is_batched:
            B, N, C = x.shape
        else:
            # 如果不是 batch_first，通常是 [Seq, Batch, Dim]
            N, B, C = x.shape
            x = x.transpose(0, 1) # 轉成 [B, N, C] 方便計算
            
        # 1. 計算 QKV
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 計算 Attention Score
        scale = (C // self.num_heads) ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        
        # (忽略 mask 處理，因為 ViT 分類任務通常不用 mask)
        attn = attn_logits.softmax(dim=-1) # [B, H, N, N]

        # --- 關鍵：在這裡攔截梯度 ---
        if self.training or x.requires_grad:
            attn.retain_grad()
            self.attn_map = attn # 存 forward map
            # 註冊 hook 抓 backward grad
            attn.register_hook(self.save_attn_grad)
            
        # 3. 計算 Output
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(output)

        # 轉回原本的 shape
        if not self.batch_first and is_batched:
            output = output.transpose(0, 1)

        # 回傳格式要跟 nn.MultiheadAttention 一樣 (output, attn_weights)
        return output, attn


# --------------------------------------------------------
# 2. 主要的 LRP Wrapper
# --------------------------------------------------------
class VIT_with_Partial_LRP_RegisterAware(nn.Module):
    def __init__(
        self,
        vit_model: nn.Module,
        num_patches: int = 196,
        num_registers: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = vit_model
        self.eps = eps
        self.num_patches = num_patches
        self.num_registers = num_registers

        # 找出 Encoder
        if hasattr(self.model, 'backbone'):
            self.encoder = self.model.backbone.encoder
        elif hasattr(self.model, 'blocks'):
            self.encoder = self.model.blocks # timm
        else:
            self.encoder = self.model.encoder # torchvision

        # --- 自動替換 Attention 層 ---
        self._replace_attention_layers()

    def _replace_attention_layers(self):
        """
        遍歷模型，把所有黑盒子的 nn.MultiheadAttention 
        換成我們自定義的 LRP_Attention
        """
        layers = self.encoder.layers if hasattr(self.encoder, 'layers') else self.encoder
        
        for idx, layer in enumerate(layers):
            # 檢查是哪種結構
            if hasattr(layer, 'self_attention') and isinstance(layer.self_attention, nn.MultiheadAttention):
                # Torchvision 風格
                print(f"Replacing layer {idx} attention...")
                layer.self_attention = LRP_Attention(layer.self_attention)
            elif hasattr(layer, 'attn') and hasattr(layer.attn, 'qkv'):
                # Timm 風格 (這個範例主要針對 torchvision，timm 結構不同需另外寫)
                print(f"Warning: Layer {idx} is Timm-style, LRP_Attention might need adaptation.")
            else:
                print(f"Warning: Could not find MultiheadAttention in layer {idx}")

    def forward(self, x):
        return self.model(x)

    def explain(
        self,
        x,
        target_class=None,
        return_map="image",
        upsample_to_input=True,
        restrict_heads: Optional[Dict[int, List[int]]] = None,
    ):
        # 準備
        self.model.eval()
        self.model.zero_grad()
        x.requires_grad_(True)
        
        # 1. Forward Pass
        logits = self.forward(x)
        B = x.shape[0]

        if target_class is None:
            target_class = logits.argmax(dim=1)
            
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * B, device=x.device)

        # 2. Backward Pass
        score = logits.gather(1, target_class.view(-1, 1)).squeeze()
        score.sum().backward()

        # 3. 開始計算 LRP
        # 我們現在直接從替換後的 module 裡面拿 map 和 grad
        
        # 收集所有的 LRP_Attention 層
        layers = self.encoder.layers if hasattr(self.encoder, 'layers') else self.encoder
        attn_layers = [l.self_attention for l in layers] # 這些現在都是 LRP_Attention 物件了
        
        # LRP Init
        # 取得實際的 token 數量 (從第一層的 map 拿)
        num_tokens = attn_layers[0].attn_map.shape[-1]
        R = torch.eye(num_tokens, device=x.device).unsqueeze(0).expand(B, -1, -1)

        # Reverse Loop
        for idx, layer in enumerate(reversed(attn_layers)):
            # 因為是 reversed，idx 0 對應最後一層
            # 但 layer 物件本身已經是對的，不用管 index 對應問題
            
            attn = layer.attn_map   # [B, H, N, N]
            grad = layer.attn_grad  # [B, H, N, N]
            
            if grad is None:
                # 如果還是 None，代表梯度真的沒傳到這層 (可能是凍結了?)
                # 為了不報錯，給個全 0
                print(f"Warning: Layer {len(attn_layers)-1-idx} gradient is None. Using zeros.")
                grad = torch.zeros_like(attn)

            # LRP Rule
            attn_prod = attn * grad
            attn_prod = attn_prod.clamp(min=0)
            
            # (Optional) Restrict Heads
            real_idx = len(attn_layers) - 1 - idx
            if restrict_heads is not None and real_idx in restrict_heads:
                mask = torch.zeros(attn.shape[1], device=x.device)
                for h in restrict_heads[real_idx]:
                    mask[h] = 1.0
                attn_prod = attn_prod * mask.view(1, -1, 1, 1)

            attn_mean = attn_prod.mean(dim=1)
            
            # Update R
            R = R + torch.matmul(attn_mean, R)

        # 4. Post-processing
        # 這裡會根據 Register 數量做切片，只留 patch 的部分
        R_patch = R[:, 0, 1 : 1 + self.num_patches] # [B, 196]
        
        H = W = int(self.num_patches ** 0.5)
        R_img = R_patch.view(B, 1, H, W)

        if upsample_to_input:
            R_img = F.interpolate(
                R_img,
                size=x.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        return R_img, R

    # 加在 VIT_with_Partial_LRP_RegisterAware 類別裡面
    
    def get_all_heads_importance(self, x, target_class=None):
        """
        一次性計算所有 Layer 所有 Head 的重要性分數。
        分數定義：該 Head 負責傳遞的 Relevance 總量 (LRP Rule 的中間產物)。
        
        Returns:
            head_scores: Numpy array [num_layers, num_heads]
        """
        self.model.eval()
        self.model.zero_grad()
        x.requires_grad_(True)
        
        # 1. Forward
        logits = self.forward(x)
        B = x.shape[0]

        if target_class is None:
            target_class = logits.argmax(dim=1)
            
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * B, device=x.device)

        # 2. Backward
        score = logits.gather(1, target_class.view(-1, 1)).squeeze()
        score.sum().backward()

        # 3. 收集分數
        # 假設你的 encoder 有 layers
        layers = self.encoder.layers if hasattr(self.encoder, 'layers') else self.encoder
        attn_layers = [l.self_attention for l in layers] # 這些是 LRP_Attention
        
        num_layers = len(attn_layers)
        num_heads = attn_layers[0].num_heads
        
        # 儲存矩陣 [Layers, Heads]
        head_scores = torch.zeros((num_layers, num_heads), device=x.device)

        # 遍歷每一層 (順序沒差，因為我們只看當層的 attn * grad)
        for i, layer in enumerate(attn_layers):
            attn = layer.attn_map   # [B, H, N, N]
            grad = layer.attn_grad  # [B, H, N, N]
            
            if grad is None:
                continue

            # Chefer's LRP Rule: (A * grad).clamp(min=0)
            # 這代表該 Head 實際上傳遞了多少正向 Relevance
            relevance_contribution = (attn * grad).clamp(min=0)
            
            # 對 [B, N, N] 加總，得到該 Head 的總貢獻量
            # [B, H]
            score_per_head = relevance_contribution.sum(dim=(2, 3))
            
            # 這裡我們取 Batch 平均 (如果 Batch > 1)
            head_scores[i] = score_per_head.mean(dim=0)
            
        return head_scores.detach().cpu().numpy()