"""
Vision Transformer (ViT) 模型，支援小 patch size (2 或 4)
相較於標準的 patch size 16，使用更小的 patch size 可以提供更高的空間解析度
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Patch Embedding 層：將圖像分割成 patches 並轉換為 embedding
    
    Args:
        img_size: 輸入圖像大小 (H, W) 或單一值（假設為正方形）
        patch_size: Patch 大小（例如 2, 4, 16）
        in_channels: 輸入通道數（預設為 3）
        embed_dim: Embedding 維度
    """
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 確保圖像大小可以被 patch size 整除
        if isinstance(img_size, (int, float)):
            img_size = (int(img_size), int(img_size))
        H, W = img_size
        
        assert H % patch_size == 0, f"圖像高度 {H} 必須能被 patch_size {patch_size} 整除"
        assert W % patch_size == 0, f"圖像寬度 {W} 必須能被 patch_size {patch_size} 整除"
        
        # 計算 patch 數量
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.patch_shape = (H // patch_size, W // patch_size)
        
        # 使用 Conv2d 來做 patch embedding
        # kernel_size 和 stride 都等於 patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 輸入 tensor [B, C, H, W]
        Returns:
            x: Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        # 確保輸入大小符合預期
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"輸入大小 ({H}, {W}) 與預期大小 {self.img_size} 不符"
        
        # Conv2d: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)
        
        # Flatten: [B, embed_dim, H', W'] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention 層
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} 必須能被 num_heads {num_heads} 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 投影層
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 輸入 tensor [B, N, embed_dim]
        Returns:
            x: 輸出 tensor [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # 計算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 計算注意力分數
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 應用注意力到 V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 輸出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network)
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    使用 nn.MultiheadAttention 以與 VIT_with_Partial_LRP 相容
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # 使用 nn.MultiheadAttention 以與 VIT_with_Partial_LRP 相容
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=attn_dropout,
            batch_first=True  # 使用 batch_first 格式
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        # nn.MultiheadAttention 需要 query, key, value 三個參數（都是同一個 x）
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VIT_SmallPatch(nn.Module):
    """
    Vision Transformer 模型，支援小 patch size (2 或 4)
    
    與標準 ViT-B/16 相比：
    - patch_size=2: 對於 224x224 圖像，產生 112x112=12,544 個 patches（vs 16: 14x14=196）
    - patch_size=4: 對於 224x224 圖像，產生 56x56=3,136 個 patches（vs 16: 14x14=196）
    
    更高的空間解析度，但計算量也更大。
    
    Args:
        in_channels: 輸入通道數（預設為 3）
        out_channels: 輸出類別數
        input_size: 輸入圖像大小 (H, W) 或單一值（假設為正方形）
        patch_size: Patch 大小（建議使用 2 或 4）
        embed_dim: Embedding 維度（預設 768，與 ViT-B 相同）
        depth: Transformer encoder 層數（預設 12，與 ViT-B 相同）
        num_heads: 注意力頭數（預設 12，與 ViT-B 相同）
        mlp_ratio: MLP 隱藏層維度比例（預設 4.0）
        dropout: Dropout 率
        attn_dropout: Attention dropout 率
        drop_path: Drop path 率（Stochastic Depth）
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        input_size: Tuple[int, int] = (32, 32),
        patch_size: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        # 驗證 patch_size
        assert patch_size in [2, 4], f"目前只支援 patch_size 2 或 4，您提供的是 {patch_size}"
        
        # 處理 input_size
        if isinstance(input_size, (int, float)):
            input_size = (int(input_size), int(input_size))
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Channel adapter（如果需要）
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            in_channels = 3
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, out_channels)
        
        # 初始化權重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型權重"""
        # Class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 其他層的初始化
        self.apply(self._init_weights_fn)
        
    def _init_weights_fn(self, m):
        """權重初始化函數"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 輸入 tensor [B, C, H, W]
        Returns:
            x: 分類 logits [B, out_channels]
        """
        B = x.shape[0]
        
        # Channel adapter（如果需要）
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # 確保輸入大小正確
        if x.shape[-2:] != self.input_size:
            x = torch.nn.functional.interpolate(
                x, size=self.input_size, mode='bilinear', align_corners=False
            )
        
        # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        
        # 添加 class token: [B, num_patches, embed_dim] -> [B, num_patches+1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加 position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # 使用 class token 進行分類
        x = x[:, 0]  # [B, embed_dim]
        
        # Classification head
        x = self.head(x)  # [B, out_channels]
        
        return x
    
    def get_patch_info(self) -> dict:
        """
        取得 patch 相關資訊
        
        Returns:
            dict: 包含 patch 數量、patch 形狀等資訊
        """
        return {
            'patch_size': self.patch_size,
            'num_patches': self.patch_embed.num_patches,
            'patch_shape': self.patch_embed.patch_shape,
            'input_size': self.input_size,
        }


# 便利函數：創建不同 patch size 的模型
def create_vit_patch2(
    in_channels: int = 3,
    out_channels: int = 10,
    input_size: Tuple[int, int] = (224, 224),
    **kwargs
) -> VIT_SmallPatch:
    """
    創建 patch_size=2 的 ViT 模型
    
    對於 224x224 圖像，會產生 112x112=12,544 個 patches
    提供最高的空間解析度，但計算量也最大
    """
    return VIT_SmallPatch(
        in_channels=in_channels,
        out_channels=out_channels,
        input_size=input_size,
        patch_size=2,
        **kwargs
    )


def create_vit_patch4(
    in_channels: int = 3,
    out_channels: int = 10,
    input_size: Tuple[int, int] = (32, 32),
    **kwargs
) -> VIT_SmallPatch:
    """
    創建 patch_size=4 的 ViT 模型
    
    對於 224x224 圖像，會產生 56x56=3,136 個 patches
    在解析度和計算量之間取得平衡
    """
    return VIT_SmallPatch(
        in_channels=in_channels,
        out_channels=out_channels,
        input_size=input_size,
        patch_size=4,
        **kwargs
    )

