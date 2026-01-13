import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# DropPath (Stochastic Depth)
# =========================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob and self.drop_prob > 0.0 and self.training:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = random_tensor.floor()
            x = x / keep_prob * random_tensor
        return x

# =========================================================
# Overlapping Patch Embedding
# =========================================================
class OverlapPatchEmbed(nn.Module):
    """
    以 Conv2d( stride < kernel_size ) 取得重疊 patch，並展平成序列。
    輸入:  (B, C_in, H, W)
    輸出:  (B, H'*W', D), 以及 H', W'
    """
    def __init__(self, in_ch, embed_dim, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.proj(x)                     # (B, D, H', W')
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)     # (B, H'*W', D)
        x = self.norm(x)
        return x, H, W

# =========================================================
# MLP / FFN
# =========================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =========================================================
# SRA: Spatial-Reduction Attention
# =========================================================
class SRAttention(nn.Module):
    """
    只對 K,V 的輸入做空間降採樣 (sr_ratio)，讓注意力複雜度從 O(N^2) 降為 O(N * (N/r^2)).
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x, H, W):
        """
        x: (B, N, C) 其中 N=H*W
        回傳: (B, N, C)
        """
        B, N, C = x.shape

        # Q
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.transpose(1, 2)  # (B, heads, N, head_dim)

        # K, V 的輸入做 SR（若 sr_ratio > 1）
        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
            x_ = self.sr(x_)                            # (B, C, H/r, W/r)
            Hr, Wr = x_.shape[-2:]
            x_ = x_.flatten(2).transpose(1, 2)          # (B, H'r*W'r, C)
            x_ = self.norm(x_)
        else:
            x_ = x
            Hr, Wr = H, W

        kv = self.kv(x_)  # (B, N', 2C)
        kv = kv.reshape(B, Hr * Wr, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, heads, N', head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N')
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                 # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)     # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# =========================================================
# Transformer Block (with SRA)
# =========================================================
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(dim, num_heads, sr_ratio, qkv_bias, attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# =========================================================
# Head 規劃：自動依 stage 遞增（且確保能整除）
# =========================================================
def _largest_divisor_at_most(n: int, k: int) -> int:
    """回傳 <=k 的最大 d 使得 n % d == 0；若找不到則回 1。"""
    k = max(1, min(k, n))
    for d in range(k, 0, -1):
        if n % d == 0:
            return d
    return 1

def make_head_schedule(dims, head_dim_target=64, min_heads=1, max_heads=None):
    """
    針對每層通道數 Ci：
      1) 估計 h* = round(Ci / head_dim_target)
      2) 取 <= h* 且可整除 Ci 的最大因子
      3) 套上 min/max
    """
    heads = []
    for Ci in dims:
        h_est = max(min_heads, round(Ci / head_dim_target))
        if max_heads is not None:
            h_est = min(h_est, max_heads)
        h = _largest_divisor_at_most(Ci, h_est)
        h = max(min_heads, h)
        heads.append(h)
    return heads

# =========================================================
# PVTv2 Backbone
# =========================================================
class PVTv2(nn.Module):
    """
    - 回傳多尺度特徵 (B, C_i, H_i, W_i)
    - 可選擇是否輸出分類 logits
    - 頭數可自動/手動規劃，並隨 stage 遞增
    Tiny 配置（可調）:
      Stage1: C=64,  blocks=2, sr=8,  stride=4  -> 224->56
      Stage2: C=128, blocks=2, sr=4,  stride=2  -> 56->28
      Stage3: C=320, blocks=2, sr=2,  stride=2  -> 28->14
      Stage4: C=512, blocks=2, sr=1,  stride=2  -> 14->7
    """
    def __init__(self,
                 num_classes=1000,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 out_indices=(0, 1, 2, 3),
                 with_cls_head=True,
                 # head 規劃
                 head_schedule="auto",    # "auto" 或 list[int,int,int,int]
                 head_dim_target=64,      # auto 模式目標每頭維度
                 max_heads=None):         # auto 模式上限（例如 8/12/16）
        super().__init__()
        self.out_indices = set(out_indices)
        self.with_cls_head = with_cls_head

        # ---- Tiny 基線（可依需求調整） ----
        self.dims   = [64, 128, 320, 512]
        self.blocks = [2,   2,   2,   2]
        self.srs    = [8,   4,   2,   1]
        self.strides= [4,   2,   2,   2]
        in_chs      = [3, self.dims[0], self.dims[1], self.dims[2]]

        # ---- 產生 heads（遞增）----
        if head_schedule == "auto":
            self.heads = make_head_schedule(self.dims, head_dim_target=head_dim_target, max_heads=max_heads)
        else:
            assert isinstance(head_schedule, (list, tuple)) and len(head_schedule) == 4
            self.heads = list(head_schedule)
            for Ci, hi in zip(self.dims, self.heads):
                if Ci % hi != 0:
                    raise ValueError(f"dim {Ci} 不能被 heads {hi} 整除，請調整。")

        # ---- Patch Embedding 模組（每個 stage 重新下採樣+提特徵）----
        ksz = [7, 3, 3, 3]
        pad = [3, 1, 1, 1]
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_chs[i], self.dims[i], kernel_size=ksz[i], stride=self.strides[i], padding=pad[i])
            for i in range(4)
        ])

        # ---- 每個 stage 的 transformer blocks ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.blocks))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.ModuleList([
                Block(self.dims[i], self.heads[i], mlp_ratio=4.0, sr_ratio=self.srs[i],
                      drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(self.blocks[i])
            ])
            cur += self.blocks[i]
            self.stages.append(stage)

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in self.dims])
        self.head = nn.Linear(self.dims[-1], num_classes) if with_cls_head else None
        
        # 初始化權重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化權重，避免梯度爆炸"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # -----------------------------
    # 取多尺度特徵
    # -----------------------------
    def forward_features(self, x):
        """
        x: (B, 3, H, W)
        回傳：
          outs: List[(B, C_i, H_i, W_i)] 依 out_indices
          last_seq: (B, N_last, C_last), H_last, W_last
        """
        outs = []
        B = x.shape[0]

        # 追蹤上一層的 (C, H, W)，以便重建影像餵下一層
        prev_C, prev_H, prev_W = None, None, None
        seq = None  # (B, N, C)

        for i in range(4):
            if i == 0:
                x_img = x  # (B, 3, H, W)
            else:
                # 把上一個 stage 的序列還原成 feature map 再做下一個 patch_embed
                x_img = seq.transpose(1, 2).reshape(B, prev_C, prev_H, prev_W)  # (B, C_{i-1}, H_{i-1}, W_{i-1})

            # Patch Embedding (下採樣 + 展平 + LN)
            seq, H, W = self.patch_embeds[i](x_img)          # seq: (B, H_i*W_i, C_i)

            # 經過多層 Block（SRA 注意力 + MLP）
            for blk in self.stages[i]:
                seq = blk(seq, H, W)                         # (B, N_i, C_i)

            # 正規化並輸出該層的 (B, C_i, H_i, W_i)
            seq_out = self.norms[i](seq)                     # (B, N_i, C_i)
            feat = seq_out.transpose(1, 2).reshape(B, self.dims[i], H, W)
            if i in self.out_indices:
                outs.append(feat)

            # 記錄給下一個 stage 使用
            prev_C, prev_H, prev_W = self.dims[i], H, W
            # 這個 seq_out 也是最後分類 head 的輸入
            last_seq = seq_out

        return outs, (last_seq, H, W)

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, x):
        # 檢查輸入是否包含 NaN 或 Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or Inf values!")
            return torch.zeros(x.shape[0], self.head.out_features if self.head else self.dims[-1], device=x.device)
        
        outs, (last_seq, H, W) = self.forward_features(x)
        if self.with_cls_head:
            feat_map = last_seq.transpose(1, 2).reshape(x.shape[0], self.dims[-1], H, W)  # (B, C4, H4, W4)
            pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)                         # (B, C4)
            logits = self.head(pooled)                                                     # (B, num_classes)
            
            # 檢查輸出是否包含 NaN 或 Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: Model output contains NaN or Inf values!")
                print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"Pooled range: [{pooled.min():.4f}, {pooled.max():.4f}]")
                return torch.zeros_like(logits)
            
            return logits  # 只回傳 logits，符合訓練框架需求
        else:
            return outs

# =========================================================
# PVT 包裝類（符合現有訓練框架介面）
# =========================================================
class PVT(nn.Module):
    """
    PVT 模型的包裝類，符合現有訓練框架的介面需求
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 10, input_size=(224, 224), **kwargs):
        super().__init__()
        
        # 通道適配器（如果需要）
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        
        # 創建 PVTv2 backbone
        self.backbone = PVTv2(
            num_classes=out_channels,
            with_cls_head=True,
            head_schedule=kwargs.get('head_schedule', 'auto'),
            head_dim_target=kwargs.get('head_dim_target', 64),
            max_heads=kwargs.get('max_heads', None),
            drop_rate=kwargs.get('drop_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', 0.0)
        )
        
        self.input_size = input_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道適配
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # 調整輸入尺寸（如果需要）
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        
        # 通過 PVT backbone
        return self.backbone(x)

# =========================================================
# Quick Test
# =========================================================
if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    
    # 測試原始 PVTv2
    print("=== 測試原始 PVTv2 ===")
    model_v2 = PVTv2(
        num_classes=1000,
        with_cls_head=True,
        head_schedule="auto",
        head_dim_target=64,
        max_heads=16
    )
    
    x = torch.randn(2, 3, 224, 224)
    logits = model_v2(x)
    print("PVTv2 Logits:", tuple(logits.shape))
    
    # 測試包裝類 PVT
    print("\n=== 測試包裝類 PVT ===")
    model = PVT(
        in_channels=3,
        out_channels=10,
        input_size=(224, 224),
        head_schedule="auto",
        head_dim_target=64,
        max_heads=16
    )
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("PVT Logits:", tuple(logits.shape))
    print("Heads per stage:", getattr(model.backbone, "heads", None))
