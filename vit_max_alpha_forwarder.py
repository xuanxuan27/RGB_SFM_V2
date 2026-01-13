import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import copy
from torchvision import transforms

# 假設妳的專案結構
from config import config
from dataloader import get_dataloader
import models

# ==========================================
# 1. 強化的 Top-K 注意力 (帶有能量歸一化)
# ==========================================
class TopKAttentionWrapper(nn.Module):
    def __init__(self, original_attn: nn.MultiheadAttention, k=5, renormalize=True):
        super().__init__()
        self.k = k
        self.renormalize = renormalize
        self.num_heads = original_attn.num_heads
        self.in_proj_weight = original_attn.in_proj_weight
        self.in_proj_bias = original_attn.in_proj_bias
        self.out_proj = original_attn.out_proj
        self.embed_dim = original_attn.embed_dim
        self.scale = (self.embed_dim // self.num_heads) ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 如果沒有指定 k，表示使用「原生完整注意力」，僅做（可選）歸一化與回傳權重
        if self.k is None:
            full_attn = attn
            if self.renormalize:
                den = full_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                full_attn = full_attn / den
            x_out = (full_attn @ v).transpose(1, 2).reshape(B, N, C)
            x_out = self.out_proj(x_out)
            return x_out, full_attn

        # 否則執行 Top-K 篩選
        topk_vals, topk_indices = torch.topk(attn, k=self.k, dim=-1)
        topk_attn = torch.zeros_like(attn).scatter_(dim=-1, index=topk_indices, src=topk_vals)
        
        if self.renormalize:
            den = topk_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_attn = topk_attn / den
        
        x_out = (topk_attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.out_proj(x_out)
        
        return x_out, topk_attn

class TopKMHAIntervention(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, k=5, renormalize=True):
        super().__init__()
        self.mha = mha
        self.k = k
        self.renormalize = renormalize
        self.last_attn = None

        # 直接沿用原本 MHA 的參數
        self.num_heads = mha.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        self.embed_dim = mha.embed_dim
        self.scale = (self.embed_dim // self.num_heads) ** -0.5

    def forward(self, query, key, value, **kwargs):
        # torchvision ViT 是 batch_first=True，所以 query: [B,N,C]
        x = query
        B, N, C = x.shape

        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        topk_vals, topk_idx = torch.topk(attn, k=min(self.k, attn.shape[-1]), dim=-1)
        topk_attn = torch.zeros_like(attn).scatter_(dim=-1, index=topk_idx, src=topk_vals)

        if self.renormalize:
            den = topk_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_attn = topk_attn / den

        self.last_attn = topk_attn  # [B, heads, N, N]

        out = (topk_attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out, topk_attn


# ==========================================
# 4. NativeAttentionRecorder：嚴謹版：
# 不重算 qkv，不改 attention，
# 只是「呼叫原始 MultiheadAttention」並回傳 attn weights。
# 回傳 attn_map shape 期望為 [B, heads, N, N]
# ==========================================
class NativeAttentionRecorder(nn.Module):
    """
    嚴謹版：不重算 qkv，不改 attention，只呼叫原始 nn.MultiheadAttention 並回傳 per-head attn weights
    回傳 attn_map: [B, heads, N, N]
    """
    def __init__(self, original_attn: nn.MultiheadAttention):
        super().__init__()
        self.attn = original_attn
        assert isinstance(self.attn, nn.MultiheadAttention)


    def forward(self, x):
        # x: [B, N, C]
        if getattr(self.attn, "batch_first", False):
            out, w = self.attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            # w: [B, heads, N, N]
            return out, w
        else:
            # default MHA expects [N, B, C]
            x_t = x.transpose(0, 1)  # [N, B, C]
            out_t, w = self.attn(
                x_t, x_t, x_t,
                need_weights=True,
                average_attn_weights=False
            )
            out = out_t.transpose(0, 1)  # [B, N, C]
            return out, w

# ==========================================
# 5. MHARecorder：記錄 MultiheadAttention 的注意力權重
# 新增符合原始 MultiheadAttention 的 recorder
# ==========================================
class MHARecorder(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.last_attn = None  # [B, heads, N, N]

    def forward(self, query, key, value, **kwargs):
        # 強制要 per-head attn weights
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        out, w = self.mha(query, key, value, **kwargs)
        self.last_attn = w  # [B, heads, N, N]
        return out, w


# ==========================================
# 2. 分析核心：解耦與多樣性工具
# ==========================================
class ExplainabilityAnalyzer:
    def __init__(self, model, num_registers):
        self.model = model
        self.R = num_registers

    def get_attention_maps(self, X, k=None, renormalize=True):
        """
        嚴謹路線：
        - k=None：使用 NativeAttentionRecorder，保證是模型原生 attention
        - k=int：使用 TopKAttentionWrapper（intervention）
        """
        test_model = copy.deepcopy(self.model).eval().to(X.device)
        layers = test_model.backbone.encoder.layers

        for i in range(len(layers)):
            orig_attn = layers[i].self_attention
            if k is None:
                layers[i].self_attention = NativeAttentionRecorder(orig_attn)
            else:
                layers[i].self_attention = TopKAttentionWrapper(orig_attn, k=k, renormalize=renormalize)
            
        all_attns = []
        with torch.no_grad():
            x = test_model.backbone.conv_proj(X)
            x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)

            cls_token = test_model.backbone.class_token.expand(x.shape[0], -1, -1)
            registers = test_model.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x, registers), dim=1)
            x = x + test_model.backbone.encoder.pos_embedding

            for layer in layers:
                res = x
                x = layer.ln_1(x)

                x, attn_map = layer.self_attention(x)  # (out, attn)
                # 嚴謹檢查
                assert attn_map.dim() == 4, f"Expected [B, heads, N, N], got {attn_map.shape}"
                all_attns.append(attn_map)

                x = x + res
                x = x + layer.mlp(layer.ln_2(x))

            x = test_model.backbone.encoder.ln(x)
            cls = x[:, 0]
            logits = test_model.backbone.heads(cls)

        return all_attns, logits


    def plot_disentanglement(self, all_layer_attns, save_dir):
        """
        分析 CLS 到底在看誰 (Patches vs Registers)
        """
        os.makedirs(save_dir, exist_ok=True)
        cls_to_patches_ratios = []
        cls_to_regs_ratios = []

        for attn in all_layer_attns:
            # attn shape: [1, heads, seq_len, seq_len]
            # CLS 是 index 0
            cls_attn = attn[0, :, 0, :] # [heads, seq_len]
            
            num_patches = attn.shape[-1] - 1 - self.R
            patch_mass = cls_attn[:, 1:1+num_patches].sum(dim=-1).mean().item()
            reg_mass = cls_attn[:, 1+num_patches:].sum(dim=-1).mean().item()
            
            cls_to_patches_ratios.append(patch_mass)
            cls_to_regs_ratios.append(reg_mass)

        plt.figure(figsize=(10, 5))
        plt.plot(cls_to_patches_ratios, label='CLS to Patches (Visual Information)')
        plt.plot(cls_to_regs_ratios, label='CLS to Registers (Internal Buffer)')
        plt.title("Information Disentanglement across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Attention Mass")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "disentanglement_trend.png"))
        plt.close()

# ==========================================
# 3. RegisterRetainedForwarder：使用 Top-K 注意力的 Forwarder
# ==========================================
class RegisterRetainedForwarder(nn.Module):
    def __init__(self, base_vit, k=5):
        super().__init__()
        self.base_vit = base_vit
        self.backbone = base_vit.backbone
        self.num_registers = base_vit.num_registers
        self.layers = self.backbone.encoder.layers
        self.k = k

        # 將每一層的 self_attention 替換成 TopKAttentionWrapper
        for i in range(len(self.layers)):
            orig_attn = self.layers[i].self_attention
            self.layers[i].self_attention = TopKAttentionWrapper(orig_attn, k=self.k)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.backbone.conv_proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)

        # 2. 完整拼接：[CLS] + [Patches] + [Registers]
        cls_token = self.backbone.class_token.expand(x.shape[0], -1, -1)
        registers = self.base_vit.register_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x, registers), dim=1)  # [B, 1 + N + R, C]

        # 3. 加上 Positional Embedding
        x = x + self.backbone.encoder.pos_embedding

        # 4. 經過所有 Transformer Block，並收集注意力圖
        all_attns = []
        for layer in self.layers:
            res = x
            x = layer.ln_1(x)
            x, attn_map = layer.self_attention(x)
            all_attns.append(attn_map)
            x = x + res

            res = x
            x = layer.ln_2(x)
            x = layer.mlp(x)
            x = x + res

        return all_attns





# ==========================================
# 4. 分析 Head 多樣性
# ==========================================
def analyze_head_diversity(all_layer_attns, k=5):
    """
    計算每一層中各個 Head 之間的 Top-K 亮點重疊率 (Intersection over Union, IoU)
    """
    layer_diversity = []
    
    for layer_idx, attn in enumerate(all_layer_attns):
        # attn: [1, num_heads, seq_len, seq_len]
        num_heads = attn.shape[1]
        
        # 取得每個 Head 對 Patches 的 Top-K 索引集合
        # 排除 CLS (index 0)，只看 Patches
        head_topk_sets = []
        for h in range(num_heads):
            # 取出 CLS 對 Patches 的注意力
            cls_attn = attn[0, h, 0, 1:] 
            _, indices = torch.topk(cls_attn, k=k)
            head_topk_sets.append(set(indices.cpu().numpy()))
            
        # 計算兩兩之間的 IoU
        iou_matrix = np.zeros((num_heads, num_heads))
        for i in range(num_heads):
            for j in range(num_heads):
                intersection = len(head_topk_sets[i].intersection(head_topk_sets[j]))
                union = len(head_topk_sets[i].union(head_topk_sets[j]))
                iou_matrix[i, j] = intersection / union if union > 0 else 0
        
        layer_diversity.append(iou_matrix)
        
    return layer_diversity

# ==========================================
# 4. 繪製 Head 多樣性分析
# ==========================================
def _iou(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def extract_patch_set_from_cls(attn_map_1head, num_patches, mode="topk", k=5, eps=0.0):
    """
    attn_map_1head: Tensor shape [seq_len] = attn[0, head, cls=0, :]
    只回傳 patches 區段的索引集合（用於 IoU）

    mode:
    - "topk": 取 patches 中 top-k 的索引（適合 native）
    - "support": 取 patches 中非零(>eps) 的索引（適合 Top-K intervention）
    """
    patches = attn_map_1head[1:1+num_patches]  # 只看 patches
    if mode == "topk":
        kk = min(k, patches.numel())
        _, idx = torch.topk(patches, k=kk, dim=-1)
        return set(idx.detach().cpu().numpy().tolist())
    elif mode == "support":
        idx = torch.nonzero(patches > eps, as_tuple=False).squeeze(-1)
        return set(idx.detach().cpu().numpy().tolist())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def plot_head_diversity_analysis(
    all_layer_attns,
    num_registers,
    top_alpha_k=5,
    save_dir="vit_max_alpha_visualization/explainability_analysis",
    set_mode="topk",   # native 用 "topk"，topk intervention 用 "support"
    support_eps=0.0
):
    os.makedirs(save_dir, exist_ok=True)
    layer_avg_iou = []

    for layer_idx, attn in enumerate(all_layer_attns):
        # attn: [B, heads, seq, seq]，此處你 batch_size=1
        num_heads = attn.shape[1]
        R = num_registers
        num_patches_total = attn.shape[-1] - 1 - R

        head_sets = []
        for h in range(num_heads):
            cls_row = attn[0, h, 0, :]  # [seq_len]
            s = extract_patch_set_from_cls(
                cls_row, num_patches=num_patches_total,
                mode=set_mode, k=top_alpha_k, eps=support_eps
            )
            head_sets.append(s)

        iou_matrix = np.zeros((num_heads, num_heads), dtype=np.float32)
        for i in range(num_heads):
            for j in range(num_heads):
                if i == j:
                    iou_matrix[i, j] = 1.0
                else:
                    iou_matrix[i, j] = _iou(head_sets[i], head_sets[j])

        mask = np.ones(iou_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        layer_avg_iou.append(float(iou_matrix[mask].mean()))

        if (layer_idx + 1) in [1, 6, 12]:
            plt.figure(figsize=(8, 6))
            sns.heatmap(iou_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(f"Layer {layer_idx+1} Head IoU Matrix (mode={set_mode}, k={top_alpha_k})")
            plt.xlabel("Head Index")
            plt.ylabel("Head Index")
            plt.savefig(os.path.join(save_dir, f"layer_{layer_idx+1}_head_iou_{set_mode}.png"))
            plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(layer_avg_iou) + 1), layer_avg_iou, marker='o', linestyle='-')
    plt.axhline(y=np.mean(layer_avg_iou), linestyle='--', label='Overall Mean')
    plt.title(f"Attention Head Redundancy across Layers (mode={set_mode})", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Inter-head IoU", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"model_diversity_trend_{set_mode}.png"))
    plt.show()

    return layer_avg_iou

def compute_attention_rollout(
    all_layer_attns,
    *,
    head_fusion="mean",        # "mean" / "max" / "none"(需指定 head_idx)
    head_idx=0,                # 當 head_fusion="none" 時用哪個 head
    add_residual=True,         # 是否用 I + A
    row_normalize=True,        # 是否做 row-normalize（很建議開）
    start_layer=0,             # 從第幾層開始 rollout（有時前幾層太雜，可略過）
    device=None
):
    """
    Attention Rollout
    Inputs:
      all_layer_attns: list of Tensor, each [B, heads, N, N]
    Output:
      rollout: Tensor [B, N, N]  (已融合 heads 後的跨層累積)
    """
    assert isinstance(all_layer_attns, (list, tuple)) and len(all_layer_attns) > 0

    A0 = all_layer_attns[0]
    assert A0.dim() == 4, f"Expected [B, heads, N, N], got {A0.shape}"
    B, H, N, N2 = A0.shape
    assert N == N2, "Attention matrix must be square"

    if device is None:
        device = A0.device

    # 初始化 rollout 為 Identity（表示尚未經過任何 attention mixing）
    rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]

    # 逐層累積
    for l in range(start_layer, len(all_layer_attns)):
        attn = all_layer_attns[l].to(device)  # [B, H, N, N]

        # ---- head fusion: 變成 [B, N, N] ----
        if head_fusion == "mean":
            A = attn.mean(dim=1)
        elif head_fusion == "max":
            A = attn.max(dim=1).values
        elif head_fusion == "none":
            assert 0 <= head_idx < attn.shape[1]
            A = attn[:, head_idx, :, :]
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # ---- residual: I + A ----
        if add_residual:
            I = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
            A = A + I

        # ---- row normalize: 每一列和為 1（避免層數相乘後爆掉/衰掉）----
        if row_normalize:
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # ---- rollout 累積：注意順序很重要 ----
        # 直覺：先經過前面層的 mixing，再經過這一層
        rollout = A @ rollout   # [B, N, N]

    return rollout


def plot_rollout_heatmap_overlay(
    rollout,                   # [B, N, N]
    img_show,                  # numpy HxWx3, 0~1
    num_registers,
    save_path,
    *,
    title="Attention Rollout",
    batch_index=0,
    overlay_alpha=0.5,
    cmap="jet",
    interpolation=cv2.INTER_CUBIC
):
    """
    將 rollout 的 CLS -> patches 部分取出，reshape 成 patch grid，resize 疊到原圖上
    """
    assert rollout.dim() == 3, f"Expected [B, N, N], got {rollout.shape}"
    B, N, N2 = rollout.shape
    assert N == N2

    R = int(num_registers)
    num_patches_total = N - 1 - R
    side = int(np.sqrt(num_patches_total))
    assert side * side == num_patches_total, f"num_patches_total={num_patches_total} not square"

    # 取 CLS row（query=CLS index 0）對 patches 的影響
    cls_to_patches = rollout[batch_index, 0, 1:1+num_patches_total].detach().cpu().numpy()  # [num_patches]
    grid = cls_to_patches.reshape(side, side)

    # resize 到影像大小
    heat = cv2.resize(grid, (img_show.shape[1], img_show.shape[0]), interpolation=interpolation)

    # normalize
    if heat.max() > heat.min():
        heat_norm = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat_norm = heat

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
    im = plt.imshow(heat_norm, cmap=cmap, alpha=overlay_alpha)
    plt.axis("off")
    plt.title(title, fontsize=14, fontweight="bold")
    # 添加 colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Rollout Attention', rotation=270, labelpad=15, fontsize=10)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    return heat_norm  # 若你想後續做量化也能用

# ==========================================
# 5. 載入權重與執行視覺化 (參考你的路徑)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. 載入資料 (參考你的 dataloader) ---
    _, test_loader = get_dataloader(
        dataset=config["dataset"],
        root=os.path.join(config["root"], "data"),
        batch_size=1,
        input_size=config["input_shape"],
    )
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)

    # --- B. 建立模型與載入權重 (參考你的路徑) ---
    model_cfg = config["model"]
    model_args = dict(model_cfg["args"])
    # 建立原始模型
    base_model = getattr(getattr(models, model_cfg["name"]), model_cfg["name"])(**model_args)
    
    # 指定你的 checkpoint 路徑
    ckpt_path = "runs/train/exp124/RegisterViT_best.pth" 
    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict(ckpt["model_weights"], strict=False)
    else:
        print("Warning: Checkpoint not found, using random weights.")

    base_model.to(device).eval()

    # ==========================================
    # 重要：先執行新方法（嚴謹雙軌分析），因為它內部使用 deepcopy，不會污染 base_model
    # ==========================================
    print("\n" + "="*60)
    print("Starting NEW Analysis Method: ExplainabilityAnalyzer (嚴謹雙軌)")
    print("="*60)
    
    analyzer = ExplainabilityAnalyzer(base_model, num_registers=base_model.num_registers)
    
    # 1) Native（嚴謹）
    native_attns, native_logits = analyzer.get_attention_maps(X, k=None)
    native_probs = F.softmax(native_logits, dim=1)
    native_conf, native_pred = torch.max(native_probs, dim=1)
    print(f"[Native] pred={native_pred.item()}, conf={native_conf.item():.2%}")
    
    # 2) Top-K intervention
    K = 5
    topk_attns, topk_logits = analyzer.get_attention_maps(X, k=K, renormalize=True)
    topk_probs = F.softmax(topk_logits, dim=1)
    topk_conf, topk_pred = torch.max(topk_probs, dim=1)
    print(f"[TopK k={K}] pred={topk_pred.item()}, conf={topk_conf.item():.2%}")
    
    # --- C. 執行預測與獲取資訊 (使用原始模型，確保不被污染) ---
    with torch.no_grad():
        # 取得模型預測 (使用原始模型，此時 base_model 尚未被污染)
        print("\n[Old Method] Running model prediction with clean base_model...")
        logits = base_model(X)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        print(f"[Old Method] Prediction completed. Confidence: {conf.item():.2%}")
        
        # 舊方法：使用 deepcopy 的模型，避免污染 base_model
        print("\n[Old Method] Creating deepcopy for old analysis method...")
        top_alpha_k = 5
        old_model = copy.deepcopy(base_model).to(device).eval()
        forwarder = RegisterRetainedForwarder(old_model, k=top_alpha_k)
        all_layer_attns = forwarder(X)
        print(f"[Old Method] Attention maps computed for {len(all_layer_attns)} layers.")

    # --- D. 顯示原圖與預測資訊 ---
    # 影像反正規化 (假設使用 ImageNet 標準)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_show = inv_normalize(X[0]).cpu().permute(1, 2, 0).numpy()
    img_show = np.clip(img_show, 0, 1)

    # 取得類別名稱 (假設妳的 dataset 有 classes 屬性)
    class_names = getattr(test_loader.dataset, 'classes', None)
    
    # 處理 y：可能是單一標籤索引或 one-hot 編碼
    if y.dim() > 1 and y.shape[1] > 1:
        # one-hot 編碼，取最大值索引
        y_idx = torch.argmax(y[0], dim=0).item()
    else:
        # 單一標籤索引
        y_idx = y[0].item() if y.dim() > 0 else y.item()
    
    gt_label = class_names[y_idx] if class_names else str(y_idx)
    pred_label = class_names[pred.item()] if class_names else str(pred.item())

    # --- D-1. 保存原圖並顯示預測資訊 ---
    vis_dir = "vit_max_alpha_visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
    plt.title(
        f"Ground Truth: {gt_label}\nPrediction: {pred_label}\nConfidence: {conf.item():.2%}", 
        fontsize=16, fontweight='bold'
    )
    plt.axis('off')
    save_path = os.path.join(vis_dir, "input_image_with_prediction.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Input image with prediction saved to: {save_path}")

    # --- E. 輸出每一層的所有 Head 顯著圖 (平滑熱力圖版) ---
    # 定義通用 heatmap 繪製函數
    def plot_heatmap_for_attns(all_layer_attns, num_registers, img_show, pred_label, conf, 
                                save_dir, title_prefix="", file_suffix=""):
        """通用函數：為給定的 attention maps 繪製 heatmap"""
        R = num_registers
        num_heads = all_layer_attns[0].shape[1]
        num_patches_total = all_layer_attns[0].shape[-1] - 1 - R
        side = int(np.sqrt(num_patches_total))
        assert side * side == num_patches_total

        os.makedirs(save_dir, exist_ok=True)
        
        for layer_idx, attn in enumerate(all_layer_attns):
            # 總共需要 num_heads + 1 個子圖
            total_plots = num_heads + 1
            cols = 4
            rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = axes.flatten()

            # 1. 先繪製每一個獨立的 Head
            for head_idx in range(num_heads):
                head_attn = attn[0, head_idx, 0, 1:(1 + num_patches_total)].cpu().numpy()
                saliency = head_attn.reshape(side, side)
                
                # 平滑化處理
                saliency_resized = cv2.resize(
                    saliency, 
                    (img_show.shape[1], img_show.shape[0]), 
                    interpolation=cv2.INTER_CUBIC
                )
                
                # 正規化（保留原始數值範圍用於 colorbar）
                saliency_min = saliency_resized.min()
                saliency_max = saliency_resized.max()
                if saliency_max > saliency_min:
                    saliency_norm = (saliency_resized - saliency_min) / (saliency_max - saliency_min)
                else:
                    saliency_norm = saliency_resized
                    
                axes[head_idx].imshow(img_show)
                im = axes[head_idx].imshow(saliency_norm, cmap='jet', alpha=0.5)
                axes[head_idx].set_title(f"Head {head_idx+1}")
                axes[head_idx].axis('off')
                # 添加 colorbar
                cbar = plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)
                cbar.set_label('Attention', rotation=270, labelpad=15, fontsize=8)

            # 2. 繪製該層所有 Head 的綜合結果 (Aggregate)
            all_heads_mean = attn[0, :, 0, 1:(1 + num_patches_total)].mean(dim=0).cpu().numpy()
            agg_saliency = all_heads_mean.reshape(side, side)
            
            agg_resized = cv2.resize(
                agg_saliency, 
                (img_show.shape[1], img_show.shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
            # 正規化（保留原始數值範圍用於 colorbar）
            agg_min = agg_resized.min()
            agg_max = agg_resized.max()
            if agg_max > agg_min:
                agg_norm = (agg_resized - agg_min) / (agg_max - agg_min)
            else:
                agg_norm = agg_resized

            axes[num_heads].imshow(img_show)
            im_agg = axes[num_heads].imshow(agg_norm, cmap='jet', alpha=0.5)
            axes[num_heads].set_title(
                f"LAYER {layer_idx+1} AGGREGATE", 
                fontsize=14, fontweight='bold', color='red'
            )
            axes[num_heads].axis('off')
            # 添加 colorbar
            cbar_agg = plt.colorbar(im_agg, ax=axes[num_heads], fraction=0.046, pad=0.04)
            cbar_agg.set_label('Attention', rotation=270, labelpad=15, fontsize=8)

            # 3. 移除剩餘沒用到的空子圖
            for j in range(total_plots, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle(
                f"{title_prefix}Layer {layer_idx + 1} Heatmap | Individual Heads & Aggregate\nPred: {pred_label} ({conf.item():.2%})", 
                fontsize=20
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(save_dir, f"layer_{layer_idx + 1}_heatmap_with_agg{file_suffix}.png")
            plt.savefig(save_path)
            plt.close(fig)

    # 繪製 Top-K heatmap（舊方法）
    R = forwarder.num_registers
    dir_path_topk = "vit_max_alpha_visualization/heatmap_visual"
    print(f"Generating Top-K Heatmap with Aggregate Subplot for {len(all_layer_attns)} layers...")
    plot_heatmap_for_attns(
        all_layer_attns, R, img_show, pred_label, conf,
        dir_path_topk, title_prefix="[Top-K] ", file_suffix=""
    )
    
    # 繪製 Native heatmap（新方法）
    dir_path_native = "vit_max_alpha_visualization/heatmap_visual_native"
    print(f"Generating Native Heatmap with Aggregate Subplot for {len(native_attns)} layers...")
    plot_heatmap_for_attns(
        native_attns, base_model.num_registers, img_show, 
        class_names[native_pred.item()] if class_names else str(native_pred.item()), 
        native_conf,
        dir_path_native, title_prefix="[Native] ", file_suffix="_native"
    )

    # --- E-1. 計算並繪製 Attention Rollout ---
    print("\nComputing Attention Rollout...")
    rollout_dir = "vit_max_alpha_visualization/rollout"
    os.makedirs(rollout_dir, exist_ok=True)
    
    # 將 attention maps 轉換為 tensor（如果還是 list）
    def to_tensor_list(attn_list):
        """確保 attention maps 是 tensor list"""
        return [torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in attn_list]
    
    # 1. Native Rollout
    print("  Computing Native Rollout...")
    native_attns_tensor = to_tensor_list(native_attns)
    native_rollout = compute_attention_rollout(
        native_attns_tensor,
        head_fusion="mean",
        add_residual=True,
        row_normalize=True,
        start_layer=0,
        device=device
    )
    
    native_rollout_path = os.path.join(rollout_dir, "native_rollout.png")
    plot_rollout_heatmap_overlay(
        native_rollout,
        img_show,
        base_model.num_registers,
        native_rollout_path,
        title=f"Native Attention Rollout\nPred: {class_names[native_pred.item()] if class_names else str(native_pred.item())} ({native_conf.item():.2%})"
    )
    print(f"  Native rollout saved to: {native_rollout_path}")
    
    # 2. Top-K Rollout
    print("  Computing Top-K Rollout...")
    topk_attns_tensor = to_tensor_list(topk_attns)
    topk_rollout = compute_attention_rollout(
        topk_attns_tensor,
        head_fusion="mean",
        add_residual=True,
        row_normalize=True,
        start_layer=0,
        device=device
    )
    
    topk_rollout_path = os.path.join(rollout_dir, f"topk_k{K}_rollout.png")
    plot_rollout_heatmap_overlay(
        topk_rollout,
        img_show,
        base_model.num_registers,
        topk_rollout_path,
        title=f"Top-K (k={K}) Attention Rollout\nPred: {class_names[topk_pred.item()] if class_names else str(topk_pred.item())} ({topk_conf.item():.2%})"
    )
    print(f"  Top-K rollout saved to: {topk_rollout_path}")
    
    # 3. Top-K (舊方法) Rollout
    print("  Computing Top-K (Old Method) Rollout...")
    old_attns_tensor = to_tensor_list(all_layer_attns)
    old_rollout = compute_attention_rollout(
        old_attns_tensor,
        head_fusion="mean",
        add_residual=True,
        row_normalize=True,
        start_layer=0,
        device=device
    )
    
    old_rollout_path = os.path.join(rollout_dir, "topk_old_method_rollout.png")
    plot_rollout_heatmap_overlay(
        old_rollout,
        img_show,
        forwarder.num_registers,
        old_rollout_path,
        title=f"Top-K (Old Method) Attention Rollout\nPred: {pred_label} ({conf.item():.2%})"
    )
    print(f"  Top-K (Old Method) rollout saved to: {old_rollout_path}")

    # --- F. 執行多頭多樣性分析 (IoU 分析) - 舊方法 ---
    print("\n[Old Method] Starting Attention Head Diversity Analysis...")
    
    avg_ious = plot_head_diversity_analysis(
        all_layer_attns, 
        num_registers=forwarder.num_registers,
        top_alpha_k=top_alpha_k,
        save_dir="vit_max_alpha_visualization/explainability_analysis",
        set_mode="topk"
    )
    print("[Old Method] Diversity analysis completed. Results saved in 'vit_max_alpha_visualization/explainability_analysis' directory.")

    # --- G. 產出新分析方式的報告 ---
    res_dir = "vit_max_alpha_visualization/final_explainability_report"
    os.makedirs(res_dir, exist_ok=True)
    
    # --- Disentanglement（你原本那兩張圖可以沿用）
    print("\n[New Method] Plotting disentanglement trend (Native Attention)...")
    analyzer.plot_disentanglement(native_attns, os.path.join(res_dir, "native"))
    print(f"[New Method] Disentanglement trend saved to: {os.path.join(res_dir, 'native/disentanglement_trend.png')}")
    
    print("[New Method] Plotting disentanglement trend (Top-K Attention)...")
    analyzer.plot_disentanglement(topk_attns, os.path.join(res_dir, f"topk_k{K}"))
    print(f"[New Method] Disentanglement trend saved to: {os.path.join(res_dir, f'topk_k{K}/disentanglement_trend.png')}")
    
    # --- Head Diversity：native 用 topk 集合；topk intervention 用 support 集合
    print("\n[Native] IoU redundancy (set_mode=topk)")
    native_iou = plot_head_diversity_analysis(
        native_attns,
        num_registers=base_model.num_registers,
        top_alpha_k=K,
        save_dir=os.path.join(res_dir, "native_diversity"),
        set_mode="topk"
    )
    
    print(f"[TopK] IoU redundancy (set_mode=support)")
    topk_iou = plot_head_diversity_analysis(
        topk_attns,
        num_registers=base_model.num_registers,
        top_alpha_k=K,  # 這裡其實不重要，support 用不到，但保留介面一致
        save_dir=os.path.join(res_dir, f"topk_k{K}_diversity"),
        set_mode="support",
        support_eps=0.0
    )
    
    print("\n" + "="*60)
    print("All Analysis Completed!")
    print("="*60)
    print(f"\nOld Method Results:")
    print(f"  - Heatmaps: vit_max_alpha_visualization/heatmap_visual/")
    print(f"  - IoU Analysis: vit_max_alpha_visualization/explainability_analysis/")
    print(f"\nNew Method Results:")
    print(f"  - Disentanglement (Native): {res_dir}/native/")
    print(f"  - Disentanglement (Top-K): {res_dir}/topk_k{K}/")
    print(f"  - Diversity (Native, topk mode): {res_dir}/native_diversity/")
    print(f"  - Diversity (Top-K, support mode): {res_dir}/topk_k{K}_diversity/")
    print("="*60)

if __name__ == "__main__":
    main()