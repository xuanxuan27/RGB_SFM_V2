#!/usr/bin/env python3
"""
視覺化：RegisterViT + VIT_with_Partial_LRP_RegisterAware

- 原圖
- All heads relevance (heatmap + overlay)
- Head relevance heatmap (所有層所有 head 的 relevance 分數)
- Per-head relevance (指定 layer 的每個 head)
- 若有 register token：畫 register relevance bar chart
"""

import os
import glob
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from config import config
from dataloader import get_dataloader
import models
from models.VIT_with_Partial_LRP_RegisterAware import VIT_with_Partial_LRP_RegisterAware
from models.RegisterViT import RegisterViT
from models.vit_lrt_register_token import vit_lrt_register_token

# =========================
# 工具函式
# =========================

def _latest_best_checkpoint(model_name: str = "RegisterViT") -> str:
    """
    嘗試尋找 runs/train/exp*/{model_name}_best.pth
    找不到時回傳空字串。
    """
    pattern = os.path.join("runs", "train", "exp*", f"{model_name}_best.pth")
    cand = sorted(glob.glob(pattern))
    if len(cand) > 0:
        return cand[-1]
    fixed = os.path.join("runs", "train", "exp", f"{model_name}_best.pth")
    return fixed if os.path.exists(fixed) else ""


def denorm_img(img_hw3: np.ndarray,
               mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
               std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    將 ImageNet 標準化後的影像還原到 [0,1] 範圍。
    """
    if img_hw3.ndim == 3 and img_hw3.shape[-1] == 3:
        mean_arr = np.array(mean).reshape(1, 1, 3)
        std_arr = np.array(std).reshape(1, 1, 3)
        img = img_hw3 * std_arr + mean_arr
        return np.clip(img, 0.0, 1.0)
    return np.clip(img_hw3, 0.0, 1.0)


def collect_n_samples(loader, device, n: int):
    """
    從 dataloader 蒐集最多 n 筆樣本（可跨多個 batch）。
    """
    xs, ys = [], []
    collected = 0
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
        collected += xb.size(0)
        if collected >= n:
            break
    if len(xs) == 0:
        return None, None
    X_cat = torch.cat(xs, dim=0)[:n]
    if isinstance(ys[0], torch.Tensor):
        y_cat = torch.cat(ys, dim=0)[:n]
    else:
        y_cat = torch.tensor(ys)[:n]
    return X_cat.to(device), y_cat.to(device)


def build_model_and_load_weights(device):
    """
    先依照 config 建立訓練時的模型（RegisterViT），載入 checkpoint 權重。
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_args = dict(model_cfg["args"])

    print(f"建立模型: {model_name}，參數: {model_args}")
    model = getattr(getattr(models, model_name), model_name)(**model_args)
    model = model.to(device).eval()

    # 載入權重
    # ckpt_path = _latest_best_checkpoint(model_name)
    ckpt_path = "runs/train/exp123/RegisterViT_best.pth"

    if not ckpt_path:
        # 若你有固定 exp 可以手動指定，例如：
        ckpt_path = "runs/train/exp123/RegisterViT_best.pth"
        print("自動尋找 checkpoint 失敗，請確認路徑或手動指定。")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"載入權重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt is not None and "model_weights" in ckpt:
            weights = ckpt["model_weights"]
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"載入結果 - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        else:
            print("checkpoint 中找不到 'model_weights'，略過載入。")
    else:
        print(f"找不到權重檔 {ckpt_path}，使用隨機初始化權重。")

    return model


def build_analyzer_from_model(model, device, num_patches: int = 196, num_registers: int = 0, eps: float = 1e-6):
    """
    從已載好權重的 RegisterViT 建立 VIT_with_Partial_LRP_RegisterAware analyzer。
    """
    # 如果本身已經是 analyzer，直接用
    if isinstance(model, (VIT_with_Partial_LRP_RegisterAware, vit_lrt_register_token)):
        return model.to(device).eval()

    # # 若是外層包裝器（RegisterViT），則取 backbone
    # if hasattr(model, "backbone"):
    #     print("偵測到包裝器模型，使用其 backbone 作為 ViT 主體。")
    #     vit_backbone = model.backbone
    #     # 若 model 本身有 num_registers，就覆蓋參數
    #     if hasattr(model, "num_registers"):
    #         num_registers = model.num_registers
    # else:
    #     vit_backbone = model

    if isinstance(model, RegisterViT):
        print("偵測到 RegisterViT，使用完整 RegisterViT（含 register tokens）")
        vit_backbone = model     # ← 不要使用 model.backbone!!
        num_registers = model.num_registers
    else:
        print("使用標準 ViT 模型")
        vit_backbone = model

    print(f"使用 VIT_with_Partial_LRP_RegisterAware 做為 analyzer，num_patches={num_patches}, num_registers={num_registers}")
    # analyzer = VIT_with_Partial_LRP_RegisterAware(
    #     vit_model=vit_backbone,
    #     num_patches=num_patches,
    #     num_registers=num_registers,
    #     eps=eps,
    # ).to(device).eval()

    analyzer = vit_lrt_register_token(
        vit_model=vit_backbone,
        num_patches=num_patches,
        num_registers=num_registers,
        eps=eps,
    ).to(device).eval()

    return analyzer


def robust_normalize(data):
    """
    使用 99 分位數進行正規化，避免極端值導致整個熱圖看不見。
    回傳 [0,1]。
    """
    vmax = np.percentile(np.abs(data), 99)
    if vmax == 0:
        vmax = 1e-12
    data_clipped = np.clip(data, -vmax, vmax)
    norm_data = (data_clipped + vmax) / (2 * vmax)
    return norm_data


def get_num_heads_from_layer(analyzer, layer_idx: int) -> int:
    """
    從指定 encoder layer 取得 head 數量。
    """
    # 使用 analyzer 的 encoder（已經處理了 RegisterViT 的情況）
    encoder = analyzer.encoder
    if layer_idx < 0 or layer_idx >= len(encoder.layers):
        raise ValueError(f"layer_idx={layer_idx} 超出範圍 (共有 {len(encoder.layers)} 個 layers)")

    blk = encoder.layers[layer_idx]
    attn = blk.self_attention
    if hasattr(attn, "num_heads"):
        return attn.num_heads
    # 若之後你有包 MultiheadAttention，可以視情況補上其它路徑
    return 12  # 預設 ViT-B/16 = 12 heads


def get_num_layers(analyzer) -> int:
    """
    取得 encoder 的總層數。
    """
    encoder = analyzer.encoder
    return len(encoder.layers)


def compute_head_relevance_heatmap(
    analyzer,
    X_sample: torch.Tensor,
    num_patches: int = 196,
) -> np.ndarray:
    """
    計算每層每個 head 的 relevance 分數，回傳 [num_layers, max_num_heads] 的矩陣。
    
    Args:
        analyzer: VIT_with_Partial_LRP_RegisterAware analyzer
        X_sample: 輸入樣本 [1, C, H, W]
        num_patches: patch 數量
    
    Returns:
        heatmap_matrix: [num_layers, max_num_heads] 的 numpy 陣列，值為每個 head 的 relevance 分數
        layer_info: dict，包含每層的 head 數量資訊
    """
    device = next(analyzer.model.parameters()).device
    X_sample = X_sample.to(device)
    
    num_layers = get_num_layers(analyzer)
    
    # 先取得每層的 head 數量
    num_heads_per_layer = []
    for layer_idx in range(num_layers):
        try:
            num_heads = get_num_heads_from_layer(analyzer, layer_idx)
            num_heads_per_layer.append(num_heads)
        except Exception as e:
            print(f"Warning: 無法取得 layer {layer_idx} 的 head 數量: {e}")
            num_heads_per_layer.append(12)  # 預設值
    
    max_num_heads = max(num_heads_per_layer) if num_heads_per_layer else 12
    
    # 初始化 heatmap 矩陣
    heatmap_matrix = np.zeros((num_layers, max_num_heads), dtype=np.float32)
    
    print(f"計算 {num_layers} 層，每層最多 {max_num_heads} 個 heads 的 relevance 分數...")
    
    # 遍歷每一層
    for layer_idx in range(num_layers):
        num_heads = num_heads_per_layer[layer_idx]
        print(f"  Layer {layer_idx}: {num_heads} heads")
        
        # 遍歷該層的每個 head
        for head_idx in range(num_heads):
            # 只保留當前層的當前 head
            restrict_config = {layer_idx: [head_idx]}
            
            # 計算該 head 的 relevance map
            relevance_h, _ = analyzer.explain(
                X_sample,
                target_class=None,
                return_map="image",
                upsample_to_input=True,
                restrict_heads=restrict_config,
            )
            
            # 計算 relevance 分數（使用絕對值的總和）
            rel_h = relevance_h[0, 0].detach().cpu().numpy()  # [H, W]
            relevance_score = np.abs(rel_h).sum()  # 絕對值總和作為分數
            
            heatmap_matrix[layer_idx, head_idx] = relevance_score
    
    layer_info = {
        'num_layers': num_layers,
        'num_heads_per_layer': num_heads_per_layer,
        'max_num_heads': max_num_heads,
    }
    
    return heatmap_matrix, layer_info


def plot_head_relevance_heatmap(
    analyzer,
    X_sample: torch.Tensor,
    sample_idx: int,
    save_dir: str,
    gt: int,
    pred: int,
    conf: float,
    status: str,
    num_patches: int = 196,
):
    """
    繪製每層每個 head 的 relevance 分數 heatmap。
    
    Args:
        analyzer: VIT_with_Partial_LRP_RegisterAware analyzer
        X_sample: 輸入樣本 [1, C, H, W]
        sample_idx: 樣本索引
        save_dir: 儲存目錄
        gt: ground truth label
        pred: predicted label
        conf: prediction confidence
        status: "✓" 或 "✗"
        num_patches: patch 數量
    """
    # 計算 heatmap 矩陣
    heatmap_matrix, layer_info = compute_head_relevance_heatmap(
        analyzer, X_sample, num_patches=num_patches
    )
    
    num_layers = layer_info['num_layers']
    max_num_heads = layer_info['max_num_heads']
    
    # 準備標籤
    layer_labels = [f"layer_{i}" for i in range(num_layers)]
    head_labels = [str(i) for i in range(max_num_heads)]
    
    # 繪製 heatmap
    plt.figure(figsize=(max(10, max_num_heads * 0.7), max(6, num_layers * 0.5)))
    im = plt.imshow(heatmap_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
    plt.colorbar(im, label='Head Relevance Score')
    plt.xticks(range(max_num_heads), head_labels)
    plt.yticks(range(num_layers), layer_labels)
    plt.xlabel('Head Index')
    plt.ylabel('Layer')
    plt.title(
        f"Sample {sample_idx} {status} - Head Relevance Heatmap\n"
        f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=12,
        pad=15
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_head_relevance_heatmap.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    # 繪製每層 head relevance 總和折線圖
    layer_sums = [
        heatmap_matrix[i, : layer_info['num_heads_per_layer'][i]].sum()
        for i in range(num_layers)
    ]
    plt.figure(figsize=(max(8, num_layers * 0.6), 4))
    plt.plot(range(num_layers), layer_sums, marker='o')
    plt.xticks(range(num_layers), layer_labels, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('Sum of Head Relevance')
    plt.title(
        f"Sample {sample_idx} {status} - Layer-wise Sum of Head Relevance\n"
        f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=11,
        pad=15,
    )
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_head_relevance_sum_line.png"),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    
    print(f"已儲存 head relevance heatmap 到 sample_{sample_idx}_head_relevance_heatmap.png")

# ======================= 新增 claude 給的分析 function =======================

def compute_texture_focus(rel_map):
    """計算紋理聚焦程度（使用局部變異）"""
    from scipy import ndimage
    # 計算局部標準差作為紋理指標
    local_std = ndimage.generic_filter(
        rel_map, 
        np.std, 
        size=3
    )
    return local_std.mean()

def plot_semantic_radar_chart(semantic_metrics, layer_idx, save_dir, sample_idx):
    """
    視覺化語義特徵雷達圖 (修正版：解決數值量級不一致導致圖形單一的問題)
    """
    try:
        from math import pi
        import matplotlib.pyplot as plt
        import numpy as np

        num_heads = len(semantic_metrics)
        if num_heads == 0:
            return
        
        categories = ['Spatial\nConcentration', 'Edge\nSensitivity', 'Center\nBias', 'Texture\nFocus']
        num_vars = len(categories)
        
        # 步驟 1: 提取所有數據以便進行「分項正規化」
        # data_matrix shape: [num_heads, 4]
        raw_data = []
        head_names = []
        for h_idx, (head_name, metrics) in enumerate(semantic_metrics.items()):
            raw_data.append([
                metrics['spatial_concentration'],
                metrics['edge_sensitivity'],
                metrics['center_bias'],
                metrics['texture_focus']
            ])
            head_names.append(head_name)
        
        raw_data = np.array(raw_data)
        
        # 步驟 2: 計算每個指標在「該層所有 Head」中的最大值 (Column-wise Max)
        # 加上 1e-8 避免除以零
        max_vals = raw_data.max(axis=0) + 1e-8
        
        # 步驟 3: 進行分項正規化 (每個指標自己跟自己比)
        # 這樣 Edge 就算數值很小，只要它是該層最強的，就會顯示為 1.0
        normalized_data = raw_data / max_vals

        # 開始繪圖
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # 閉合
        
        rows = int(np.ceil(np.sqrt(num_heads)))
        cols = int(np.ceil(num_heads / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), subplot_kw=dict(projection='polar'))
        
        if num_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for h_idx in range(num_heads):
            values = normalized_data[h_idx].tolist()
            values += [values[0]]  # 閉合
            
            ax = axes[h_idx]
            
            # 繪製雷達線
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # 設定標籤與範圍
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 1.1)  # 固定範圍
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([]) # 隱藏內部刻度數字以保持整潔
            
            # 標題顯示 Head ID 與 該 Head 最強的特徵
            # 找出這個 Head 在四個指標中相對最強的 (原本數值正規化後的 argmax)
            dominant_feature_idx = np.argmax(normalized_data[h_idx])
            dominant_feature = categories[dominant_feature_idx].replace('\n', ' ')
            
            ax.set_title(f'{head_names[h_idx]}\n(Main: {dominant_feature})', fontsize=10, pad=15)
        
        # 關閉多餘子圖
        for j in range(num_heads, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Sample {sample_idx} - Layer {layer_idx} Semantic Features (Relative Scale)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{sample_idx}_layer{layer_idx}_semantic_radar.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: 無法繪製語義雷達圖: {e}")
        import traceback
        traceback.print_exc()

def analyze_head_semantics_advanced(
    analyzer, 
    X_sample: torch.Tensor, 
    layer_idx: int,
    save_dir: str,
    sample_idx: int
):
    """進階的 head 語義分析"""
    device = next(analyzer.model.parameters()).device
    X_sample = X_sample.to(device)
    
    num_heads = get_num_heads_from_layer(analyzer, layer_idx)
    
    # 語義特徵分析
    semantic_metrics = {}
    
    for h in range(num_heads):
        restrict_config = {layer_idx: [h]}
        relevance_h, _ = analyzer.explain(
            X_sample,
            target_class=None,
            return_map="image",
            upsample_to_input=True,
            restrict_heads=restrict_config
        )
        rel_map = relevance_h[0, 0].detach().cpu().numpy()
        
        # 計算各種語義指標
        metrics = {
            'spatial_concentration': compute_spatial_concentration(rel_map),
            'edge_sensitivity': compute_edge_sensitivity(rel_map),
            'center_bias': compute_center_bias(rel_map),
            'texture_focus': compute_texture_focus(rel_map),
        }
        semantic_metrics[f'head_{h}'] = metrics
    
    # 視覺化語義特徵雷達圖
    plot_semantic_radar_chart(semantic_metrics, layer_idx, save_dir, sample_idx)
    
    return semantic_metrics

def compute_spatial_concentration(rel_map):
    """計算空間集中度（熵值）"""
    rel_flat = rel_map.flatten()
    rel_norm = rel_flat / (rel_flat.sum() + 1e-8)
    entropy = -np.sum(rel_norm * np.log(rel_norm + 1e-8))
    return 1.0 / (1.0 + entropy)  # 轉換為集中度分數

def compute_edge_sensitivity(rel_map):
    """計算邊緣敏感度"""
    from scipy import ndimage
    sobel_x = ndimage.sobel(rel_map, axis=1)
    sobel_y = ndimage.sobel(rel_map, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return edge_magnitude.mean()

def compute_center_bias(rel_map):
    """計算中心偏向程度"""
    h, w = rel_map.shape
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # 計算加權平均距離（relevance 作為權重）
    weights = np.abs(rel_map) + 1e-8
    weighted_distance = np.sum(distances * weights) / np.sum(weights)
    max_distance = np.sqrt((h/2)**2 + (w/2)**2)
    
    return 1.0 - (weighted_distance / max_distance)  # 轉換為中心偏向分數

def compute_head_similarity_matrix(analyzer, X_sample, layer_idx):
    """計算同一層不同 head 之間的相似度"""
    num_heads = get_num_heads_from_layer(analyzer, layer_idx)
    similarity_matrix = np.zeros((num_heads, num_heads))
    
    # 收集所有 head 的 relevance maps
    head_maps = []
    for h in range(num_heads):
        restrict_config = {layer_idx: [h]}
        relevance_h, _ = analyzer.explain(
            X_sample,
            target_class=None,
            return_map="image",
            upsample_to_input=True,
            restrict_heads=restrict_config
        )
        rel_map = relevance_h[0, 0].detach().cpu().numpy().flatten()
        head_maps.append(rel_map)
    
    # 計算相似度矩陣
    for i in range(num_heads):
        for j in range(num_heads):
            correlation = np.corrcoef(head_maps[i], head_maps[j])[0, 1]
            similarity_matrix[i, j] = correlation if not np.isnan(correlation) else 0
    
    return similarity_matrix

def plot_head_similarity_heatmap(similarity_matrix, layer_idx, save_dir, sample_idx):
    """視覺化 head 相似度矩陣"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title(f'Sample {sample_idx} - Layer {layer_idx} Head Similarity Matrix')
    plt.xlabel('Head Index')
    plt.ylabel('Head Index')
    
    # 添加數值標註（僅在矩陣較小時顯示）
    if similarity_matrix.shape[0] <= 16:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                plt.text(j, i, f'{similarity_matrix[i,j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{sample_idx}_layer{layer_idx}_head_similarity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def analyze_register_interaction(analyzer, X_sample, save_dir, sample_idx, num_registers, num_patches=196):
    """分析 register token 與其他 token 的交互作用"""
    if num_registers == 0:
        return
    
    # 取得完整的 relevance 矩陣
    _, R_tokens = analyzer.explain(
        X_sample,
        target_class=None,
        return_map="image",
        upsample_to_input=True,
        restrict_heads=None
    )
    R_matrix = R_tokens[0].detach().cpu().numpy()  # [N_tokens, N_tokens]
    
    # 分析 register token 的交互模式
    cls_reg_interaction = R_matrix[0, 1+num_patches:1+num_patches+num_registers]  # CLS 對 register 的影響
    patch_reg_interaction = R_matrix[1:1+num_patches, 1+num_patches:1+num_patches+num_registers]  # patch 對 register 的影響
    
    # 視覺化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. CLS-Register 交互
    axes[0].bar(range(num_registers), cls_reg_interaction)
    axes[0].set_title('CLS → Register Token Interaction')
    axes[0].set_xlabel('Register Token Index')
    axes[0].set_ylabel('Relevance Flow')
    
    # 2. Patch-Register 交互（空間視圖）
    patch_size = int(np.sqrt(num_patches))
    patch_reg_sum = patch_reg_interaction.sum(axis=1).reshape(patch_size, patch_size)
    im = axes[1].imshow(patch_reg_sum, cmap='viridis')
    axes[1].set_title('Patch → Register Token Interaction\n(Spatial View)')
    plt.colorbar(im, ax=axes[1])
    
    # 3. Register 之間的交互（如果有多個）
    if num_registers > 1:
        reg_reg_interaction = R_matrix[1+num_patches:1+num_patches+num_registers, 1+num_patches:1+num_patches+num_registers]
        im = axes[2].imshow(reg_reg_interaction, cmap='RdBu_r')
        axes[2].set_title('Register ↔ Register Interaction')
        plt.colorbar(im, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, 'Only 1 Register\nToken', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Register Self-Interaction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{sample_idx}_register_interaction_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

def compare_correct_vs_incorrect_predictions(analyzer, X_batch, y_batch, save_dir):
    """比較正確預測和錯誤預測的 head 使用模式"""
    
    correct_head_scores = []
    incorrect_head_scores = []
    
    for i in range(len(X_batch)):
        X_sample = X_batch[i:i+1]
        
        # 判斷預測是否正確
        with torch.no_grad():
            logits = analyzer.model(X_sample)
            pred = logits.argmax(dim=1).item()
            if y_batch.dim() > 1 and y_batch.shape[1] > 1:
                gt = y_batch[i].argmax(dim=0).item() if y_batch.dim() > 1 else y_batch[i].item()
            else:
                gt = y_batch[i].item()
        
        # 計算 head 重要性
        try:
            head_scores = analyzer.get_all_heads_importance(X_sample)
        except Exception as e:
            print(f"Warning: 無法計算 head 重要性: {e}")
            continue
        
        if pred == gt:
            correct_head_scores.append(head_scores)
        else:
            incorrect_head_scores.append(head_scores)
    
    if len(correct_head_scores) > 0 and len(incorrect_head_scores) > 0:
        # 計算平均模式
        correct_avg = np.mean(correct_head_scores, axis=0)
        incorrect_avg = np.mean(incorrect_head_scores, axis=0)
        
        # 視覺化差異
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(correct_avg, cmap='plasma')
        axes[0].set_title('Correct Predictions\nAverage Head Importance')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(incorrect_avg, cmap='plasma')
        axes[1].set_title('Incorrect Predictions\nAverage Head Importance')
        plt.colorbar(im2, ax=axes[1])
        
        difference = correct_avg - incorrect_avg
        vmax = max(abs(difference.max()), abs(difference.min()))
        im3 = axes[2].imshow(difference, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title('Difference\n(Correct - Incorrect)')
        plt.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Layer Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correct_vs_incorrect_head_usage.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已儲存正確 vs 錯誤預測比較圖")
    else:
        print(f"Warning: 正確或錯誤樣本數量不足，無法比較（正確: {len(correct_head_scores)}, 錯誤: {len(incorrect_head_scores)}）")

# ======================= 新增 claude 給的分析 function =======================

# =========================
# 視覺化主程式
# =========================

def visualize_sample(
    analyzer,
    X: torch.Tensor,
    y: torch.Tensor,
    sample_idx: int,
    save_dir: str,
    target_layer_idx: int = None,
    num_patches: int = 196,
    num_registers: int = 0,
    enable_advanced_analysis: bool = False,
    analysis_layer: int = None,
):
    """
    視覺化單一樣本：
    - 原圖
    - All heads relevance (heatmap + overlay)
    - Register relevance（若有）
    - Head relevance heatmap (所有層所有 head 的 relevance 分數)
    - Per-head relevance：預設迭代所有層（或僅 target_layer_idx 若指定）
    """
    device = next(analyzer.model.parameters()).device
    X_sample = X[sample_idx:sample_idx+1].to(device)
    y_sample = y[sample_idx:sample_idx+1].to(device)

    # ========= 0. 基本資訊 =========
    with torch.no_grad():
        logits = analyzer.model(X_sample)
        pred = logits.argmax(dim=1).cpu().item()
        conf = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().item()

    if y_sample.dim() > 1 and y_sample.shape[1] > 1:
        gt = y_sample.argmax(dim=1).cpu().item()
    else:
        gt = y_sample.cpu().item()

    is_correct = (pred == gt)
    status = "✓" if is_correct else "✗"

    # ========= 1. 原圖 =========
    X_vis = X_sample.detach().cpu()
    if X_vis.shape[1] == 3:
        X_vis = X_vis.permute(0, 2, 3, 1).contiguous()
    img = X_vis[0].numpy()
    img_denorm = denorm_img(img)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_denorm)
    plt.title(f"Sample {sample_idx} {status}\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}", fontsize=11, pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_original.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ========= 2. All heads relevance =========
    relevance_all, R_tokens_all = analyzer.explain(
        X_sample,
        target_class=None,
        return_map="image",
        upsample_to_input=True,
        restrict_heads=None,
    )
    rel_all = relevance_all[0, 0].detach().cpu().numpy()
    rel_all_vis = robust_normalize(rel_all)

    # (2-1) heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(rel_all_vis, cmap="jet")
    plt.colorbar(label="Relevance")
    plt.title(
        f"Sample {sample_idx} {status}\nAll Heads Relevance\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=10,
        pad=15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_relevance_all_heads.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # (2-2) overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(img_denorm)
    plt.imshow(rel_all_vis, cmap="jet", alpha=0.6)
    plt.title(
        f"Sample {sample_idx} {status}\nAll Heads Overlay\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=10,
        pad=15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_overlay_all_heads.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ========= 3. Register relevance（若有） =========
    if num_registers > 0:
        # R_tokens_all: [B, 1+P+R, dim]
        R_reg = R_tokens_all[:, 1+num_patches : 1+num_patches+num_registers]  # [B,R,dim]
        R_reg_scalar = R_reg.sum(dim=-1)[0].detach().cpu().numpy()            # [R]

        # normalize for plotting
        R_reg_norm = R_reg_scalar / (np.abs(R_reg_scalar).max() + 1e-6)

        plt.figure(figsize=(6, 4))
        xs = np.arange(num_registers)
        plt.bar(xs, R_reg_norm)
        plt.xticks(xs, [f"Reg{i}" for i in range(num_registers)])
        plt.ylim(-1.1, 1.1)
        plt.title(
            f"Sample {sample_idx} Register Token Relevance\n"
            f"Sum over feature dim (normalized)",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_register_relevance.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ========= 3.5. Head Relevance Heatmap (所有層所有 head) =========
    print(f"Sample {sample_idx}: 計算所有層所有 head 的 relevance heatmap...")
    plot_head_relevance_heatmap(
        analyzer,
        X_sample,
        sample_idx,
        save_dir,
        gt,
        pred,
        conf,
        status,
        num_patches=num_patches,
    )

    # ========= 4. Per-head relevance (全部或指定 layer) =========
    try:
        total_layers = get_num_layers(analyzer)
    except Exception as e:
        print(f"Warning: 無法取得總層數: {e}")
        return

    target_layers = (
        list(range(total_layers))
        if target_layer_idx is None or target_layer_idx < 0
        else [target_layer_idx]
    )

    for layer_idx in target_layers:
        try:
            num_heads = get_num_heads_from_layer(analyzer, layer_idx)
        except Exception as e:
            print(f"Warning: 無法取得 layer {layer_idx} 的 head 數量: {e}")
            continue

        print(
            f"Sample {sample_idx}: Layer {layer_idx} 有 {num_heads} 個 heads，"
            f"逐一計算 per-head relevance..."
        )

        rows = int(np.ceil(np.sqrt(num_heads)))
        cols = int(np.ceil(num_heads / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
        overlay_fig, overlay_axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))

        if num_heads == 1:
            axes = [axes]
            overlay_axes = [overlay_axes]
        else:
            axes = axes.flatten()
            overlay_axes = overlay_axes.flatten()

        for h in range(num_heads):
            print(f"  - layer {layer_idx}, head {h} relevance 計算中...")

            restrict_config = {layer_idx: [h]}

            relevance_h, _ = analyzer.explain(
                X_sample,
                target_class=None,
                return_map="image",
                upsample_to_input=True,
                restrict_heads=restrict_config,
            )
            rel_h = relevance_h[0, 0].detach().cpu().numpy()
            rel_h_vis = robust_normalize(rel_h)

            ax = axes[h]
            ax.imshow(rel_h_vis, cmap="jet")
            ax.set_title(f"Head {h}", fontsize=9)
            ax.axis("off")

            overlay_ax = overlay_axes[h]
            overlay_ax.imshow(img_denorm)
            overlay_ax.imshow(rel_h_vis, cmap="jet", alpha=0.6)
            overlay_ax.set_title(f"Head {h}", fontsize=9)
            overlay_ax.axis("off")

        # 關閉多餘子圖
        for j in range(num_heads, len(axes)):
            axes[j].axis("off")
            overlay_axes[j].axis("off")

        plt.suptitle(f"Sample {sample_idx} - Layer {layer_idx} Per-Head Relevance", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"sample_{sample_idx}_layer{layer_idx}_heads_grid.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        overlay_fig.suptitle(
            f"Sample {sample_idx} - Layer {layer_idx} Per-Head Relevance Overlay", fontsize=14
        )
        overlay_fig.tight_layout()
        overlay_fig.savefig(
            os.path.join(save_dir, f"sample_{sample_idx}_layer{layer_idx}_overlay_heads_grid.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(overlay_fig)

    # ========= 5. 進階分析（可選） =========
    if enable_advanced_analysis:
        print(f"\nSample {sample_idx}: 執行進階分析...")
        
        # 決定要分析的層
        if analysis_layer is not None:
            # 如果明確指定了 analysis_layer，只分析那一層
            analysis_layers = [analysis_layer]
        elif len(target_layers) > 0:
            # 如果指定了 target_layers（可能是所有層或特定層），使用所有 target_layers
            # 但為了避免計算時間過長，如果超過 12 層，只分析前 12 層
            if len(target_layers) > 12:
                print(f"  注意: 進階分析層數過多（{len(target_layers)} 層），只分析前 12 層以避免計算時間過長")
                analysis_layers = target_layers[:12]
            else:
                analysis_layers = target_layers
        else:
            # 如果沒有 target_layers，預設只分析前3層（避免計算時間過長）
            try:
                total_layers = get_num_layers(analyzer)
                analysis_layers = list(range(min(3, total_layers)))
                print(f"  注意: 未指定分析層，預設只分析前 {len(analysis_layers)} 層")
            except:
                analysis_layers = [0, 1, 2]  # 預設只分析前3層
                print(f"  注意: 未指定分析層，預設只分析前 3 層")
        
        for layer_idx in analysis_layers:
            try:
                print(f"  - Layer {layer_idx}: Head 語義分析...")
                analyze_head_semantics_advanced(
                    analyzer, X_sample, layer_idx, save_dir, sample_idx
                )
                
                print(f"  - Layer {layer_idx}: Head 相似度分析...")
                similarity_matrix = compute_head_similarity_matrix(analyzer, X_sample, layer_idx)
                plot_head_similarity_heatmap(similarity_matrix, layer_idx, save_dir, sample_idx)
            except Exception as e:
                print(f"  Warning: Layer {layer_idx} 進階分析失敗: {e}")
        
        # Register 交互分析
        if num_registers > 0:
            try:
                print(f"  - Register 交互分析...")
                analyze_register_interaction(
                    analyzer, X_sample, save_dir, sample_idx, num_registers, num_patches
                )
            except Exception as e:
                print(f"  Warning: Register 交互分析失敗: {e}")


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="視覺化 RegisterViT + Register-aware Partial LRP 的 per-head relevance"
    )
    parser.add_argument("--num_samples", type=int, default=4,
                        help="要視覺化的樣本數量")
    parser.add_argument("--target_layer", type=int, default=-1,
                        help="想看的 encoder layer index；<0 代表全部層")
    parser.add_argument("--save_dir", type=str,
                        default="./vit_register_lrp_vis",
                        help="輸出資料夾")
    parser.add_argument("--num_patches", type=int, default=196,
                        help="patch 數量（預設 196 = 14x14）")
    parser.add_argument("--num_registers", type=int, default=1,
                        help="register token 數量（如果 RegisterViT 有設會被覆蓋）")
    parser.add_argument("--enable_advanced_analysis", action="store_true",
                        help="啟用進階分析（語義分析、相似度分析、register 交互分析）")
    parser.add_argument("--analysis_layer", type=int, default=None,
                        help="進階分析要分析的層（預設為前3層）")
    parser.add_argument("--batch_analysis_only", action="store_true",
                        help="只執行批次分析（正確 vs 錯誤預測比較），跳過單樣本視覺化")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # dataloader
    print("建立資料載入器與模型...")
    train_loader, test_loader = get_dataloader(
        dataset=config["dataset"],
        root=os.path.join(config["root"], "data"),
        batch_size=config["batch_size"],
        input_size=config["input_shape"],
    )

    # 建立 model + analyzer
    base_model = build_model_and_load_weights(device)
    analyzer = build_analyzer_from_model(
        base_model,
        device=device,
        num_patches=args.num_patches,
        num_registers=args.num_registers,
    )

    # 取樣本
    N = max(1, int(args.num_samples))
    X, y = collect_n_samples(test_loader, device, N)
    if X is None:
        X, y = collect_n_samples(train_loader, device, N)
    if X is None:
        raise RuntimeError("無法從資料載入器中取得樣本。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"結果將輸出到: {args.save_dir}")

    # 如果只執行批次分析，跳過單樣本視覺化
    if args.batch_analysis_only:
        print(f"\n批次分析模式：跳過單樣本視覺化，只執行正確 vs 錯誤預測比較...")
        print(f"處理 {N} 個樣本...")
        try:
            compare_correct_vs_incorrect_predictions(analyzer, X, y, args.save_dir)
        except Exception as e:
            print(f"Error: 批次分析失敗: {e}")
            raise
    else:
        # 正常模式：執行單樣本視覺化
        for i in range(min(N, X.size(0))):
            print(f"\n處理樣本 {i} ...")
            visualize_sample(
                analyzer,
                X,
                y,
                sample_idx=i,
                save_dir=args.save_dir,
                target_layer_idx=(None if args.target_layer < 0 else args.target_layer),
                num_patches=args.num_patches,
                num_registers=args.num_registers,
                enable_advanced_analysis=args.enable_advanced_analysis,
                analysis_layer=args.analysis_layer,
            )

        # 批次分析：正確 vs 錯誤預測比較（如果樣本數足夠且啟用進階分析）
        if args.enable_advanced_analysis and N >= 4:
            print("\n執行批次分析：正確 vs 錯誤預測比較...")
            try:
                compare_correct_vs_incorrect_predictions(analyzer, X, y, args.save_dir)
            except Exception as e:
                print(f"Warning: 批次分析失敗: {e}")

    print("\n全部樣本處理完成。")


if __name__ == "__main__":
    main()
