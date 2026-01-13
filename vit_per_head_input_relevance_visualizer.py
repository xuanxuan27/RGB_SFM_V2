#!/usr/bin/env python3
"""
視覺化：最接近 input 的 transformer layer，每個 head 的 input relevance map

特色：
- 支援兩種 analyzer：
  1) VIT_with_Partial_LRP
  2) VIT_PartialLRP_PatchMerging
- 每個 head 會各自產生：
  - 單獨的 relevance heatmap
  - 疊加在原圖上的 relevance heatmap
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
from models.VIT_with_Partial_LRP import VIT_with_Partial_LRP
from models.VIT_PartialLRP_PatchMerging import VIT_PartialLRP_PatchMerging
from models.VIT_with_Partial_LRP_RegisterAware import VIT_with_Partial_LRP_RegisterAware
from models.RegisterViT import RegisterViT


# =========================
# 工具函式
# =========================

def _latest_best_checkpoint(model_name: str = "VIT") -> str:
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
    仿照 vit_patch_merging_visualizer.py：
    先依照 config 建立訓練時的模型，載入 checkpoint 權重，再交給後續建立 analyzer 使用。
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_args = dict(model_cfg["args"])

    print(f"建立模型: {model_name}，參數: {model_args}")
    model = getattr(getattr(models, model_name), model_name)(**model_args)
    model = model.to(device).eval()

    # 載入權重（先載到原始模型，再包 analyzer，確保 MHAWrapper 的 buffer 從正確權重推導）
    ckpt_path = "runs/train/exp119/RegisterViT_best.pth"
    if os.path.exists(ckpt_path):
        print(f"載入權重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt is not None and "model_weights" in ckpt:
            weights = ckpt["model_weights"]
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            print(f"原始模型載入結果 - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        else:
            print("checkpoint 中找不到 'model_weights'，略過載入。")
    else:
        print(f"找不到權重檔 {ckpt_path}，使用當前隨機初始化權重。")

    return model


def build_analyzer_from_model(model, analyzer_type: str, device, eps: float = 1e-6):
    """
    根據 analyzer_type 從已載好權重的 model 建立對應的 analyzer。
    analyzer_type: 'vit_lrp' 或 'vit_patch_merging'
    
    如果偵測到 RegisterViT，會自動使用 VIT_with_Partial_LRP_RegisterAware。
    """
    # 如果本身已經是 analyzer，直接用
    if isinstance(model, (VIT_with_Partial_LRP, VIT_PartialLRP_PatchMerging, VIT_with_Partial_LRP_RegisterAware)):
        return model.to(device).eval()

    # 檢查是否為 RegisterViT
    is_register_vit = isinstance(model, RegisterViT)
    num_registers = 0
    if is_register_vit:
        num_registers = model.num_registers
        print(f"偵測到 RegisterViT，將使用 VIT_with_Partial_LRP_RegisterAware（num_registers={num_registers}）")
        # RegisterViT 要使用完整模型，不要只取 backbone
        vit_backbone = model
    elif hasattr(model, "backbone"):
        print("偵測到包裝器模型，使用其 backbone 作為 ViT 主體。")
        vit_backbone = model.backbone
    else:
        vit_backbone = model

    # 如果是 RegisterViT，強制使用 RegisterAware analyzer
    if is_register_vit:
        model_args = dict(config["model"]["args"])
        num_patches = model_args.get("input_size", (224, 224))
        if isinstance(num_patches, tuple):
            num_patches = (num_patches[0] // 16) * (num_patches[1] // 16)  # 計算 patch 數量
        else:
            num_patches = 196  # 預設 ViT-B/16
        
        print(f"使用 VIT_with_Partial_LRP_RegisterAware 做為 analyzer（num_patches={num_patches}, num_registers={num_registers}）")
        analyzer = VIT_with_Partial_LRP_RegisterAware(
            vit_model=vit_backbone,
            num_patches=num_patches,
            num_registers=num_registers,
            eps=eps,
        ).to(device).eval()
        return analyzer

    # 非 RegisterViT 的情況，使用原本的邏輯
    if analyzer_type == "vit_lrp":
        print("使用 VIT_with_Partial_LRP 做為 analyzer")
        analyzer = VIT_with_Partial_LRP(
            vit_backbone,
            topk_heads=None,
            head_weighting="none",
            eps=eps,
        ).to(device).eval()
    elif analyzer_type == "vit_patch_merging":
        print("使用 VIT_PartialLRP_PatchMerging 做為 analyzer")
        model_args = dict(config["model"]["args"])
        in_channels = model_args.get("in_channels", 3)
        out_channels = model_args.get("out_channels", 10)
        input_size = model_args.get("input_size", (224, 224))

        analyzer = VIT_PartialLRP_PatchMerging(
            vit_model=vit_backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            topk_heads=None,
            head_weighting="none",
            eps=eps,
            enable_patch_merging=True,
        ).to(device).eval()
    else:
        raise ValueError(f"未知的 analyzer_type: {analyzer_type}")

    return analyzer


# def visualize_per_head_input_relevance(
#     analyzer,
#     X: torch.Tensor,
#     y: torch.Tensor,
#     sample_idx: int,
#     save_dir: str,
#     target_layer_idx: int = 0,
# ):
#     """
#     對單一樣本，針對指定的 transformer layer (target_layer_idx) 的每個 head
#     分別計算「只保留該 head」時的 image-level relevance，並存圖。
#     同時也會產生「使用所有 heads」的 relevance map（定義與 vit_patch_merging_visualizer 相同）。
#     """
#     device = next(analyzer.model.parameters()).device
#     X_sample = X[sample_idx : sample_idx + 1].to(device)
#     y_sample = y[sample_idx : sample_idx + 1].to(device)

#     # 取得 baseline 預測資訊（全部 head）
#     analyzer.clear_head_mask()
#     with torch.no_grad():
#         logits = analyzer.model(X_sample)
#         pred = logits.argmax(dim=1).cpu().item()
#         conf = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().item()

#     # ground truth
#     if y_sample.dim() > 1 and y_sample.shape[1] > 1:
#         gt = y_sample.argmax(dim=1).cpu().item()
#     else:
#         gt = y_sample.cpu().item()

#     is_correct = pred == gt
#     status = "✓" if is_correct else "✗"

#     # 準備原圖（反正規化）
#     X_vis = X_sample.detach().cpu()
#     if X_vis.shape[1] == 3:
#         X_vis = X_vis.permute(0, 2, 3, 1).contiguous()
#     img = X_vis[0].numpy()
#     img_denorm = denorm_img(img)

#     # 先把原圖輸出一次
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_denorm)
#     title = f"Sample {sample_idx} {status}\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}"
#     plt.title(title, fontsize=11, pad=15)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(save_dir, f"sample_{sample_idx}_original.png"),
#         dpi=150,
#         bbox_inches="tight",
#     )
#     plt.close()

#     # ===============================
#     # 先計算「所有 heads」共同作用下的 input relevance
#     # （使用 explain(..., return_map='image')，與 vit_patch_merging_visualizer 保持一致）
#     # ===============================
#     analyzer.clear_head_mask()
#     relevance_all = analyzer.explain(
#         X_sample,
#         target_class=pred,
#         return_map="image",
#         upsample_to_input=True,
#     )
#     rel_all = relevance_all[0].detach().cpu().numpy()  # [H, W]

#     # 正規化到 [0, 1]
#     rmin_all, rmax_all = rel_all.min(), rel_all.max()
#     if rmax_all > rmin_all:
#         rel_all_norm = (rel_all - rmin_all) / (rmax_all - rmin_all)
#     else:
#         rel_all_norm = rel_all

#     # (A) 所有 heads 的單獨 heatmap
#     plt.figure(figsize=(6, 6))
#     plt.imshow(rel_all_norm, cmap="jet")
#     plt.colorbar(label="Relevance Score")
#     plt.title(
#         f"Sample {sample_idx} {status}\n"
#         f"All Heads Relevance\n"
#         f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
#         fontsize=10,
#         pad=15,
#     )
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(
#             save_dir, f"sample_{sample_idx}_input_relevance_all_heads.png"
#         ),
#         dpi=150,
#         bbox_inches="tight",
#     )
#     plt.close()

#     # (B) 所有 heads 疊加在原圖上的 heatmap
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_denorm)
#     plt.imshow(rel_all_norm, cmap="jet", alpha=0.6)
#     plt.title(
#         f"Sample {sample_idx} {status}\n"
#         f"All Heads Overlay\n"
#         f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
#         fontsize=10,
#         pad=15,
#     )
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(
#             save_dir, f"sample_{sample_idx}_input_relevance_overlay_all_heads.png"
#         ),
#         dpi=150,
#         bbox_inches="tight",
#     )
#     plt.close()

#     # 從 analyzer 的 attn_wrappers 拿 head 數量
#     if not hasattr(analyzer, "attn_wrappers") or len(analyzer.attn_wrappers) == 0:
#         print("Warning: analyzer 沒有 attn_wrappers，無法取得 head 數量。")
#         return

#     if target_layer_idx < 0 or target_layer_idx >= len(analyzer.attn_wrappers):
#         print(
#             f"Warning: target_layer_idx={target_layer_idx} 超出範圍 "
#             f"(共有 {len(analyzer.attn_wrappers)} 個 attention blocks)。"
#         )
#         return

#     mha_wrapper = analyzer.attn_wrappers[target_layer_idx]
#     num_heads = mha_wrapper.mha.num_heads

#     print(
#         f"Sample {sample_idx}: 在 layer {target_layer_idx} 上共有 {num_heads} 個 heads，"
#         f"逐一計算 input relevance..."
#     )

#     for head_idx in range(num_heads):
#         print(f"  - layer {target_layer_idx}, head {head_idx} relevance 計算中...")

#         # 每次只保留單一 head
#         analyzer.clear_head_mask()
#         layer_to_heads = {target_layer_idx: [head_idx]}
#         analyzer.set_head_mask(layer_to_heads, keep_only=True)

#         # 計算 image-level relevance
#         relevance_map = analyzer.explain(
#             X_sample,
#             target_class=pred,
#             return_map="image",
#             upsample_to_input=True,
#         )
#         rel = relevance_map[0].detach().cpu().numpy()  # [H, W]

#         # 正規化到 [0, 1]
#         rmin, rmax = rel.min(), rel.max()
#         if rmax > rmin:
#             rel_norm = (rel - rmin) / (rmax - rmin)
#         else:
#             rel_norm = rel

#         suffix = f"layer{target_layer_idx}_head{head_idx}"

#         # (1) 單獨 heatmap
#         plt.figure(figsize=(6, 6))
#         plt.imshow(rel_norm, cmap="jet")
#         plt.colorbar(label="Relevance Score")
#         plt.title(
#             f"Sample {sample_idx} {status}\n"
#             f"Layer {target_layer_idx} Head {head_idx} Relevance\n"
#             f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
#             fontsize=10,
#             pad=15,
#         )
#         plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir, f"sample_{sample_idx}_input_relevance_{suffix}.png"
#             ),
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.close()

#         # (2) 疊加在原圖上
#         plt.figure(figsize=(6, 6))
#         plt.imshow(img_denorm)
#         plt.imshow(rel_norm, cmap="jet", alpha=0.6)
#         plt.title(
#             f"Sample {sample_idx} {status}\n"
#             f"Layer {target_layer_idx} Head {head_idx} Overlay\n"
#             f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
#             fontsize=10,
#             pad=15,
#         )
#         plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir, f"sample_{sample_idx}_input_relevance_overlay_{suffix}.png"
#             ),
#             dpi=150,
#             bbox_inches="tight",
#         )
#         plt.close()

#         # 清理 cache，降低顯存 / 記憶體壓力
#         if hasattr(analyzer, "_clear_memory_cache"):
#             analyzer._clear_memory_cache()

#     # 最後恢復成 all heads
#     analyzer.clear_head_mask()

def get_num_heads_from_analyzer(analyzer, layer_idx: int) -> int:
    """
    從 analyzer 取得指定 layer 的 head 數量。
    支援 VIT_with_Partial_LRP（有 attn_wrappers）和 VIT_with_Partial_LRP_RegisterAware（沒有 attn_wrappers）。
    """
    # 方法 1: 有 attn_wrappers 的情況（VIT_with_Partial_LRP）
    if hasattr(analyzer, "attn_wrappers") and len(analyzer.attn_wrappers) > 0:
        if layer_idx < 0 or layer_idx >= len(analyzer.attn_wrappers):
            raise ValueError(f"layer_idx={layer_idx} 超出範圍 (共有 {len(analyzer.attn_wrappers)} 個 layers)")
        mha_wrapper = analyzer.attn_wrappers[layer_idx]
        return mha_wrapper.mha.num_heads
    
    # 方法 2: 沒有 attn_wrappers 的情況（VIT_with_Partial_LRP_RegisterAware）
    if hasattr(analyzer, "encoder"):
        encoder = analyzer.encoder
        if layer_idx < 0 or layer_idx >= len(encoder.layers):
            raise ValueError(f"layer_idx={layer_idx} 超出範圍 (共有 {len(encoder.layers)} 個 layers)")
        blk = encoder.layers[layer_idx]
        attn = blk.self_attention
        if hasattr(attn, "num_heads"):
            return attn.num_heads
        # 預設 ViT-B/16 = 12 heads
        return 12
    
    raise ValueError("無法從 analyzer 取得 head 數量，analyzer 結構不符預期")


def robust_normalize(data):
    """
    使用 99 分位數進行正規化，避免極端值導致整個熱圖看不見。
    同時保留正負號的資訊結構（雖然最後通常取絕對值或 0-1，但在中間處理很重要）。
    這裡回傳 [0, 1] 供 imshow 使用。
    """
    # 取絕對值後找 99% 分位數作為最大值
    vmax = np.percentile(np.abs(data), 99)
    if vmax == 0: 
        vmax = 1e-12
    
    # 截斷極端值
    data_clipped = np.clip(data, -vmax, vmax)
    
    # 正規化到 [0, 1] (將 -vmax 對應到 0, vmax 對應到 1)
    # 這樣 0 會在 0.5 的位置（灰色），正值偏紅，負值偏藍
    norm_data = (data_clipped + vmax) / (2 * vmax)
    return norm_data

def visualize_per_head_input_relevance(
    analyzer,
    X: torch.Tensor,
    y: torch.Tensor,
    sample_idx: int,
    save_dir: str,
    target_layer_idx: int = 0,
):
    """
    對單一樣本，針對指定的 transformer layer (target_layer_idx) 的每個 head
    分別計算「只保留該 head」時的 image-level relevance (Backward Filtering)。
    """
    device = next(analyzer.model.parameters()).device
    X_sample = X[sample_idx : sample_idx + 1].to(device)
    y_sample = y[sample_idx : sample_idx + 1].to(device)

    # 1. 確保 Forward Pass 是「完整」的 (不使用 head_mask)
    analyzer.clear_head_mask()
    
    # 取得 baseline 預測資訊
    with torch.no_grad():
        logits = analyzer.model(X_sample)
        pred = logits.argmax(dim=1).cpu().item()
        conf = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().item()

    # ground truth
    if y_sample.dim() > 1 and y_sample.shape[1] > 1:
        gt = y_sample.argmax(dim=1).cpu().item()
    else:
        gt = y_sample.cpu().item()

    is_correct = pred == gt
    status = "✓" if is_correct else "✗"

    # 準備原圖
    X_vis = X_sample.detach().cpu()
    if X_vis.shape[1] == 3:
        X_vis = X_vis.permute(0, 2, 3, 1).contiguous()
    img = X_vis[0].numpy()
    img_denorm = denorm_img(img)

    # Save Original
    plt.figure(figsize=(4, 4))
    plt.imshow(img_denorm)
    plt.title(f"GT: {gt} | Pred: {pred} {status}", fontsize=10)
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_orig.png"), bbox_inches="tight", dpi=100)
    plt.close()

    # ===============================
    # (A) All Heads (Baseline)
    # ===============================
    relevance_all = analyzer.explain(
        X_sample,
        target_class=None, # 使用預測類別
        return_map="image",
        upsample_to_input=True,
        restrict_heads=None # [修正] 不遮蔽任何 head
    )
    # VIT_with_Partial_LRP_RegisterAware 回傳 (R_img, R)，其中 R_img 是 [B, 1, H, W]
    # VIT_with_Partial_LRP 回傳 [B, H, W]
    if isinstance(relevance_all, tuple):
        rel_tensor = relevance_all[0]  # [B, 1, H, W] 或 [B, H, W]
        if rel_tensor.dim() == 4:
            rel_all = rel_tensor[0, 0].detach().cpu().numpy()  # [H, W]
        else:
            rel_all = rel_tensor[0].detach().cpu().numpy()  # [H, W]
    else:
        # VIT_with_Partial_LRP 回傳 [B, H, W]
        rel_all = relevance_all[0].detach().cpu().numpy()  # [H, W]
    
    # 使用 Robust Normalization
    rel_all_vis = robust_normalize(rel_all)

    plt.figure(figsize=(4, 4))
    plt.imshow(rel_all_vis, cmap="jet") # jet 對應 0-1 範圍
    plt.title("All Heads", fontsize=10)
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_all.png"), bbox_inches="tight", dpi=100)
    plt.close()

    # ===============================
    # (B) Per Head Visualization (Backward Masking)
    # ===============================
    try:
        num_heads = get_num_heads_from_analyzer(analyzer, target_layer_idx)
    except ValueError as e:
        print(f"Warning: {e}")
        return

    print(f"Sample {sample_idx}: Layer {target_layer_idx} ({num_heads} heads)")

    # 先收集所有 head 的 relevance maps（避免重複計算）
    all_head_relevances = []
    
    for head_idx in range(num_heads):
        # [核心修正] 使用 restrict_heads 參數，只在 LRP 計算時遮蔽
        restrict_config = {target_layer_idx: [head_idx]}
        
        relevance_map = analyzer.explain(
            X_sample,
            target_class=None,
            return_map="image",
            upsample_to_input=True,
            restrict_heads=restrict_config # [修正] 傳入過濾設定
        )
        # VIT_with_Partial_LRP_RegisterAware 回傳 (R_img, R)，其中 R_img 是 [B, 1, H, W]
        # VIT_with_Partial_LRP 回傳 [B, H, W]
        if isinstance(relevance_map, tuple):
            rel_tensor = relevance_map[0]  # [B, 1, H, W] 或 [B, H, W]
            if rel_tensor.dim() == 4:
                rel = rel_tensor[0, 0].detach().cpu().numpy()  # [H, W]
            else:
                rel = rel_tensor[0].detach().cpu().numpy()  # [H, W]
        else:
            # VIT_with_Partial_LRP 回傳 [B, H, W]
            rel = relevance_map[0].detach().cpu().numpy()  # [H, W]
        all_head_relevances.append(rel)
    
    # ===============================
    # (B1) 個別 Head Grid 圖
    # ===============================
    rows = int(np.ceil(np.sqrt(num_heads)))
    cols = int(np.ceil(num_heads / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        rel = all_head_relevances[head_idx]
        # 使用 Robust Normalization (讓微弱訊號也能被看見)
        rel_vis = robust_normalize(rel)

        ax = axes[head_idx]
        ax.imshow(rel_vis, cmap="jet")
        ax.set_title(f"Head {head_idx}", fontsize=9)
        ax.axis("off")

    # 關閉多餘的 subplot
    for j in range(num_heads, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(f"Layer {target_layer_idx} Per-Head Relevance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_layer{target_layer_idx}_heads_grid.png"), dpi=150)
    plt.close()
    
    # ===============================
    # (B2) 完整視覺化：原圖 + All Heads + 所有個別 Head
    # ===============================
    # 計算需要的 subplot 數量：1 (原圖) + 1 (All Heads) + num_heads (個別 head)
    total_plots = 2 + num_heads
    # 計算合適的 grid 大小
    cols_complete = int(np.ceil(np.sqrt(total_plots)))
    rows_complete = int(np.ceil(total_plots / cols_complete))
    
    fig_complete, axes_complete = plt.subplots(
        rows_complete, cols_complete, 
        figsize=(cols_complete * 3, rows_complete * 3)
    )
    if total_plots == 1:
        axes_complete = [axes_complete]
    else:
        axes_complete = axes_complete.flatten()
    
    plot_idx = 0
    
    # 1. 原圖
    axes_complete[plot_idx].imshow(img_denorm)
    axes_complete[plot_idx].set_title(f"Original\nGT: {gt} | Pred: {pred} {status}", fontsize=10)
    axes_complete[plot_idx].axis("off")
    plot_idx += 1
    
    # 2. All Heads (Baseline)
    axes_complete[plot_idx].imshow(rel_all_vis, cmap="jet")
    axes_complete[plot_idx].set_title("All Heads", fontsize=10)
    axes_complete[plot_idx].axis("off")
    plot_idx += 1
    
    # 3. 每個個別 Head（使用已計算的結果）
    for head_idx in range(num_heads):
        rel = all_head_relevances[head_idx]
        rel_vis = robust_normalize(rel)
        
        axes_complete[plot_idx].imshow(rel_vis, cmap="jet")
        axes_complete[plot_idx].set_title(f"Head {head_idx}", fontsize=9)
        axes_complete[plot_idx].axis("off")
        plot_idx += 1
    
    # 關閉多餘的 subplot
    for j in range(plot_idx, len(axes_complete)):
        axes_complete[j].axis("off")
    
    plt.suptitle(
        f"Sample {sample_idx} {status} - Layer {target_layer_idx} Complete View\n"
        f"GT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_layer{target_layer_idx}_complete.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    
    # 清理
    analyzer._clear_memory_cache()


def main():
    parser = argparse.ArgumentParser(
        description="視覺化最接近 input 的 transformer layer，每個 head 的 input relevance map"
    )
    parser.add_argument(
        "--analyzer",
        type=str,
        default="vit_lrp",
        choices=["vit_lrp", "vit_patch_merging"],
        help="選擇使用哪一個 analyzer：'vit_lrp' (VIT_with_Partial_LRP) 或 'vit_patch_merging' (VIT_PartialLRP_PatchMerging)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="要視覺化的樣本數量（從 test_loader / train_loader 前面依序取）",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=0,
        help="最接近 input 的 transformer layer index，預設為 0（第一層）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./vit_per_head_input_relevance",
        help="結果輸出資料夾",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # dataloader 與 base model
    print("建立資料載入器與模型...")
    train_loader, test_loader = get_dataloader(
        dataset=config["dataset"],
        root=os.path.join(config["root"], "data"),
        batch_size=config["batch_size"],
        input_size=config["input_shape"],
    )

    # 先按照訓練時結構建立模型並載入權重，再包成 analyzer
    base_model = build_model_and_load_weights(device)
    analyzer = build_analyzer_from_model(
        base_model,
        analyzer_type=args.analyzer,
        device=device,
    )

    # 蒐集樣本
    N = max(1, int(args.num_samples))
    X, y = collect_n_samples(test_loader, device, N)
    if X is None:
        X, y = collect_n_samples(train_loader, device, N)
    if X is None:
        raise RuntimeError("無法從任何資料載入器中取得樣本。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"結果將輸出到: {args.save_dir}")

    for i in range(min(N, X.size(0))):
        print(f"\n處理樣本 {i} ...")
        visualize_per_head_input_relevance(
            analyzer,
            X,
            y,
            sample_idx=i,
            save_dir=args.save_dir,
            target_layer_idx=args.target_layer,
        )

    print(f"\n所有樣本處理完成！結果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()


