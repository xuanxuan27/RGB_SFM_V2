#!/usr/bin/env python3
"""
視覺化：使用 VIT_with_Partial_LRP 中的 get_cls_attention_map
輸出：
  - 原始影像（反正規化）
  - 最後一層 CLS → patch attention map（單獨 heatmap）
  - CLS attention 疊加在原圖上的 heatmap

參考：
  - vit_per_head_input_relevance_visualizer.py
  - models/VIT_with_Partial_LRP.py
"""

import os
import glob
import argparse
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from config import config
from dataloader import get_dataloader
import models
from models.VIT_with_Partial_LRP import VIT_with_Partial_LRP
from models.VIT_PartialLRP_PatchMerging import VIT_PartialLRP_PatchMerging


# =========================
# 工具函式
# =========================

def _latest_best_checkpoint(model_name: str = "VIT") -> str:
    """
    嘗試尋找 runs/train/exp*/{model_name}_best.pth
    找不到時回傳空字串。
    （這裡保留與原 visualizer 一致的介面，但實際上會固定用 exp108）
    """
    pattern = os.path.join("runs", "train", "exp*", f"{model_name}_best.pth")
    cand = sorted(glob.glob(pattern))
    if len(cand) > 0:
        return cand[-1]
    fixed = os.path.join("runs", "train", "exp", f"{model_name}_best.pth")

    return fixed if os.path.exists(fixed) else ""


def denorm_img(
    img_hw3: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
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
    與 vit_per_head_input_relevance_visualizer 保持一致。
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
    仿照 vit_per_head_input_relevance_visualizer.py：
    先依照 config 建立訓練時的模型，載入 checkpoint 權重，再交給後續建立 analyzer 使用。
    這裡固定使用 exp108 的 VIT_best.pth。
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_args = dict(model_cfg["args"])

    print(f"建立模型: {model_name}，參數: {model_args}")
    model = getattr(getattr(models, model_name), model_name)(**model_args)
    model = model.to(device).eval()

    # 載入權重（與 vit_per_head_input_relevance_visualizer 相同：exp108）
    ckpt_path = "runs/train/exp117/RegisterViT_best.pth"
    if os.path.exists(ckpt_path):
        print(f"載入權重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt is not None and "model_weights" in ckpt:
            weights = ckpt["model_weights"]
            missing_keys, unexpected_keys = model.load_state_dict(
                weights, strict=False
            )
            print(
                f"原始模型載入結果 - Missing keys: {len(missing_keys)}, "
                f"Unexpected keys: {len(unexpected_keys)}"
            )
        else:
            print("checkpoint 中找不到 'model_weights'，略過載入。")
    else:
        print(f"找不到權重檔 {ckpt_path}，使用當前隨機初始化權重。")

    return model


def build_analyzer_from_model(model, analyzer_type: str, device, eps: float = 1e-6):
    """
    根據 analyzer_type 從已載好權重的 model 建立對應的 analyzer。
    主要是為了使用 VIT_with_Partial_LRP（其中包含 get_cls_attention_map）。
    """
    # 如果本身已經是 analyzer，直接用
    if isinstance(model, (VIT_with_Partial_LRP, VIT_PartialLRP_PatchMerging)):
        return model.to(device).eval()

    # 檢查是否為 RegisterViT（有 register tokens 的模型）
    is_register_vit = False
    num_registers = 0
    if hasattr(model, "num_registers") and model.num_registers > 0:
        is_register_vit = True
        num_registers = model.num_registers
        print(f"偵測到 RegisterViT 模型（有 {num_registers} 個 register tokens）")

    # 若是外層包裝器（例如 VIT），則取 backbone
    if hasattr(model, "backbone"):
        print("偵測到包裝器模型，使用其 backbone 作為 ViT 主體。")
        vit_backbone = model.backbone
        
        # 如果是 RegisterViT，需要恢復 pos_embedding 到原始大小（197 tokens）
        # 因為 VIT_with_Partial_LRP 只處理標準 ViT，不包含 register tokens
        if is_register_vit and hasattr(vit_backbone, "encoder") and hasattr(vit_backbone.encoder, "pos_embedding"):
            current_pos = vit_backbone.encoder.pos_embedding
            current_len = current_pos.shape[1]
            # 標準 ViT-B/16 有 197 個 tokens (1 CLS + 196 patches)
            standard_len = 197
            if current_len > standard_len:
                print(f"恢復 pos_embedding 從 {current_len} 到 {standard_len} tokens（移除 register tokens）")
                # 只保留前 197 個 tokens（CLS + patches）
                restored_pos = current_pos[:, :standard_len, :].clone()
                vit_backbone.encoder.pos_embedding = nn.Parameter(restored_pos)
    else:
        vit_backbone = model

    if analyzer_type == "vit_lrp":
        print("使用 VIT_with_Partial_LRP 做為 analyzer（提供 get_cls_attention_map）")
        analyzer = VIT_with_Partial_LRP(
            vit_backbone,
            topk_heads=None,
            head_weighting="none",
            eps=eps,
        ).to(device).eval()
        
        # 建立 analyzer 後，再次檢查並恢復 pos_embedding（因為 VIT_with_Partial_LRP 會複製模型）
        if is_register_vit and hasattr(analyzer.model, "encoder") and hasattr(analyzer.model.encoder, "pos_embedding"):
            current_pos = analyzer.model.encoder.pos_embedding
            current_len = current_pos.shape[1]
            standard_len = 197
            if current_len > standard_len:
                print(f"再次恢復 analyzer 的 pos_embedding 從 {current_len} 到 {standard_len} tokens")
                restored_pos = current_pos[:, :standard_len, :].clone()
                analyzer.model.encoder.pos_embedding = nn.Parameter(restored_pos)
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


def visualize_cls_attention_for_sample(
    analyzer,
    X: torch.Tensor,
    y: torch.Tensor,
    sample_idx: int,
    save_dir: str,
):
    """
    對單一樣本：
      - 輸出原圖
      - 輸出 CLS attention heatmap
      - 輸出 CLS attention 疊加在原圖上的 heatmap
    """
    device = next(analyzer.model.parameters()).device
    X_sample = X[sample_idx : sample_idx + 1].to(device)
    y_sample = y[sample_idx : sample_idx + 1].to(device)

    # baseline prediction
    analyzer.clear_head_mask() if hasattr(analyzer, "clear_head_mask") else None
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

    # 準備原圖（反正規化）
    X_vis = X_sample.detach().cpu()
    if X_vis.shape[1] == 3:
        X_vis = X_vis.permute(0, 2, 3, 1).contiguous()
    img = X_vis[0].numpy()
    img_denorm = denorm_img(img)

    # (1) 存原圖
    plt.figure(figsize=(4, 4))
    plt.imshow(img_denorm)
    plt.title(
        f"Sample {sample_idx} {status}\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=9,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_original.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # (2) 取得 CLS attention map
    # 需要先跑一遍 forward（上面已跑過一次）；此時 analyzer.cache 已就緒
    with torch.no_grad():
        cls_attn = analyzer.get_cls_attention_map()  # [B, 1, H, W]，已經 upsample & normalize

    attn_map = cls_attn[0, 0].detach().cpu().numpy()  # [H, W]

    # (3) 單獨 CLS attention heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(attn_map, cmap="jet")
    plt.colorbar(label="Attention Weight")
    plt.title("CLS Attention Map", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_cls_attention.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()

    # (4) CLS attention 疊加在原圖上
    plt.figure(figsize=(4, 4))
    plt.imshow(img_denorm)
    plt.imshow(attn_map, cmap="jet", alpha=0.6)
    plt.title(
        f"CLS Attention Overlay\nGT: {gt}, Pred: {pred}, Conf: {conf:.3f}",
        fontsize=9,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"sample_{sample_idx}_cls_attention_overlay.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="視覺化 ViT 最後一層 CLS→patch attention map（原圖 + attention + overlay）"
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
        default=1,
        help="要視覺化的樣本數量（從 test_loader / train_loader 前面依序取）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./vit_cls_attention_maps",
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
        visualize_cls_attention_for_sample(
            analyzer,
            X,
            y,
            sample_idx=i,
            save_dir=args.save_dir,
        )

    print(f"\n所有樣本處理完成！結果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()


