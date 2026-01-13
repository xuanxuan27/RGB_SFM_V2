#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PVTv2 (timm) Attention Visualization via Monkey Patch (Clean / Paper-like)

你目前覺得「醜」的核心原因通常是：
- 每張子圖各自 colorbar（不可比、很擠）
- 每個 head 使用不同 colormap（像彩虹拼貼）
- vmin/vmax 不一致（顏色無法比較）
- overlay 整張染色（原圖變髒）

本版本改成：
1) heatmap montage：同一 layer 的所有 head 統一 vmin/vmax (percentile clip)，只放一個 shared colorbar
2) overlay montage：固定 colormap，並且「只在高注意力區域上色」(mask_top_pct)
3) top-k：改成「點 + rank」，比框格子乾淨
4) 強制白底輸出，不吃到你環境的 dark style

用法：
python pvtv2_attn_vis_clean.py --ckpt runs/train/exp125/PVTv2_B0_best.pth --save_dir out_vis --device cuda --k 5

Author: for your project use
"""

import os
import math
import argparse
from typing import Dict, Tuple, Any, List

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---- Your project imports (adjust if needed) ----
from models.PVTv2_B0 import PVTv2_B0
from dataloader import get_dataloader
import config


# ============================================================
# 0) Matplotlib defaults (force clean white background)
# ============================================================

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")


# ============================================================
# 1) Monkey patch: patch each attention module's forward method
# ============================================================

def _is_pvt_attention_module(m: nn.Module) -> bool:
    """Heuristic to detect timm PVTv2 Attention module."""
    needed = ["q", "kv", "num_heads", "attn_drop", "proj", "proj_drop"]
    return all(hasattr(m, k) for k in needed)


def monkey_patch_pvt_attn_forward(
    attn_module: nn.Module,
    store: Dict[str, torch.Tensor],
    name: str,
) -> None:
    """
    Monkey-patch a single attention module's forward method.
    It will compute the SAME forward as timm's PVTv2 attention,
    but also store attn weights (softmax after scaling, before dropout) into store[name].
    """
    if hasattr(attn_module, "_orig_forward"):
        return

    orig_forward = attn_module.forward
    attn_module._orig_forward = orig_forward

    def patched_forward(x: torch.Tensor, feat_size):
        """
        Args:
            x: (B, N, C)
            feat_size: [H, W] for spatial dims
        """
        try:
            if isinstance(feat_size, (list, tuple)) and len(feat_size) >= 2:
                H, W = int(feat_size[0]), int(feat_size[1])
            elif isinstance(feat_size, torch.Tensor):
                H, W = int(feat_size[0].item()), int(feat_size[1].item())
            else:
                B, N, C = x.shape
                s = int(math.sqrt(N))
                H, W = s, s

            B, N, C = x.shape
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            scale = getattr(attn_module, "scale", head_dim ** -0.5)

            # Q: (B, heads, Nq, head_dim)
            q = attn_module.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

            linear_flag = bool(getattr(attn_module, "linear", False))

            # infer sr_ratio
            if hasattr(attn_module, "sr_ratio"):
                sr_ratio = int(attn_module.sr_ratio)
            elif hasattr(attn_module, "sr") and attn_module.sr is not None:
                if hasattr(attn_module.sr, "stride"):
                    stride = attn_module.sr.stride
                    sr_ratio = int(stride[0] if isinstance(stride, (tuple, list)) else stride)
                elif hasattr(attn_module.sr, "kernel_size"):
                    ks = attn_module.sr.kernel_size
                    sr_ratio = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
                else:
                    sr_ratio = 1
            else:
                sr_ratio = 1

            if not linear_flag:
                # non-linear SRA
                if sr_ratio > 1 and hasattr(attn_module, "sr") and attn_module.sr is not None:
                    x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                    x_ = attn_module.sr(x_)  # (B,C,H',W')
                    Hk, Wk = x_.shape[-2], x_.shape[-1]
                    x_ = x_.reshape(B, C, -1).permute(0, 2, 1)  # (B, Nk, C)
                    if hasattr(attn_module, "norm") and attn_module.norm is not None:
                        x_ = attn_module.norm(x_)
                else:
                    x_ = x
                    Hk, Wk = H, W
            else:
                # linear SRA
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                if hasattr(attn_module, "pool") and attn_module.pool is not None:
                    x_ = attn_module.pool(x_)
                if hasattr(attn_module, "sr") and attn_module.sr is not None:
                    x_ = attn_module.sr(x_)
                Hk, Wk = x_.shape[-2], x_.shape[-1]
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
                if hasattr(attn_module, "norm") and attn_module.norm is not None:
                    x_ = attn_module.norm(x_)
                if hasattr(attn_module, "act") and attn_module.act is not None:
                    x_ = attn_module.act(x_)

            kv = attn_module.kv(x_)  # (B, Nk, 2C)
            Nk = kv.shape[1]
            kv = kv.reshape(B, Nk, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # (B, heads, Nk, head_dim)

            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            store[name] = attn.detach().float().cpu()

            attn2 = attn_module.attn_drop(attn)
            out = (attn2 @ v).transpose(1, 2).reshape(B, N, C)
            out = attn_module.proj(out)
            out = attn_module.proj_drop(out)
            return out

        except Exception as e:
            print(f"[WARN] patched_forward failed at {name}: {e}. Fallback to original forward.")
            return orig_forward(x, feat_size)

    attn_module.forward = patched_forward


def patch_all_pvt_attn(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Patch all attention modules in timm PVTv2 model.
    Return a dict mapping layer_name -> attn_module.
    """
    store_modules = {}

    if not hasattr(model, "model") or not hasattr(model.model, "stages"):
        raise RuntimeError("Expected wrapper model with .model.stages (your PVTv2_B0 wrapper).")

    for s_idx, stage in enumerate(model.model.stages):
        if not hasattr(stage, "blocks"):
            continue
        for b_idx, block in enumerate(stage.blocks):
            if not hasattr(block, "attn"):
                continue
            attn = block.attn
            if _is_pvt_attention_module(attn):
                layer_name = f"stage{s_idx+1}_block{b_idx+1}"
                store_modules[layer_name] = attn

    return store_modules


# ============================================================
# 2) Attention -> heatmap / key importance
# ============================================================

def tensor_to_uint8_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: (1,3,H,W) normalized (maybe)
    Return RGB uint8 image for visualization.
    """
    img = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def key_importance_from_attn(attn_bhnk: torch.Tensor, k: int = None) -> torch.Tensor:
    """
    attn_bhnk: (B, heads, Nq, Nk)
    k: If provided, only keep top-k important positions per head, set others to 0
    Return key importance per head: (heads, Nk) by averaging over queries.
    If k is provided, only top-k positions are kept, others are set to 0.
    """
    key_imp = attn_bhnk.mean(dim=2)[0]  # (heads, Nk) using batch 0
    
    if k is not None and k > 0:
        # For each head, keep only top-k positions, set others to 0
        heads, Nk = key_imp.shape
        k_actual = min(k, Nk)
        
        # Create a mask for top-k positions per head
        result = torch.zeros_like(key_imp)
        for h in range(heads):
            topk_values, topk_indices = torch.topk(key_imp[h], k=k_actual)
            result[h, topk_indices] = topk_values
        
        return result
    
    return key_imp


def infer_key_grid_size(Nk: int, H_hint: int = None, W_hint: int = None) -> Tuple[int, int]:
    if H_hint is not None and W_hint is not None and H_hint * W_hint == Nk:
        return H_hint, W_hint

    s = int(math.sqrt(Nk))
    if s * s == Nk:
        return s, s

    best = (1, Nk)
    best_gap = Nk
    for h in range(1, int(math.sqrt(Nk)) + 1):
        if Nk % h == 0:
            w = Nk // h
            gap = abs(h - w)
            if gap < best_gap:
                best = (h, w)
                best_gap = gap
    return best


# ============================================================
# 3) Clean visualization helpers
# ============================================================

def normalize_hm_percentile(hm: np.ndarray, low: float = 5.0, high: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    """Percentile clip + minmax normalize for stable contrast."""
    lo = np.percentile(hm, low)
    hi = np.percentile(hm, high)
    hm = np.clip(hm, lo, hi)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + eps)
    return hm


# def overlay_heatmap_masked(
#     img_bgr: np.ndarray,
#     heatmap_2d: np.ndarray,
#     input_size: int,
#     alpha: float = 0.55,
#     colormap: int = cv2.COLORMAP_MAGMA,
#     use_gaussian: bool = True,
#     mask_top_pct: float = 15.0,   # only colorize top X% attention
# ) -> np.ndarray:
#     """
#     Overlay heatmap onto image, but ONLY on high-attention region to avoid "dirty" look.
#     """
#     hm = cv2.resize(heatmap_2d, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
#     hm = normalize_hm_percentile(hm, low=5.0, high=99.0)

#     if use_gaussian:
#         k = max(3, int(input_size / 70))
#         if k % 2 == 0:
#             k += 1
#         hm = cv2.GaussianBlur(hm, (k, k), 0)
#         hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

#     thr = np.percentile(hm, 100.0 - mask_top_pct)
#     mask = (hm >= thr).astype(np.float32)

#     hm_u8 = np.uint8(255 * hm)
#     hm_color = cv2.applyColorMap(hm_u8, colormap).astype(np.float32) / 255.0

#     img = img_bgr.astype(np.float32) / 255.0
#     out = img.copy()

#     overlay = (1 - alpha) * img + alpha * hm_color
#     out = overlay * mask[..., None] + out * (1 - mask[..., None])
#     return (out * 255).astype(np.uint8)

def overlay_heatmap_smooth(
    img_bgr: np.ndarray,
    heatmap_2d: np.ndarray,
    input_size: int,
    alpha: float = 0.5,        # 調整熱圖透明度，0.5 是一半一半
    colormap: int = cv2.COLORMAP_JET, # 使用 JET (傳統紅藍) 或 TURBO
    blur_sigma: int = 15       # 增加模糊半徑讓它更平滑
) -> np.ndarray:
    """
    經典平滑熱圖：背景清晰可見，注意力區域自然過渡
    """
    # 1. 放大熱圖並進行基本的歸一化
    hm = cv2.resize(heatmap_2d, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    
    # 2. 強力平滑化處理 (解決 PVTv2 小特徵圖產生的方格感)
    k_size = int(input_size / 10)
    if k_size % 2 == 0: k_size += 1
    hm = cv2.GaussianBlur(hm, (k_size, k_size), 0)
    
    # 重新歸一化到 0-255
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_u8 = np.uint8(255 * hm)

    # 3. 生成彩色熱圖
    heatmap_color = cv2.applyColorMap(hm_u8, colormap)

    # 4. Alpha 混合 (不使用 Mask，讓原圖全圖保留)
    # 運算公式：結果 = 原圖 * (1 - alpha) + 彩色熱圖 * alpha
    out = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    
    return out


def draw_topk_points_on_image(
    img_bgr: np.ndarray,
    topk_indices: np.ndarray,
    grid_hw: Tuple[int, int],
    input_size: int,
    radius: int = None,
) -> np.ndarray:
    """Draw top-k as points + rank labels (cleaner than boxes)."""
    Hk, Wk = grid_hw
    stride_y = input_size / Hk
    stride_x = input_size / Wk

    out = img_bgr.copy()
    if radius is None:
        radius = max(3, int(input_size / 80))

    for rank, idx in enumerate(topk_indices, start=1):
        r = int(idx // Wk)
        c = int(idx % Wk)
        cx = int(round((c + 0.5) * stride_x))
        cy = int(round((r + 0.5) * stride_y))

        # black rim + white dot
        cv2.circle(out, (cx, cy), radius + 2, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), radius, (255, 255, 255), -1)

        # rank text (white with black outline)
        tx, ty = cx + radius + 4, cy - radius - 2
        cv2.putText(out, str(rank), (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(rank), (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_heads_heatmap_montage(
    key_imp: torch.Tensor,
    Hk: int,
    Wk: int,
    layer_name: str,
    fig_path: str,
    cols: int = 4,
    cmap: str = "magma",
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    interpolation: str = "nearest",  # <-- PVT 小圖強烈建議 nearest
) -> None:
    """
    Paper-like montage:
    - All heads share SAME vmin/vmax (robust percentiles)
    - Only ONE shared colorbar
    - White background, compact titles
    """
    key_np = key_imp.detach().cpu().numpy()  # (heads, Nk)
    heads = key_np.shape[0]
    cols = min(cols, heads)
    rows = int(math.ceil(heads / cols))

    all_vals = key_np.reshape(-1)
    vmin = float(np.percentile(all_vals, vmin_percentile))
    vmax = float(np.percentile(all_vals, vmax_percentile))
    if vmax <= vmin + 1e-12:
        vmin = float(all_vals.min())
        vmax = float(all_vals.max() + 1e-8)

    # 更緊湊、比較像 paper
    fig = plt.figure(figsize=(3.2 * cols, 3.0 * rows), facecolor="white")
    gs = fig.add_gridspec(
        rows, cols + 1,
        width_ratios=[1] * cols + [0.03],  # <-- 色條變細
        wspace=0.08, hspace=0.18           # <-- 留白變少
    )

    last_im = None
    for h in range(heads):
        r, c = divmod(h, cols)
        ax = fig.add_subplot(gs[r, c])
        hm = key_np[h].reshape(Hk, Wk)

        last_im = ax.imshow(
            hm, cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation=interpolation
        )
        ax.set_title(f"H{h+1}", fontsize=10, pad=2)
        ax.axis("off")

        # 可選：畫出格線，讓 7x7/14x14 更「乾淨、清楚」
        # ax.set_xticks(np.arange(-.5, Wk, 1), minor=True)
        # ax.set_yticks(np.arange(-.5, Hk, 1), minor=True)
        # ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.35)
        # ax.tick_params(which="minor", bottom=False, left=False)

    # 空格補齊
    for j in range(heads, rows * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    # shared colorbar
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)

    fig.suptitle(layer_name, fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(fig_path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_heads_overlay_montage(
    overlays_bgr: List[np.ndarray],
    layer_name: str,
    fig_path: str,
    cols: int = 4,
) -> None:
    heads = len(overlays_bgr)
    cols = min(cols, heads)
    rows = int(math.ceil(heads / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.2 * cols, 3.0 * rows),
        facecolor="white",
        constrained_layout=True  # <-- 比 tight_layout 更穩
    )
    axes = np.array(axes).reshape(-1)

    for h in range(heads):
        rgb = cv2.cvtColor(overlays_bgr[h], cv2.COLOR_BGR2RGB)
        axes[h].imshow(rgb)
        axes[h].set_title(f"H{h+1}", fontsize=10, pad=2)
        axes[h].axis("off")

    for j in range(heads, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{layer_name} (overlay)", fontsize=13, fontweight="bold")
    fig.savefig(fig_path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_heads_topk_montage(
    topk_imgs_bgr: List[np.ndarray],
    layer_name: str,
    fig_path: str,
    cols: int = 4,
) -> None:
    heads = len(topk_imgs_bgr)
    cols = min(cols, heads)
    rows = int(math.ceil(heads / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.2 * cols, 3.0 * rows),
        facecolor="white",
        constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    for h in range(heads):
        rgb = cv2.cvtColor(topk_imgs_bgr[h], cv2.COLOR_BGR2RGB)
        axes[h].imshow(rgb)
        axes[h].set_title(f"H{h+1}", fontsize=10, pad=2)
        axes[h].axis("off")

    for j in range(heads, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{layer_name} (top-k)", fontsize=13, fontweight="bold")
    fig.savefig(fig_path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 4) Main analyze + visualize
# ============================================================

def analyze_and_visualize(
    ckpt_path: str,
    save_dir: str,
    k: int = 5,
    device: str = "cpu",
    input_size: int = 224,
    cmap_mpl: str = "magma",
    cmap_cv: int = cv2.COLORMAP_MAGMA,
    mask_top_pct: float = 15.0,
    alpha: float = 0.55,
    dump_single: bool = False,
    vmax_percentile: float = 99.0,
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load checkpoint ----
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_weights"] if isinstance(checkpoint, dict) and "model_weights" in checkpoint else checkpoint

    if "model.head.weight" in state_dict:
        out_channels = state_dict["model.head.weight"].shape[0]
        print(f"[INFO] out_channels inferred from checkpoint: {out_channels}")
    else:
        out_channels = 10
        print(f"[WARN] cannot infer out_channels from checkpoint, fallback: {out_channels}")

    # ---- Build model ----
    model = PVTv2_B0(out_channels=out_channels, input_size=(input_size, input_size))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # ---- Prepare dataloader ----
    print(f"[INFO] dataset: {config.config['dataset']}")
    _, test_dataloader = get_dataloader(
        dataset=config.config["dataset"],
        root=config.config["root"] + "/data/",
        batch_size=1,
        input_size=config.config["input_shape"],
        use_pretrained_vit=True,
    )

    img, label = next(iter(test_dataloader))
    img = img.to(device)
    print(f"[INFO] test image: {tuple(img.shape)}, label: {label}")

    # ---- Patch all attention modules ----
    attn_modules = patch_all_pvt_attn(model)
    print(f"[INFO] found attention layers to patch: {len(attn_modules)}")

    attn_store: Dict[str, torch.Tensor] = {}
    for layer_name, attn in attn_modules.items():
        monkey_patch_pvt_attn_forward(attn, attn_store, layer_name)

    # ---- Forward (patched forward will populate attn_store) ----
    with torch.no_grad():
        _ = model(img)

    # ---- Prepare base image ----
    rgb = tensor_to_uint8_rgb(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_dir, "original.png"), bgr)
    print(f"[INFO] saved original: {os.path.join(save_dir, 'original.png')}")

    # ---- Per-layer visualizations ----
    layer_keys = sorted(attn_store.keys())
    print("[INFO] layers:", layer_keys)

    for layer_name in layer_keys:
        attn = attn_store[layer_name]  # (B, heads, Nq, Nk)
        if attn.dim() != 4:
            print(f"[WARN] unexpected attn dim at {layer_name}: {attn.shape}")
            continue

        heads = attn.shape[1]
        Nk = attn.shape[-1]
        key_imp = key_importance_from_attn(attn, k=k)  # (heads, Nk) - only top-k positions per head

        Hk, Wk = infer_key_grid_size(Nk)

        layer_dir = os.path.join(save_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        # 1) Heatmap montage (clean)
        heatmap_path = os.path.join(layer_dir, f"{layer_name}_heads_heatmap.png")
        save_heads_heatmap_montage(
            key_imp=key_imp,
            Hk=Hk,
            Wk=Wk,
            layer_name=layer_name,
            fig_path=heatmap_path,
            cols=4,
            cmap="jet",  # Use JET colormap to match overlay visualization
            vmax_percentile=vmax_percentile,
        )
        print(f"[INFO] saved heatmap montage: {heatmap_path}")

        overlays = []
        topk_imgs = []

        # Precompute topk per head
        for h in range(heads):
            hm = key_imp[h].numpy().reshape(Hk, Wk)

            flat = key_imp[h]
            kk = min(k, flat.numel())
            topk_idx = torch.topk(flat, k=kk).indices.cpu().numpy()

            # overlay (masked)
            over = overlay_heatmap_smooth(
                img_bgr=bgr,
                heatmap_2d=hm,
                input_size=input_size,
                alpha=0.4,           # 降低一點透明度，原圖細節會更清楚
                colormap=cv2.COLORMAP_JET,    # 你可以從參數傳入
                blur_sigma=input_size // 8  # 根據輸入大小自動調整模糊程度
            )
            overlays.append(over)

            # topk points
            pts = draw_topk_points_on_image(
                img_bgr=bgr,
                topk_indices=topk_idx,
                grid_hw=(Hk, Wk),
                input_size=input_size,
            )
            topk_imgs.append(pts)

            if dump_single:
                cv2.imwrite(os.path.join(layer_dir, f"head{h:02d}_overlay.png"), over)
                cv2.imwrite(os.path.join(layer_dir, f"head{h:02d}_topk_points.png"), pts)

        # 2) Overlay montage (clean)
        overlay_montage_path = os.path.join(layer_dir, f"{layer_name}_overlay_montage.png")
        save_heads_overlay_montage(
            overlays_bgr=overlays,
            layer_name=layer_name,
            fig_path=overlay_montage_path,
            cols=4,
        )
        print(f"[INFO] saved overlay montage: {overlay_montage_path}")

        # 3) Top-k montage (clean)
        topk_montage_path = os.path.join(layer_dir, f"{layer_name}_topk_montage.png")
        save_heads_topk_montage(
            topk_imgs_bgr=topk_imgs,
            layer_name=layer_name,
            fig_path=topk_montage_path,
            cols=4,
        )
        print(f"[INFO] saved topk montage: {topk_montage_path}")

    # ---- Stage-wise average summary ----
    stage_groups: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for layer_name, attn in attn_store.items():
        if layer_name.startswith("stage"):
            s = int(layer_name.split("_")[0].replace("stage", ""))
            stage_groups[s].append(attn)

    for s, attns in stage_groups.items():
        if not attns:
            continue

        stack = torch.stack(attns, dim=0)     # (L, B, heads, Nq, Nk)
        mean_attn = stack.mean(dim=0)         # (B, heads, Nq, Nk)

        key_imp_s = key_importance_from_attn(mean_attn, k=k)   # (heads, Nk) - only top-k positions per head
        key_imp_avg = key_imp_s.mean(dim=0)               # (Nk,)

        Nk = key_imp_avg.numel()
        Hk, Wk = infer_key_grid_size(Nk)
        hm = key_imp_avg.numpy().reshape(Hk, Wk)

        out = overlay_heatmap_smooth(
            img_bgr=bgr,
            heatmap_2d=hm,
            input_size=input_size,
            alpha=0.4,           # 降低一點透明度，原圖細節會更清楚
            colormap=cv2.COLORMAP_JET,    # 你可以從參數傳入
            blur_sigma=input_size // 8  # 根據輸入大小自動調整模糊程度
        )
        out_path = os.path.join(save_dir, f"stage{s}_avg_overlay.png")
        cv2.imwrite(out_path, out)
        print(f"[INFO] saved stage summary overlay: {out_path}")

    print(f"\n[DONE] all outputs saved to: {save_dir}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/train/exp125/PVTv2_B0_best.pth", help="path to checkpoint .pth")
    ap.add_argument("--save_dir", type=str, default="pvt_v2_attn_vis_clean_7x7", help="output directory")
    ap.add_argument("--k", type=int, default=5, help="top-k positions per head")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--input_size", type=int, default=224, help="input resolution for visualization")

    # visualization knobs
    ap.add_argument("--mask_top_pct", type=float, default=15.0, help="only overlay top X%% heatmap region")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay alpha")
    ap.add_argument("--dump_single", action="store_true", help="also save single head images (overlay/topk)")
    ap.add_argument("--vmax_percentile", type=float, default=99.0, help="percentile for shared vmax in heatmap montage")

    # choose colormap
    ap.add_argument("--cmap", type=str, default="magma", help="matplotlib colormap name (e.g., magma/viridis)")
    return ap


def cmap_name_to_cv2(cmap_name: str) -> int:
    cmap_name = cmap_name.lower()
    if cmap_name == "viridis":
        return cv2.COLORMAP_VIRIDIS
    if cmap_name == "plasma":
        return cv2.COLORMAP_PLASMA
    if cmap_name == "inferno":
        return cv2.COLORMAP_INFERNO
    if cmap_name == "magma":
        return cv2.COLORMAP_MAGMA
    # fallback
    return cv2.COLORMAP_MAGMA


if __name__ == "__main__":
    args = build_argparser().parse_args()
    analyze_and_visualize(
        ckpt_path=args.ckpt,
        save_dir=args.save_dir,
        k=args.k,
        device=args.device,
        input_size=args.input_size,
        cmap_mpl=args.cmap,
        cmap_cv=cmap_name_to_cv2(args.cmap),
        mask_top_pct=args.mask_top_pct,
        alpha=args.alpha,
        dump_single=args.dump_single,
        vmax_percentile=args.vmax_percentile,
    )
