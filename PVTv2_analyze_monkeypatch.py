#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PVTv2 (timm) Attention Visualization via Monkey Patch

- Monkey-patch Attention.forward (method-level), NOT module replacement.
- Store attention weights (softmax after scaling, before dropout) for each stage/block.
- Visualize per-layer per-head:
  1) Top-k key positions on image (grid box approx)
  2) Heatmap overlay (upsample key-importance to input size)

Notes:
- PVTv2 in timm typically has no cls token.
- Attention shape in PVTv2: (B, heads, Nq, Nk)
- We convert to "key importance" by averaging over queries:
    key_imp[h, k] = mean_q attn[h, q, k]
  then reshape (Hk, Wk) where Hk*Wk = Nk

Author: (for your project use)
"""

import os
import math
import argparse
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---- Your project imports (adjust if needed) ----
from models.PVTv2_B0 import PVTv2_B0
from dataloader import get_dataloader
import config


# ============================================================
# 1) Monkey patch: patch each attention module's forward method
# ============================================================

def _is_pvt_attention_module(m: nn.Module) -> bool:
    """
    Heuristic to detect timm PVTv2 Attention module:
    has q, kv, num_heads, attn_drop, proj, proj_drop, and forward signature (x, feat_size).
    """
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
        # already patched
        return

    orig_forward = attn_module.forward
    attn_module._orig_forward = orig_forward  # keep reference

    def patched_forward(x: torch.Tensor, feat_size):
        """
        Re-implement the timm PVTv2 Attention.forward logic to capture attn.
        This is still "monkey patch" because we replace the method on the SAME module,
        not swapping modules.
        
        Args:
            x: input tensor (B, N, C)
            feat_size: List[int] containing [H, W] for spatial dimensions
            [(56, 56), (28, 28), (14, 14), (7, 7)]
        """
        try:            # Extract H, W from feat_size (which is List[int] in timm)
            if isinstance(feat_size, (list, tuple)) and len(feat_size) >= 2:
                H, W = feat_size[0], feat_size[1]
            elif isinstance(feat_size, torch.Tensor):
                H, W = feat_size[0].item(), feat_size[1].item()
            else:
                # Fallback: infer from input shape
                B, N, C = x.shape
                s = int(math.sqrt(N))
                H, W = s, s
            B, N, C = x.shape
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            # timm uses either attn_module.scale or head_dim ** -0.5
            scale = getattr(attn_module, "scale", head_dim ** -0.5)
            # Q: (B, heads, Nq, head_dim)
            q = attn_module.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            # Build K,V with SRA / Linear SRA branches
            # (The following mirrors the common PVTv2 implementation in timm)
            linear_flag = bool(getattr(attn_module, "linear", False))
            
            # timm 的 PVTv2 Attention 不會將 sr_ratio 存為屬性，需要從 self.sr 推斷
            # 如果直接有 sr_ratio 屬性就用它，否則從 sr 的 stride 推斷
            if hasattr(attn_module, "sr_ratio"):
                sr_ratio = int(attn_module.sr_ratio)
            elif hasattr(attn_module, "sr") and attn_module.sr is not None:
                # 從 Conv2d 的 stride 推斷 sr_ratio
                if hasattr(attn_module.sr, "stride"):
                    stride = attn_module.sr.stride
                    if isinstance(stride, (tuple, list)):
                        sr_ratio = int(stride[0])  # 通常 stride 是 (sr_ratio, sr_ratio)
                    else:
                        sr_ratio = int(stride)
                elif hasattr(attn_module.sr, "kernel_size"):
                    # 如果沒有 stride，從 kernel_size 推斷（通常 kernel_size == stride）
                    kernel_size = attn_module.sr.kernel_size
                    if isinstance(kernel_size, (tuple, list)):
                        sr_ratio = int(kernel_size[0])
                    else:
                        sr_ratio = int(kernel_size)
                else:
                    sr_ratio = 1
            else:
                sr_ratio = 1

            if not linear_flag:
                # non-linear SRA
                if sr_ratio > 1 and hasattr(attn_module, "sr") and attn_module.sr is not None:
                    x_ = x.permute(0, 2, 1).reshape(B, C, H, W)     # (B,C,H,W)
                    x_ = attn_module.sr(x_)                         # (B,C,H',W') with stride=sr_ratio
                    Hk, Wk = x_.shape[-2], x_.shape[-1]
                    x_ = x_.reshape(B, C, -1).permute(0, 2, 1)      # (B, Nk, C)
                    if hasattr(attn_module, "norm") and attn_module.norm is not None:
                        x_ = attn_module.norm(x_)
                else:
                    x_ = x
                    Hk, Wk = H, W

            else:
                # linear SRA (usually: pool to 7x7 then 1x1 conv + norm + act)
                # NOTE: in timm, linear=True is typically used in *_li variants
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)         # (B,C,H,W)
                if hasattr(attn_module, "pool") and attn_module.pool is not None:
                    x_ = attn_module.pool(x_)                      # e.g., AdaptiveAvgPool2d(7)
                if hasattr(attn_module, "sr") and attn_module.sr is not None:
                    x_ = attn_module.sr(x_)                        # 1x1 conv
                Hk, Wk = x_.shape[-2], x_.shape[-1]
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)          # (B, Nk, C)
                if hasattr(attn_module, "norm") and attn_module.norm is not None:
                    x_ = attn_module.norm(x_)
                if hasattr(attn_module, "act") and attn_module.act is not None:
                    x_ = attn_module.act(x_)

            # KV: (B, heads, Nk, head_dim) each
            kv = attn_module.kv(x_)  # (B, Nk, 2C)
            Nk = kv.shape[1]
            kv = kv.reshape(B, Nk, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]      # (B, heads, Nk, head_dim)

            # attn: (B, heads, Nq, Nk)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            # store BEFORE dropout
            # store float32 on CPU to avoid GPU memory blow
            store[name] = attn.detach().float().cpu()

            # continue original forward path
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
    Return a dict mapping layer_name -> attn_module (for debugging).
    """
    store_modules = {}

    # For timm PVTv2, structure typically: model.model.stages[s].blocks[b].attn
    # We'll follow that to have stable names.
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
# 2) Attention -> heatmap / top-k mapping
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


def key_importance_from_attn(attn_bhnk: torch.Tensor) -> torch.Tensor:
    """
    attn_bhnk: (B, heads, Nq, Nk)
    Return key importance per head: (heads, Nk) by averaging over queries.
    """
    # average over Nq
    return attn_bhnk.mean(dim=2)[0]  # (heads, Nk) using batch 0


def infer_key_grid_size(Nk: int, H_hint: int = None, W_hint: int = None) -> Tuple[int, int]:
    """
    Infer (Hk, Wk) from Nk.
    Prefer square if possible.
    """
    if H_hint is not None and W_hint is not None and H_hint * W_hint == Nk:
        return H_hint, W_hint

    s = int(math.sqrt(Nk))
    if s * s == Nk:
        return s, s

    # fallback: best factor pair close to square
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


def draw_topk_boxes_on_image(
    img_bgr: np.ndarray,
    topk_indices: np.ndarray,
    grid_hw: Tuple[int, int],
    input_size: int,
    key_importance: np.ndarray = None,
    color_scheme: str = "gradient",
) -> np.ndarray:
    """
    Draw top-k boxes based on grid indices on the input image with uniform styling.
    
    Args:
        img_bgr: Input image in BGR format
        topk_indices: Indices of top-k positions
        grid_hw: Grid dimensions (Hk, Wk)
        input_size: Input image size
        key_importance: (Deprecated, not used) Optional importance values
        color_scheme: (Deprecated, not used) Color scheme option
    """
    Hk, Wk = grid_hw
    stride_y = input_size / Hk
    stride_x = input_size / Wk

    # Convert to RGB for better color handling
    out = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Use uniform color for all boxes (green in RGB, normalized to 0-1 range)
    color = (0.0, 1.0, 0.0)  # Green color in RGB (0-1 range)
    thickness = max(2, int(input_size / 112))  # Scale thickness with image size
    
    # Draw boxes with uniform styling
    for idx in topk_indices:
        r = int(idx // Wk)
        c = int(idx % Wk)
        x1 = int(round(c * stride_x))
        y1 = int(round(r * stride_y))
        x2 = int(round((c + 1) * stride_x))
        y2 = int(round((r + 1) * stride_y))
        
        # Draw border only (no fill, no numbers)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    
    # Convert back to BGR uint8
    out = (out * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def overlay_heatmap(
    img_bgr: np.ndarray,
    heatmap_2d: np.ndarray,
    input_size: int,
    alpha_img: float = 0.5,
    alpha_hm: float = 0.5,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    use_gaussian: bool = True,
) -> np.ndarray:
    """
    Overlay heatmap (Hk,Wk) onto image with improved visualization.
    
    Args:
        img_bgr: Input image in BGR format
        heatmap_2d: 2D heatmap array
        input_size: Target size for resizing
        alpha_img: Transparency of original image
        alpha_hm: Transparency of heatmap
        colormap: OpenCV colormap (VIRIDIS, PLASMA, INFERNO, JET, etc.)
        use_gaussian: Apply Gaussian blur for smoother visualization
    """
    # Resize heatmap
    hm = cv2.resize(heatmap_2d, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    
    # Apply Gaussian blur for smoother visualization
    if use_gaussian:
        kernel_size = max(3, int(input_size / 50))
        if kernel_size % 2 == 0:
            kernel_size += 1
        hm = cv2.GaussianBlur(hm, (kernel_size, kernel_size), 0)
        # Renormalize after blur
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    
    # Apply colormap
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    
    # Convert to float for blending
    img_float = img_bgr.astype(np.float32) / 255.0
    hm_float = hm_color.astype(np.float32) / 255.0
    
    # Blend with improved contrast
    blended = img_float * alpha_img + hm_float * alpha_hm
    
    # Convert back to uint8
    return (blended * 255).astype(np.uint8)


# ============================================================
# 3) Main analyze + visualize
# ============================================================

def analyze_and_visualize(
    ckpt_path: str,
    save_dir: str,
    k: int = 5,
    device: str = "cpu",
    input_size: int = 224,
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load checkpoint, infer out_channels ----
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

    # ---- Visualize ----
    rgb = tensor_to_uint8_rgb(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # save original
    cv2.imwrite(os.path.join(save_dir, "original.png"), bgr)
    print(f"[INFO] saved original: {os.path.join(save_dir, 'original.png')}")

    # Per-layer montage (all heads)
    layer_keys = sorted(attn_store.keys())
    print(layer_keys)
    for layer_name in layer_keys:
        attn = attn_store[layer_name]  # (B, heads, Nq, Nk) on CPU float
        if attn.dim() != 4:
            print(f"[WARN] unexpected attn dim at {layer_name}: {attn.shape}")
            continue

        heads = attn.shape[1]
        Nk = attn.shape[-1]
        key_imp = key_importance_from_attn(attn)  # (heads, Nk)

        # infer key grid (Nk -> Hk,Wk)
        Hk, Wk = infer_key_grid_size(Nk)

        layer_dir = os.path.join(save_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        # create a matplotlib montage: each head heatmap (Hk,Wk)
        cols = min(4, heads)
        rows = int(math.ceil(heads / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for h in range(heads):
            hm = key_imp[h].numpy().reshape(Hk, Wk)
            # Use modern colormaps for better visualization
            colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            cmap = colormaps[h % len(colormaps)]
            im = axes[h].imshow(hm, cmap=cmap, interpolation='bilinear')
            axes[h].set_title(f"{layer_name} | head {h}", fontsize=10, fontweight='bold')
            axes[h].axis("off")
            # Add colorbar for better interpretation
            plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

        for j in range(heads, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        fig_path = os.path.join(layer_dir, f"{layer_name}_heads_heatmap.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved head montage: {fig_path}")

        # For each head: overlay + topk boxes
        overlay_images = []  # Store all overlay images for montage
        boxed_images = []    # Store all boxed images for montage
        
        for h in range(heads):
            hm = key_imp[h].numpy().reshape(Hk, Wk)

            # top-k on key positions
            flat = key_imp[h]
            kk = min(k, flat.numel())
            topk_idx = torch.topk(flat, k=kk).indices.cpu().numpy()

            # Improved visualizations
            boxed = draw_topk_boxes_on_image(
                bgr, topk_idx, (Hk, Wk), 
                input_size=input_size,
                key_importance=flat.numpy(),
                color_scheme="gradient"
            )
            
            # Use different colormaps for variety (cycle through)
            colormaps = [cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO, cv2.COLORMAP_MAGMA]
            colormap = colormaps[h % len(colormaps)]
            over = overlay_heatmap(
                bgr, hm, 
                input_size=input_size,
                colormap=colormap,
                use_gaussian=True
            )

            cv2.imwrite(os.path.join(layer_dir, f"head{h:02d}_topk_boxes.png"), boxed)
            cv2.imwrite(os.path.join(layer_dir, f"head{h:02d}_overlay.png"), over)
            
            # Store for montage
            overlay_images.append((over, hm, colormap))
            boxed_images.append(boxed)
        
        # Create montage: all heads overlay with colorbar
        cols = min(4, heads)
        rows = int(math.ceil(heads / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).reshape(-1)
        
        # Map colormap enum to matplotlib colormap name
        cmap_map = {
            cv2.COLORMAP_VIRIDIS: 'viridis',
            cv2.COLORMAP_PLASMA: 'plasma',
            cv2.COLORMAP_INFERNO: 'inferno',
            cv2.COLORMAP_MAGMA: 'magma'
        }
        
        for h in range(heads):
            over, hm, colormap = overlay_images[h]
            # Convert BGR to RGB for matplotlib
            over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
            axes[h].imshow(over_rgb)
            axes[h].set_title(f"Head {h+1}", fontsize=12, fontweight='bold')
            axes[h].axis("off")
            
            # Add colorbar using the heatmap data
            mpl_cmap = cmap_map.get(colormap, 'viridis')
            sm = ScalarMappable(cmap=mpl_cmap, norm=Normalize(vmin=hm.min(), vmax=hm.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[h], fraction=0.046, pad=0.04)
        
        for j in range(heads, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        overlay_montage_path = os.path.join(layer_dir, f"{layer_name}_overlay_montage.png")
        plt.savefig(overlay_montage_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved overlay montage: {overlay_montage_path}")
        
        # Create montage: all heads boxes
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).reshape(-1)
        
        for h in range(heads):
            boxed = boxed_images[h]
            # Convert BGR to RGB for matplotlib
            boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            axes[h].imshow(boxed_rgb)
            axes[h].set_title(f"Head {h+1}", fontsize=12, fontweight='bold')
            axes[h].axis("off")
        
        for j in range(heads, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        boxes_montage_path = os.path.join(layer_dir, f"{layer_name}_boxes_montage.png")
        plt.savefig(boxes_montage_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved boxes montage: {boxes_montage_path}")

    # Summary: stage-wise average over (blocks, heads)
    stage_groups: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for layer_name, attn in attn_store.items():
        # layer_name = stage{S}_block{B}
        if layer_name.startswith("stage"):
            s = int(layer_name.split("_")[0].replace("stage", ""))
            stage_groups[s].append(attn)

    for s, attns in stage_groups.items():
        if not attns:
            continue
        # average all layers within stage: average over blocks and heads and queries => key map
        # attn: (B, heads, Nq, Nk)
        # -> mean over blocks: stack then mean
        stack = torch.stack(attns, dim=0)           # (L, B, heads, Nq, Nk)
        mean_attn = stack.mean(dim=0)               # (B, heads, Nq, Nk)
        key_imp = key_importance_from_attn(mean_attn)  # (heads, Nk)
        key_imp_avg = key_imp.mean(dim=0)           # (Nk,)
        Nk = key_imp_avg.numel()
        Hk, Wk = infer_key_grid_size(Nk)
        hm = key_imp_avg.numpy().reshape(Hk, Wk)

        # Use VIRIDIS for stage summaries (clean and professional)
        out = overlay_heatmap(
            bgr, hm, 
            input_size=input_size,
            colormap=cv2.COLORMAP_VIRIDIS,
            use_gaussian=True
        )
        out_path = os.path.join(save_dir, f"stage{s}_avg_overlay.png")
        cv2.imwrite(out_path, out)
        print(f"[INFO] saved stage summary overlay: {out_path}")

    print(f"\n[DONE] all outputs saved to: {save_dir}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/train/exp125/PVTv2_B0_best.pth", help="path to checkpoint .pth")
    ap.add_argument("--save_dir", type=str, default="pvt_v2_attn_monkeypatch", help="output directory")
    ap.add_argument("--k", type=int, default=5, help="top-k positions per head")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--input_size", type=int, default=224, help="input resolution for visualization")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    analyze_and_visualize(
        ckpt_path=args.ckpt,
        save_dir=args.save_dir,
        k=args.k,
        device=args.device,
        input_size=args.input_size,
    )
