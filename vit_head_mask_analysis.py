#!/usr/bin/env python3
"""
使用 VIT_with_Partial_LRP 的 head masking 功能：
1) 先以 Partial LRP 計算每層 head 重要度
2) 依每層前 K 個重要 head 建立 mask，對模型做一次 masked forward
3) 再以 masked 條件重算逐層 relevance maps 與總圖，並與原始版本對照繪圖
不修改既有 example_head_importance_analysis.py。
"""
import os
import glob
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import config
from dataloader import get_dataloader
import models
from models.VIT_with_Partial_LRP import VIT_with_Partial_LRP


def _infer_patch_size(vit_model):
    try:
        if hasattr(vit_model, 'patch_embed') and hasattr(vit_model.patch_embed, 'patch_size'):
            ps = vit_model.patch_embed.patch_size
            if isinstance(ps, (tuple, list)) and len(ps) == 2:
                return int(ps[0]), int(ps[1])
            if isinstance(ps, int):
                return int(ps), int(ps)
        if hasattr(vit_model, 'patch_embed') and hasattr(vit_model.patch_embed, 'proj'):
            proj = vit_model.patch_embed.proj
            if hasattr(proj, 'kernel_size'):
                ks = proj.kernel_size
                if isinstance(ks, (tuple, list)) and len(ks) == 2:
                    return int(ks[0]), int(ks[1])
                if isinstance(ks, int):
                    return int(ks), int(ks)
            if hasattr(proj, 'stride'):
                st = proj.stride
                if isinstance(st, (tuple, list)) and len(st) == 2:
                    return int(st[0]), int(st[1])
                if isinstance(st, int):
                    return int(st), int(st)
    except Exception:
        pass
    return None


def _draw_patch_grid(ax, height, width, patch_h, patch_w, color='white', lw=0.6, alpha=0.8):
    if patch_w and patch_w > 0:
        for x in range(0, width + 1, patch_w):
            ax.axvline(x=x - 0.5, color=color, linewidth=lw, alpha=alpha)
    if patch_h and patch_h > 0:
        for y in range(0, height + 1, patch_h):
            ax.axhline(y=y - 0.5, color=color, linewidth=lw, alpha=alpha)


def _latest_best_checkpoint():
    # 嘗試尋找 runs/train/exp*/VIT_best.pth
    cand = sorted(glob.glob("runs/train/exp62/VIT_best.pth"))
    if len(cand) > 0:
        return cand[-1]
    # 後備：固定 exp
    fixed = "runs/train/exp62/VIT_best.pth"
    return fixed if os.path.exists(fixed) else None


def select_topk_heads_per_layer(head_importance: List[Dict], topk: int) -> Dict[int, List[int]]:
    layer_to_heads: Dict[int, List[int]] = {}
    for info in head_importance:
        layer_idx = int(info['layer'])
        head_scores = info['head_scores']  # [B,H] (tensor on CPU)
        mean_scores = head_scores.mean(dim=0)  # [H]
        k = min(int(topk), mean_scores.numel())
        if k <= 0:
            continue
        top_indices = mean_scores.topk(k).indices.tolist()
        layer_to_heads[layer_idx] = top_indices
    return layer_to_heads


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # dataloader 與模型
    print("建立資料載入器與模型...")
    train_loader, test_loader = get_dataloader(
        dataset=config['dataset'],
        root=config['root'] + '/data/',
        batch_size=config['batch_size'],
        input_size=config['input_shape']
    )
    model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
    model = model.to(device).eval()

    # 載入最佳權重
    ckpt_path = _latest_best_checkpoint()
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"載入權重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        weights = ckpt.get('model_weights', None)
        if weights is not None:
            missing, unexpected = model.load_state_dict(weights, strict=False)
            print(f"載入到模型 - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("未找到最佳權重，使用當前模型參數。")

    # 取得 backbone 並建立 analyzer（此時已套用權重）
    vit_backbone = model.backbone if hasattr(model, 'backbone') else model
    analyzer = VIT_with_Partial_LRP(vit_backbone, topk_heads=None, head_weighting='normalize', eps=0.01).to(device).eval()

    # 蒐集少量樣本
    def collect_n(loader, n=4):
        xs, ys = [], []
        cnt = 0
        for xb, yb in loader:
            xs.append(xb); ys.append(yb); cnt += xb.size(0)
            if cnt >= n:
                break
        if len(xs) == 0:
            return None, None
        X = torch.cat(xs, dim=0)[:n].to(device)
        if isinstance(ys[0], torch.Tensor):
            Y = torch.cat(ys, dim=0)[:n].to(device)
        else:
            Y = torch.tensor(ys)[:n].to(device)
        return X, Y

    N = 4
    X, y = collect_n(test_loader, N)
    if X is None:
        X, y = collect_n(train_loader, N)
    if X is None:
        raise RuntimeError("無法取得樣本")

    # baseline logits
    with torch.no_grad():
        logits_base = analyzer.model(X)
        pred_base = logits_base.argmax(dim=1)
        conf_base = torch.softmax(logits_base, dim=1).max(dim=1)[0]
    pred_base = pred_base.cpu()
    conf_base = conf_base.cpu()
    print(f"Baseline preds: {pred_base.tolist()}, conf: {[float(v) for v in conf_base]}")

    # 取得 head 重要度（不設定 topk，先拿分數）
    print("計算 head 重要度（Partial LRP）...")
    result = analyzer.explain(X, target_class=None, return_map='patch', return_head_importance=True)
    head_importance = result['head_importance']
    # 依每層取前 K 個 head
    TOPK = 5
    layer_to_heads = select_topk_heads_per_layer(head_importance, topk=TOPK)
    print("每層選取的 heads（Top-K）:", {k: v for k, v in sorted(layer_to_heads.items())})

    # 設定 head mask，進行 masked forward
    analyzer.set_head_mask(layer_to_heads, keep_only=True)
    with torch.no_grad():
        logits_mask = analyzer.model(X)
        pred_keep = logits_mask.argmax(dim=1)
        conf_keep = torch.softmax(logits_mask, dim=1).max(dim=1)[0]
    pred_keep = pred_keep.cpu()
    conf_keep = conf_keep.cpu()
    print(f"保留 Top-{TOPK} 頭部後的 preds: {pred_keep.tolist()}, conf: {[float(v) for v in conf_keep]}")

    # 產生 masked 條件下的逐層結果
    print("計算 masked 條件之逐層結果...")
    # 第一層：relevance（input patch relevance）
    relevance_keep = analyzer.explain(X, target_class=None, return_map='image', upsample_to_input=True)
    # 後續層：activation
    activations_keep = analyzer.get_per_layer_activations(X, return_map='image', upsample_to_input=True)
    
    # 合併：第一層用 relevance，後續層用 activation
    # relevance_mask 是 [B, H, W]，需要包裝成字典格式
    layer_results_keep = {'input_relevance': relevance_keep}
    layer_results_keep.update(activations_keep)

    # 設定 head mask：移除前 K 個重要 head
    analyzer.clear_head_mask()
    analyzer.set_head_mask(layer_to_heads, keep_only=False)
    with torch.no_grad():
        logits_drop = analyzer.model(X)
        pred_drop = logits_drop.argmax(dim=1)
        conf_drop = torch.softmax(logits_drop, dim=1).max(dim=1)[0]
    pred_drop = pred_drop.cpu()
    conf_drop = conf_drop.cpu()
    print(f"移除 Top-{TOPK} 頭部後的 preds: {pred_drop.tolist()}, conf: {[float(v) for v in conf_drop]}")

    relevance_drop = analyzer.explain(X, target_class=None, return_map='image', upsample_to_input=True)
    activations_drop = analyzer.get_per_layer_activations(X, return_map='image', upsample_to_input=True)
    layer_results_drop = {'input_relevance': relevance_drop}
    layer_results_drop.update(activations_drop)

    # 也產生 baseline 條件下的逐層結果（為公平，先清除 mask 再算）
    analyzer.clear_head_mask()
    relevance_base = analyzer.explain(X, target_class=None, return_map='image', upsample_to_input=True)
    activations_base = analyzer.get_per_layer_activations(X, return_map='image', upsample_to_input=True)
    
    # 合併：第一層用 relevance，後續層用 activation
    # relevance_base 是 [B, H, W]，需要包裝成字典格式
    layer_results_base = {'input_relevance': relevance_base}
    layer_results_base.update(activations_base)

    # 視覺化輸出
    out_dir = "./head_mask_results"
    os.makedirs(out_dir, exist_ok=True)
    patch_ps = _infer_patch_size(analyzer.model)
    patch_h, patch_w = (patch_ps if patch_ps is not None else (None, None))

    X_vis = X.detach().cpu()
    if X_vis.shape[1] == 3:
        X_vis = X_vis.permute(0, 2, 3, 1).contiguous()
    y_cpu = (y.argmax(dim=1) if (y.dim() > 1 and y.shape[1] > 1) else y).cpu()

    # ImageNet 反標準化用常數
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def denorm_img(img_hw3: np.ndarray) -> np.ndarray:
        # img 來源為 dataloader 已做 Normalize 的張量；這裡還原至 [0,1] 範圍
        if img_hw3.ndim == 3 and img_hw3.shape[-1] == 3:
            img = img_hw3 * IMAGENET_STD + IMAGENET_MEAN
            return np.clip(img, 0.0, 1.0)
        # 灰階或其他情況直接裁剪
        return np.clip(img_hw3, 0.0, 1.0)

    scenarios = [
        {
            'key': 'baseline',
            'title': 'Baseline',
            'layer_results': layer_results_base,
            'preds': pred_base,
            'confs': conf_base,
        },
        {
            'key': 'topk_keep',
            'title': f'only Top-{TOPK} Head',
            'layer_results': layer_results_keep,
            'preds': pred_keep,
            'confs': conf_keep,
        },
        {
            'key': 'topk_drop',
            'title': f'remove Top-{TOPK} Head',
            'layer_results': layer_results_drop,
            'preds': pred_drop,
            'confs': conf_drop,
        },
    ]

    # 儲存每個樣本的多情境對照圖（每層）
    for sample_idx in range(min(N, X.shape[0])):
        img = X_vis[sample_idx].numpy()
        img_dn = denorm_img(img)
        gt = int(y_cpu[sample_idx].item())
        # 原圖（加上 GT 和 Prediction）
        plt.figure(figsize=(6, 5))
        if img_dn.ndim == 3 and img_dn.shape[-1] == 3:
            plt.imshow(img_dn)
        else:
            plt.imshow(img_dn.squeeze(), cmap='gray')
        
        # 添加文字標籤
        status_symbol = "✓" if int(pred_base[sample_idx].item()) == gt else "✗"
        title_text = f"GT: {gt} | Pred: {int(pred_base[sample_idx].item())} {status_symbol} | Conf: {float(conf_base[sample_idx].item()):.3f}"
        plt.title(title_text, fontsize=12, pad=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{sample_idx}_original.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 單獨輸出各情境
        for scenario in scenarios:
            layer_items = list(scenario['layer_results'].items())
            num_layers = len(layer_items)
            cols = min(4, num_layers)
            rows = (num_layers + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
            axes = axes.flatten()
            for i, (lname, lmap) in enumerate(layer_items):
                if i >= len(axes):
                    break
                ax = axes[i]
                heat = lmap[sample_idx].detach().cpu().numpy()
                hmin, hmax = heat.min(), heat.max()
                heat_norm = (heat - hmin) / (hmax - hmin + 1e-12)
                im = ax.imshow(heat_norm, cmap='jet')
                layer_type = "Relevance" if lname == "input_relevance" else "Activation"
                ax.set_title(f'{lname} ({layer_type})\n[{hmin:.3f},{hmax:.3f}]', fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(num_layers, len(axes)):
                axes[i].axis('off')

            pred_val = int(scenario['preds'][sample_idx].item())
            conf_val = float(scenario['confs'][sample_idx].item())
            status = "✓" if pred_val == gt else "✗"
            fig.suptitle(
                f"{scenario['title']} Sample {sample_idx} {status} (GT:{gt}, Pred:{pred_val}, Conf:{conf_val:.3f})",
                fontsize=12
            )
            plt.tight_layout()
            plt.savefig(f"{out_dir}/sample_{sample_idx}_{scenario['key']}_all_layers.png", dpi=150, bbox_inches='tight')
            plt.close()

            # 疊加最後一層
            last_layer_map = layer_items[-1][1][sample_idx].detach().cpu().numpy()
            hl_min, hl_max = last_layer_map.min(), last_layer_map.max()
            hl_norm = (last_layer_map - hl_min) / (hl_max - hl_min + 1e-12)

            plt.figure(figsize=(6, 5))
            if img_dn.ndim == 3 and img_dn.shape[-1] == 3:
                plt.imshow(img_dn)
            else:
                plt.imshow(img_dn.squeeze(), cmap='gray')
            plt.imshow(hl_norm, cmap='jet', alpha=0.6)
            ax = plt.gca()
            if patch_h is not None and patch_w is not None:
                _draw_patch_grid(ax, hl_norm.shape[0], hl_norm.shape[1], patch_h, patch_w,
                                 color='white', lw=0.6, alpha=0.8)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/sample_{sample_idx}_{scenario['key']}_overlay.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 三種情境一次繪出（逐層比較）
        layer_names = list(layer_results_base.keys())
        num_layers = len(layer_names)
        num_scenarios = len(scenarios)
        fig, axes = plt.subplots(num_layers, num_scenarios,
                                 figsize=(4*num_scenarios, 4*num_layers))
        if num_layers == 1 and num_scenarios == 1:
            axes = np.array([[axes]])
        elif num_layers == 1:
            axes = axes[np.newaxis, :]
        elif num_scenarios == 1:
            axes = axes[:, np.newaxis]

        for row, lname in enumerate(layer_names):
            for col, scenario in enumerate(scenarios):
                ax = axes[row, col]
                lmap = scenario['layer_results'][lname][sample_idx].detach().cpu().numpy()
                hmin, hmax = lmap.min(), lmap.max()
                heat_norm = (lmap - hmin) / (hmax - hmin + 1e-12)
                im = ax.imshow(heat_norm, cmap='jet')
                layer_type = "Relevance" if lname == "input_relevance" else "Activation"
                ax.set_title(f"{scenario['title']}\n{lname} ({layer_type})\n[{hmin:.3f},{hmax:.3f}]", fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Sample {sample_idx} - 三情境逐層比較 (GT:{gt})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{sample_idx}_combined_all_layers.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 三種情境 overlay 對照
        fig, axes = plt.subplots(1, num_scenarios, figsize=(6*num_scenarios, 5))
        if num_scenarios == 1:
            axes = [axes]
        for col, scenario in enumerate(scenarios):
            ax = axes[col]
            last_layer_map = scenario['layer_results'][layer_names[-1]][sample_idx].detach().cpu().numpy()
            hl_min, hl_max = last_layer_map.min(), last_layer_map.max()
            hl_norm = (last_layer_map - hl_min) / (hl_max - hl_min + 1e-12)
            if img_dn.ndim == 3 and img_dn.shape[-1] == 3:
                ax.imshow(img_dn)
            else:
                ax.imshow(img_dn.squeeze(), cmap='gray')
            ax.imshow(hl_norm, cmap='jet', alpha=0.6)
            if patch_h is not None and patch_w is not None:
                _draw_patch_grid(ax, hl_norm.shape[0], hl_norm.shape[1], patch_h, patch_w,
                                 color='white', lw=0.6, alpha=0.8)
            pred_val = int(scenario['preds'][sample_idx].item())
            conf_val = float(scenario['confs'][sample_idx].item())
            status = "✓" if pred_val == gt else "✗"
            ax.set_title(f"{scenario['title']}\nPred:{pred_val} {status} | Conf:{conf_val:.3f}", fontsize=11)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{sample_idx}_combined_overlay.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"結果已輸出至: {out_dir}")


if __name__ == "__main__":
    main()


