#!/usr/bin/env python3
"""
範例：使用 VIT_with_Partial_LRP 分析 head 重要性
展示如何取得每層 attention head 的重要性分數，以及 input 的哪些部分比較重要
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models.VIT_with_Partial_LRP import VIT_with_Partial_LRP
from dataloader import get_dataloader
from config import config
import models

def _infer_patch_size(vit_model):
    """
    嘗試從 ViT 模型中推斷 patch 大小。
    回傳 (patch_h, patch_w) 的整數 tuple，失敗時回傳 None。
    """
    try:
        # 常見實作: model.patch_embed.patch_size -> (ph, pw)
        if hasattr(vit_model, 'patch_embed') and hasattr(vit_model.patch_embed, 'patch_size'):
            ps = vit_model.patch_embed.patch_size
            if isinstance(ps, (tuple, list)) and len(ps) == 2:
                return int(ps[0]), int(ps[1])
            if isinstance(ps, int):
                return int(ps), int(ps)
        # 另一些實作: model.patch_embed.proj 為 Conv2d，kernel_size 與 stride 為 patch 大小
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
    """
    在指定 axes 上依照 patch 尺寸畫出格線。
    """
    # 垂直線
    if patch_w > 0:
        x_positions = list(range(0, width + 1, patch_w))
        for x in x_positions:
            ax.axvline(x=x - 0.5, color=color, linewidth=lw, alpha=alpha)
    # 水平線
    if patch_h > 0:
        y_positions = list(range(0, height + 1, patch_h))
        for y in y_positions:
            ax.axhline(y=y - 0.5, color=color, linewidth=lw, alpha=alpha)

def main():
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 構建與訓練一致的 dataloader 與模型
    print("建立資料載入器與模型...")
    train_loader, test_loader = get_dataloader(
        dataset=config['dataset'],
        root=config['root'] + '/data/',
        batch_size=config['batch_size'],
        input_size=config['input_shape']
    )

    # 依照 config 生成同名模型
    model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
    model = model.to(device).eval()

    # 嘗試載入最新 checkpoint（與 train.py 一致的兩個保存位置）
    ckpt = None
    
    # 自動找到最新的 exp 資料夾
    import glob
    import os
    

    path = f"runs/train/exp62/VIT_best.pth"
    ckpt = torch.load(path, map_location=device)
    

    # 判斷是否需要包一層
    if isinstance(model, VIT_with_Partial_LRP):
        analyzer = model
    else:
        # 如果 model 是 VIT 包裝器，需要提取其 backbone
        if hasattr(model, 'backbone'):
            print("檢測到 VIT 包裝器，提取 backbone 用於 Partial LRP")
            vit_backbone = model.backbone
        else:
            vit_backbone = model
        
        analyzer = VIT_with_Partial_LRP(vit_backbone, topk_heads=None, head_weighting='normalize', eps=0.01).to(device).eval()

    # 載入權重（避免層級不對齊，使用 strict=False）
    if ckpt is not None and 'model_weights' in ckpt:
        # 診斷：列出 checkpoint 中分類頭的形狀（若存在）
        try:
            mw = ckpt['model_weights']
            w_keys = [k for k in mw.keys() if k.endswith('heads.head.weight')]
            b_keys = [k for k in mw.keys() if k.endswith('heads.head.bias')]
            if len(w_keys) == 0:
                # 嘗試 backbone 前綴
                w_keys = [k for k in mw.keys() if k.endswith('backbone.heads.head.weight')]
                b_keys = [k for k in mw.keys() if k.endswith('backbone.heads.head.bias')]
            if len(w_keys) > 0:
                w_shape = tuple(mw[w_keys[0]].shape)
                print(f"CKPT classifier weight key: {w_keys[0]}, shape={w_shape}")
            if len(b_keys) > 0:
                b_shape = tuple(mw[b_keys[0]].shape)
                print(f"CKPT classifier bias   key: {b_keys[0]}, shape={b_shape}")
        except Exception as e:
            print(f"診斷: 讀取 CKPT 分類頭形狀失敗: {e}")
        # 如果原始模型是 VIT 包裝器，需要調整權重路徑
        if hasattr(model, 'backbone'):
            print("調整權重路徑以匹配 VIT 包裝器結構")
            # 從 VIT 包裝器的權重中提取 backbone 的權重
            backbone_weights = {}
            for key, value in ckpt['model_weights'].items():
                if key.startswith('backbone.'):
                    # 移除 'backbone.' 前綴
                    new_key = key[9:]  # 移除 'backbone.'
                    backbone_weights[new_key] = value
                elif key.startswith('channel_adapter.'):
                    # 保留 channel_adapter 的權重
                    backbone_weights[key] = value
            
            # 載入到 analyzer 的內部模型
            missing_keys, unexpected_keys = analyzer.model.load_state_dict(backbone_weights, strict=False)
            print(f"載入權重到內部模型 - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        else:
            # 直接載入到 analyzer 的內部模型
            print("直接載入到 analyzer 的內部模型")
            missing_keys, unexpected_keys = analyzer.model.load_state_dict(ckpt['model_weights'], strict=False)
            print(f"載入權重到內部模型 - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # 從 dataloader 蒐集最多 N 筆樣本（可跨多個 batch）
    print("蒐集最多 20 筆樣本以一次繪製...")
    N = 20
    def collect_n_samples(loader, device, n):
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

    X, y = collect_n_samples(test_loader, device, N)
    if X is None:
        X, y = collect_n_samples(train_loader, device, N)
    if X is None:
        raise RuntimeError("無法從任何資料載入器中取得樣本。")
    batch_size = X.size(0)
    
    # 獲取模型預測結果
    print("獲取模型預測結果...")
    with torch.no_grad():
        logits = analyzer.model(X)
        predictions = logits.argmax(dim=1)
        confidence = torch.softmax(logits, dim=1).max(dim=1)[0]

    # 診斷：模型分類頭輸出維度與當前設定
    try:
        head_lin = analyzer.model.heads.head
        out_features = head_lin.out_features
        in_features = head_lin.in_features
        print(f"Model classifier head: in={in_features}, out={out_features}")
        # 額外：分類頭權重/偏置統計與偏置最大類別
        w = head_lin.weight.detach()
        b = head_lin.bias.detach() if head_lin.bias is not None else None
        print(f"Head weight stats -> mean: {w.mean().item():.4e}, std: {w.std().item():.4e}, absmax: {w.abs().max().item():.4e}")
        if b is not None:
            b_maxv, b_maxi = b.max().item(), int(b.argmax().item())
            b_minv, b_mini = b.min().item(), int(b.argmin().item())
            print(f"Head bias stats   -> min: {b_minv:.4e} (idx {b_mini}), max: {b_maxv:.4e} (idx {b_maxi})")
    except Exception as e:
        print(f"診斷: 取得模型分類頭資訊失敗: {e}")

    # 診斷：預測分布與信心統計
    try:
        uniq, counts = torch.unique(predictions, return_counts=True)
        print(f"Pred unique: {uniq.cpu().tolist()} counts: {counts.cpu().tolist()}")
        print(f"Confidence stats -> mean: {confidence.mean().item():.4f}, min: {confidence.min().item():.4f}, max: {confidence.max().item():.4f}")
        # logits 統計（每類平均），觀察是否某一類被偏置主導
        logit_means = logits.mean(dim=0)
        topk_vals, topk_idx = torch.topk(logit_means, k=min(5, logits.shape[1]))
        print(f"Mean logits topk -> idx: {topk_idx.cpu().tolist()}, vals: {[float(v) for v in topk_vals.cpu()]}")
    except Exception as e:
        print(f"診斷: 統計預測/信心失敗: {e}")

    # 診斷：CLS token 特徵範圍/範數（若 hooks 成功）
    try:
        if hasattr(analyzer, 'last_tokens') and analyzer.last_tokens is not None:
            h_cls = analyzer.last_tokens[:, 0, :]
            norms = h_cls.norm(dim=1)
            print(f"CLS token stats -> feature range: [{h_cls.min().item():.4e}, {h_cls.max().item():.4e}], L2 norm mean: {norms.mean().item():.4e}")
        else:
            print("CLS token stats -> last_tokens 尚未擷取，可能 hooks 未觸發")
    except Exception as e:
        print(f"診斷: 擷取 CLS token 統計失敗: {e}")
    
    # 處理 one-hot 編碼的標籤
    if y.dim() > 1 and y.shape[1] > 1:
        # 如果是 one-hot 編碼，轉換為整數標籤
        y_labels = y.argmax(dim=1)
        print(f"Ground Truth (one-hot): {y.cpu().tolist()}")
        print(f"Ground Truth (labels): {y_labels.cpu().tolist()}")
    else:
        # 如果已經是整數標籤
        y_labels = y
        print(f"Ground Truth: {y_labels.cpu().tolist()}")
    
    print(f"Predictions:  {predictions.cpu().tolist()}")
    print(f"Confidence:   {confidence.cpu().tolist()}")
    
    # 計算準確率
    correct = (predictions == y_labels).sum().item()
    accuracy = correct / batch_size
    print(f"Batch Accuracy: {correct}/{batch_size} = {accuracy:.2%}")
    
    # 方法 1: 基本使用 - 取得 relevance map 和 head 重要性
    print("\n=== 方法 1: 基本分析 ===")
    result = analyzer.explain(
        X, 
        target_class=None,  # 使用預測的類別
        return_map='patch', 
        return_head_importance=True
    )
    
    print(f"Relevance map 形狀: {result['relevance_map'].shape}")
    print(f"目標類別: {result['target_class']}")
    print(f"Head 重要性層數: {len(result['head_importance'])}")
    
    # 顯示每層的 head 重要性
    for i, layer_info in enumerate(result['head_importance']):
        head_scores = layer_info['head_scores']  # [B, H]
        mean_scores = head_scores.mean(dim=0)    # [H]
        top_heads = mean_scores.topk(3).indices.tolist()
        
        print(f"Layer {layer_info['layer']}:")
        print(f"  - Head 數量: {len(mean_scores)}")
        print(f"  - Top 3 heads: {top_heads}")
        print(f"  - 最高重要性分數: {mean_scores.max():.4f}")
        print(f"  - 平均重要性分數: {mean_scores.mean():.4f}")
    
    # 方法 2: 詳細分析 - 使用 analyze_head_importance
    print("\n=== 方法 2: 詳細分析與視覺化 ===")
    analysis = analyzer.analyze_head_importance(
        X, 
        target_class=None,
        save_plots=True,
        save_dir="./head_analysis_results"
    )
    
    print(f"分析結果已儲存到: ./head_analysis_results/")
    print(f"目標類別: {analysis['target_class']}")
    
    # 顯示詳細統計
    for layer_analysis in analysis['head_analysis']:
        print(f"\nLayer {layer_analysis['layer']} 詳細分析:")
        print(f"  - Head 數量: {layer_analysis['num_heads']}")
        print(f"  - Top 5 heads: {layer_analysis['top_heads']}")
        print(f"  - TopK 設定: {layer_analysis['topk_heads']}")
        print(f"  - 權重方式: {layer_analysis['head_weighting']}")
        
        # 顯示前 5 個 head 的分數
        mean_scores = np.array(layer_analysis['mean_scores'])
        top5_indices = np.argsort(mean_scores)[-5:][::-1]
        print(f"  - 前 5 個 head 分數:")
        for idx in top5_indices:
            print(f"    Head {idx}: {mean_scores[idx]:.4f}")
    
    # 方法 3: 比較不同類別的 head 重要性
    print("\n=== 方法 3: 比較不同類別 ===")
    target_classes = [0, 1, 2]  # 比較前 3 個類別
    
    for target_class in target_classes:
        print(f"\n分析類別 {target_class}:")
        result = analyzer.explain(
            X, 
            target_class=target_class,
            return_map='patch', 
            return_head_importance=True
        )
        
        # 計算每層最重要的 head
        for layer_info in result['head_importance']:
            head_scores = layer_info['head_scores'].mean(dim=0)
            top_head = head_scores.argmax().item()
            max_score = head_scores.max().item()
            print(f"  Layer {layer_info['layer']}: 最重要 head = {top_head} (分數: {max_score:.4f})")
    
    # 方法 4: 取得 input 重要性 (patch-level)
    print("\n=== 方法 4: Input 重要性分析 ===")
    relevance_map = analyzer.explain(X, return_map='patch')  # [B, num_patches]
    
    print(f"Patch relevance 形狀: {relevance_map.shape}")
    
    # 找出最重要的 patches
    for i in range(batch_size):
        patch_scores = relevance_map[i]  # [num_patches]
        top_patches = patch_scores.topk(10).indices.tolist()
        top_scores = patch_scores.topk(10).values.tolist()
        
        print(f"圖片 {i} 最重要的 10 個 patches:")
        for j, (patch_idx, score) in enumerate(zip(top_patches, top_scores)):
            print(f"  {j+1}. Patch {patch_idx}: {score:.4f}")
    
    # 方法 5: 取得 image-level 重要性
    print("\n=== 方法 5: Image-level 重要性 ===")
    image_relevance = analyzer.explain(X, return_map='image', upsample_to_input=True)
    print(f"Image relevance 形狀: {image_relevance.shape}")
    
    # 計算每個圖片的重要性統計
    for i in range(batch_size):
        img_rel = image_relevance[i]  # [H, W]
        max_val = img_rel.max().item()
        mean_val = img_rel.mean().item()
        std_val = img_rel.std().item()
        
        print(f"圖片 {i} 重要性統計:")
        print(f"  - 最大值: {max_val:.4f}")
        print(f"  - 平均值: {mean_val:.4f}")
        print(f"  - 標準差: {std_val:.4f}")
    
    # 另存熱圖對照（所有 head 綜合的整體熱圖）
    os.makedirs("./head_analysis_results", exist_ok=True)
    X_vis = X.detach().cpu()
    if X_vis.shape[1] == 3:
        X_vis = X_vis.permute(0, 2, 3, 1).contiguous()  # [B,H,W,3]
    img_rel = image_relevance.detach().cpu()

    # 取得 patch 大小 (若無法推斷，格線將略過)
    inferred_ps = _infer_patch_size(analyzer.model)
    if inferred_ps is not None:
        patch_h, patch_w = inferred_ps
    else:
        patch_h, patch_w = None, None
    
    # 獲取預測結果用於顯示
    pred_cpu = predictions.cpu()
    gt_cpu = y_labels.cpu()  # 使用轉換後的整數標籤
    conf_cpu = confidence.cpu()
    
    for i in range(batch_size):
        img = X_vis[i].numpy()
        heat = img_rel[i].numpy()
        
        # 正規化熱圖到 [0,1]
        hmin, hmax = heat.min(), heat.max()
        heat_norm = (heat - hmin) / (hmax - hmin + 1e-12)
        
        # 獲取當前樣本的預測資訊
        pred_class = pred_cpu[i].item()
        gt_class = gt_cpu[i].item()
        conf = conf_cpu[i].item()
        is_correct = pred_class == gt_class
        
        # 創建標題文字
        status = "✓" if is_correct else "✗"
        title = f"Sample {i} {status}\nGT: {gt_class}, Pred: {pred_class}, Conf: {conf:.3f}"

        # 單獨原圖
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.title(title, fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./head_analysis_results/sample_{i}_original.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 單獨熱圖
        plt.figure(figsize=(6, 5))
        plt.imshow(heat_norm, cmap='jet')
        plt.colorbar(label='Relevance Score')
        plt.title(title, fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./head_analysis_results/sample_{i}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 疊加原圖
        plt.figure(figsize=(6, 5))
        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imshow(np.clip(img, 0, 1))
        else:
            plt.imshow(img.squeeze(), cmap='gray')
        plt.imshow(heat_norm, cmap='jet', alpha=0.6)
        plt.title(title, fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./head_analysis_results/sample_{i}_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 原圖 + 格線
        if patch_h is not None and patch_w is not None:
            h_px, w_px = heat.shape
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            if img.ndim == 3 and img.shape[-1] == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img.squeeze(), cmap='gray')
            _draw_patch_grid(ax, h_px, w_px, patch_h, patch_w, color='white', lw=0.6, alpha=0.8)
            ax.set_title(title, fontsize=12, pad=20)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f'./head_analysis_results/sample_{i}_grid.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 熱圖 + 格線
        if patch_h is not None and patch_w is not None:
            h_px, w_px = heat.shape
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            ax.imshow(heat_norm, cmap='jet')
            _draw_patch_grid(ax, h_px, w_px, patch_h, patch_w, color='black', lw=0.6, alpha=0.8)
            ax.set_title(title, fontsize=12, pad=20)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f'./head_analysis_results/sample_{i}_heatmap_grid.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 疊加原圖 + 熱圖 + 格線
        if patch_h is not None and patch_w is not None:
            h_px, w_px = heat.shape
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            if img.ndim == 3 and img.shape[-1] == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img.squeeze(), cmap='gray')
            ax.imshow(heat_norm, cmap='jet', alpha=0.6)
            _draw_patch_grid(ax, h_px, w_px, patch_h, patch_w, color='white', lw=0.6, alpha=0.9)
            ax.set_title(title, fontsize=12, pad=20)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f'./head_analysis_results/sample_{i}_overlay_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 顯示每個樣本的詳細資訊
        print(f"\nSample {i}:")
        print(f"  - Ground Truth: {gt_class}")
        print(f"  - Prediction: {pred_class}")
        print(f"  - Confidence: {conf:.3f}")
        print(f"  - Correct: {'Yes' if is_correct else 'No'}")
        print(f"  - Heat range: [{hmin:.2f}, {hmax:.2f}]")

    print("\n=== 分析完成 ===")
    print("你可以查看 ./head_analysis_results/ 資料夾中的視覺化結果")

if __name__ == "__main__":
    main()


