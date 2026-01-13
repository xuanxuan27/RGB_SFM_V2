#!/usr/bin/env python3
"""
VIT_PartialLRP_PatchMerging 視覺化腳本
比較使用 all head、topk head、non-topk head 的 activation map
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from models.VIT_PartialLRP_PatchMerging import VIT_PartialLRP_PatchMerging
from dataloader import get_dataloader
from config import config
import models

def _infer_patch_size(vit_model):
    """嘗試從 ViT 模型中推斷 patch 大小"""
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
        if hasattr(vit_model, 'conv_proj'):
            ks = vit_model.conv_proj.kernel_size
            if isinstance(ks, (tuple, list)) and len(ks) == 2:
                return int(ks[0]), int(ks[1])
            if isinstance(ks, int):
                return int(ks), int(ks)
    except Exception:
        pass
    return None

def _draw_patch_grid(ax, height, width, patch_h, patch_w, color='white', lw=0.6, alpha=0.8):
    """在指定 axes 上依照 patch 尺寸畫出格線"""
    if patch_w > 0:
        x_positions = list(range(0, width + 1, patch_w))
        for x in x_positions:
            ax.axvline(x=x - 0.5, color=color, linewidth=lw, alpha=alpha)
    if patch_h > 0:
        y_positions = list(range(0, height + 1, patch_h))
        for y in y_positions:
            ax.axhline(y=y - 0.5, color=color, linewidth=lw, alpha=alpha)

def normalize_act(act: np.ndarray) -> np.ndarray:
    """正規化 activation map 到 [0, 1]"""
    act_min, act_max = act.min(), act.max()
    if act_max > act_min:
        return (act - act_min) / (act_max - act_min)
    return act

def denorm_img(img_hw3: np.ndarray, imagenet_mean=None, imagenet_std=None) -> np.ndarray:
    """將正規化後的圖像還原至 [0,1] 範圍"""
    if imagenet_mean is None:
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    if imagenet_std is None:
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    if img_hw3.ndim == 3 and img_hw3.shape[-1] == 3:
        img = img_hw3 * imagenet_std + imagenet_mean
        return np.clip(img, 0.0, 1.0)
    return np.clip(img_hw3, 0.0, 1.0)

def get_topk_heads_per_layer(analyzer, X, topk: int = 5) -> Dict[int, List[int]]:
    """
    獲取每層的 topk heads（使用新的 build_head_mask_from_importance 函數）
    Returns:
        Dict: {layer_idx: [head_idx1, head_idx2, ...]}
    """
    # 使用 analyze_head_importance 獲取 head importance 分析
    analysis = analyzer.analyze_head_importance(X, save_plots=False)
    
    # 使用新的 build_head_mask_from_importance 函數建立 mapping
    layer_to_heads = analyzer.build_head_mask_from_importance(analysis, topk=topk)
    
    return layer_to_heads

def get_activation_maps(analyzer, X, head_config: str = 'all', topk: int = 5, 
                      layer_to_heads: Optional[Dict[int, List[int]]] = None) -> Dict[str, torch.Tensor]:
    """
    獲取每層的 activation maps
    Args:
        analyzer: VIT_PartialLRP_PatchMerging 模型
        X: 輸入圖像 [B, 3, H, W]
        head_config: 'all', 'topk', 'non_topk'
        topk: topk heads 的數量
        layer_to_heads: 預先計算的 layer_to_heads（避免重複計算）
    Returns:
        Dict: {layer_name: activation_map [B, H, W]}
    """
    # 清除之前的 cache
    analyzer._clear_memory_cache()
    
    # 根據 head_config 設定 head mask
    if head_config == 'all':
        analyzer.clear_head_mask()
    elif head_config == 'topk':
        # 使用預先計算的或重新計算 topk heads
        if layer_to_heads is None:
            layer_to_heads = get_topk_heads_per_layer(analyzer, X, topk=topk)
        analyzer.set_head_mask(layer_to_heads, keep_only=True)
    elif head_config == 'non_topk':
        # 獲取 topk heads，然後設定為保留其他 heads
        if layer_to_heads is None:
            layer_to_heads = get_topk_heads_per_layer(analyzer, X, topk=topk)
        analyzer.set_head_mask(layer_to_heads, keep_only=False)
    else:
        analyzer.clear_head_mask()
    
    # 獲取 activation maps（同時獲取原始大小和上採樣版本）
    # 原始大小用於獲取實際空間維度信息
    activation_maps_raw = analyzer.get_per_layer_activations(
        X,
        return_map='image',
        upsample_to_input=False  # 原始大小，用於獲取實際維度
    )
    
    # 上採樣版本用於顯示（更平滑，沒有馬賽克）
    activation_maps_upsampled = analyzer.get_per_layer_activations(
        X,
        return_map='image',
        upsample_to_input=False  # 上採樣版本，視覺效果更好
    )
    
    # 合併：使用上採樣版本顯示，但保存原始維度信息
    activation_maps = {}
    for layer_name in activation_maps_raw.keys():
        raw_map = activation_maps_raw[layer_name][0]  # [H_raw, W_raw]
        upsampled_map = activation_maps_upsampled[layer_name][0]  # [H_input, W_input]
        H_raw, W_raw = raw_map.shape
        
        # 保存上採樣版本用於顯示，但記錄原始維度
        activation_maps[layer_name] = {
            'map': upsampled_map,  # 上採樣後的圖像（用於顯示）
            'raw_shape': (H_raw, W_raw)  # 原始空間維度（用於標題）
        }
    
    return activation_maps

def evaluate_head_config_accuracy(analyzer,
                                  data_loader,
                                  device,
                                  head_config: str = 'all',
                                  topk: int = 5,
                                  layer_to_heads: Optional[Dict[int, List[int]]] = None,
                                  max_samples: Optional[int] = None,
                                  compute_per_sample: bool = True) -> float:
    """
    計算指定 head 設定下的分類準確率
    Args:
        max_samples: 限制評估的樣本數量（避免 OOM）
        compute_per_sample: 是否對每個樣本都重新計算 topk heads（僅對 topk/non_topk 有效）
    """
    # 先清理所有 cache
    analyzer._clear_memory_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    analyzer.model.eval()
    correct = 0
    total = 0
    
    # 如果 head_config 是 'all'，不需要動態計算
    if head_config == 'all':
        analyzer.clear_head_mask()
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(data_loader):
                if max_samples is not None and total >= max_samples:
                    break
                xb = xb.to(device)
                yb = yb.to(device)
                logits = analyzer.model(xb)
                preds = logits.argmax(dim=1)
                if yb.dim() > 1 and yb.shape[1] > 1:
                    labels = yb.argmax(dim=1)
                else:
                    labels = yb.view(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # 每個 batch 後清理 cache
                analyzer._clear_memory_cache()
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    else:
        # 對於 topk 或 non_topk，需要動態計算 head importance
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(data_loader):
                if max_samples is not None and total >= max_samples:
                    break
                xb = xb.to(device)
                yb = yb.to(device)
                
                # 對 batch 中的每個樣本分別處理
                batch_size = xb.size(0)
                batch_preds = []
                batch_labels = []
                
                for sample_idx in range(batch_size):
                    x_sample = xb[sample_idx:sample_idx+1]  # [1, C, H, W]
                    y_sample = yb[sample_idx:sample_idx+1]
                    
                    # 對每個樣本重新計算 head importance
                    if compute_per_sample:
                        analyzer._clear_memory_cache()
                        sample_analysis = analyzer.analyze_head_importance(x_sample, save_plots=False)
                        sample_layer_to_heads = analyzer.build_head_mask_from_importance(sample_analysis, topk=topk)
                    else:
                        # 使用預先計算的 layer_to_heads
                        if layer_to_heads is None:
                            raise ValueError("需要提供 layer_to_heads 或設定 compute_per_sample=True")
                        sample_layer_to_heads = layer_to_heads
                    
                    # 設定 head mask
                    analyzer.clear_head_mask()
                    if head_config == 'topk':
                        analyzer.set_head_mask(sample_layer_to_heads, keep_only=True)
                    elif head_config == 'non_topk':
                        analyzer.set_head_mask(sample_layer_to_heads, keep_only=False)
                    
                    # 進行預測
                    logits = analyzer.model(x_sample)
                    pred = logits.argmax(dim=1)
                    batch_preds.append(pred)
                    
                    # 處理 label
                    if y_sample.dim() > 1 and y_sample.shape[1] > 1:
                        label = y_sample.argmax(dim=1)
                    else:
                        label = y_sample.view(-1)
                    batch_labels.append(label)
                    
                    # 清理 cache
                    analyzer._clear_memory_cache()
                
                # 合併 batch 結果
                batch_preds = torch.cat(batch_preds, dim=0)
                batch_labels = torch.cat(batch_labels, dim=0)
                correct += (batch_preds == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                # 每個 batch 後清理 cache
                analyzer._clear_memory_cache()
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    analyzer.clear_head_mask()
    analyzer._clear_memory_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return correct / total if total > 0 else 0.0

def plot_head_relevance_heatmap(analysis: Dict, sample_idx: int, save_dir: str,
                                gt: int, pred: int, confidence: float, status: str):
    """
    根據 analyzer.analyze_head_importance 的結果繪製每層每個 head 的 relevance heatmap
    """
    if not isinstance(analysis, dict) or 'head_analysis' not in analysis:
        print("Warning: 無 head importance 資料可視覺化")
        return
    
    head_analysis = analysis.get('head_analysis', [])
    if len(head_analysis) == 0:
        print("Warning: head_analysis 為空，略過繪圖")
        return
    
    # 依層索引排序
    head_analysis = sorted(head_analysis, key=lambda x: x.get('layer', 0))
    layer_labels = [f"layer_{item.get('layer', idx)}" for idx, item in enumerate(head_analysis)]
    max_heads = max(len(item.get('mean_scores', [])) for item in head_analysis)
    if max_heads == 0:
        print("Warning: head_analysis 裡沒有 mean_scores 資料")
        return
    
    heatmap = np.full((len(layer_labels), max_heads), np.nan, dtype=np.float32)
    for row_idx, item in enumerate(head_analysis):
        scores = item.get('mean_scores', [])
        if isinstance(scores, list):
            scores_np = np.array(scores, dtype=np.float32)
        else:
            scores_np = np.array(scores.detach().cpu().tolist(), dtype=np.float32)
        heatmap[row_idx, :len(scores_np)] = scores_np
    
    plt.figure(figsize=(max(10, max_heads * 0.7), max(6, len(layer_labels) * 0.5)))
    im = plt.imshow(heatmap, aspect='auto', cmap='plasma')
    plt.colorbar(im, label='Head Relevance Score')
    plt.xticks(range(max_heads), [str(i) for i in range(max_heads)])
    plt.yticks(range(len(layer_labels)), layer_labels)
    plt.xlabel('Head Index')
    plt.ylabel('Layer')
    plt.title(f"Sample {sample_idx} {status} - Head Relevance Heatmap\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_{sample_idx}_head_relevance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_sample(analyzer, X, y, sample_idx: int, save_dir: str, topk: int = 8):
    """
    視覺化單個樣本
    """
    device = next(analyzer.model.parameters()).device
    X_sample = X[sample_idx:sample_idx+1].to(device)
    y_sample = y[sample_idx:sample_idx+1].to(device)
    
    # 獲取預測結果
    with torch.no_grad():
        logits = analyzer.model(X_sample)
        pred = logits.argmax(dim=1).cpu().item()
        confidence = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().item()
    
    # 處理 ground truth
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
    
    # 1. 保存原圖（帶 GT 和 Pred）
    plt.figure(figsize=(8, 8))
    plt.imshow(img_denorm)
    title = f"Sample {sample_idx} {status}\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}"
    plt.title(title, fontsize=12, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_{sample_idx}_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 先計算 head importance（用於後續的 topk 配置）
    print(f"計算 sample {sample_idx} 的 head importance...")
    analysis = analyzer.analyze_head_importance(X_sample, save_plots=False)
    plot_head_relevance_heatmap(
        analysis,
        sample_idx=sample_idx,
        save_dir=save_dir,
        gt=gt,
        pred=pred,
        confidence=confidence,
        status=status
    )
    layer_to_heads = analyzer.build_head_mask_from_importance(analysis, topk=topk)
    print(f"Sample {sample_idx} 使用的 topk heads: {layer_to_heads}")
    
    # 3. 為三種 head 配置分別計算 relevance map
    head_configs = [
        ('all', 'All Heads', None),
        ('topk', f'TopK Heads (k={topk})', layer_to_heads),
        ('non_topk', 'Non-TopK Heads', layer_to_heads)
    ]
    
    for head_config, config_name, layer_to_heads_for_config in head_configs:
        print(f"計算 sample {sample_idx} 的 relevance map ({config_name})...")
        
        # 設定 head mask
        analyzer.clear_head_mask()
        if head_config == 'topk' and layer_to_heads_for_config is not None:
            analyzer.set_head_mask(layer_to_heads_for_config, keep_only=True)
        elif head_config == 'non_topk' and layer_to_heads_for_config is not None:
            analyzer.set_head_mask(layer_to_heads_for_config, keep_only=False)
        # 'all' 的情況已經 clear_head_mask() 處理了
        
        # 計算 relevance map
        relevance_map = analyzer.explain(
            X_sample,
            target_class=None,
            return_map='image',
            upsample_to_input=True
        )
        rel_map = relevance_map[0].detach().cpu().numpy()  # [H, W]
        
        # 正規化 relevance map
        rel_min, rel_max = rel_map.min(), rel_map.max()
        if rel_max > rel_min:
            rel_norm = (rel_map - rel_min) / (rel_max - rel_min)
        else:
            rel_norm = rel_map
        
        # 保存 relevance map
        config_suffix = {
            'all': 'all_heads',
            'topk': 'topk_heads',
            'non_topk': 'non_topk_heads'
        }[head_config]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(rel_norm, cmap='jet')
        plt.colorbar(label='Relevance Score')
        plt.title(f"Sample {sample_idx} {status} - {config_name} Relevance Map\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}", 
                 fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sample_{sample_idx}_relevance_map_{config_suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存 relevance map 疊加圖
        plt.figure(figsize=(8, 8))
        plt.imshow(img_denorm)
        plt.imshow(rel_norm, cmap='jet', alpha=0.6)
        plt.title(f"Sample {sample_idx} {status} - {config_name} Relevance Overlay\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}", 
                 fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sample_{sample_idx}_relevance_overlay_{config_suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 清理 cache
        analyzer._clear_memory_cache()
    
    # 恢復到 all heads 狀態
    analyzer.clear_head_mask()
    
    # 4. 獲取三種 head 配置的 activation maps
    
    print(f"計算 sample {sample_idx} 的 activation maps (all heads)...")
    act_all = get_activation_maps(analyzer, X_sample, head_config='all', topk=topk, layer_to_heads=layer_to_heads)
    
    print(f"計算 sample {sample_idx} 的 activation maps (topk heads)...")
    act_topk = get_activation_maps(analyzer, X_sample, head_config='topk', topk=topk, layer_to_heads=layer_to_heads)
    
    print(f"計算 sample {sample_idx} 的 activation maps (non-topk heads)...")
    act_non_topk = get_activation_maps(analyzer, X_sample, head_config='non_topk', topk=topk, layer_to_heads=layer_to_heads)
    
    # 獲取所有層的名稱（假設三種配置的層數相同）
    # 按照層號的數字順序排序，而不是字符串順序
    def extract_layer_number(layer_name):
        """從 layer_name 中提取層號（例如 'layer_10' -> 10）"""
        try:
            return int(layer_name.split('_')[1])
        except (IndexError, ValueError):
            return -1
    
    layer_names = sorted(
        [k for k in act_all.keys() if k.startswith('layer_')],
        key=extract_layer_number
    )
    
    # 創建所有層的綜合比較圖（每種配置一行，顯示所有層）
    num_layers = len(layer_names)
    if num_layers > 0:
        # 計算合適的列數（每行顯示所有層）
        cols = num_layers
        rows = 3  # 三種配置：All Heads, TopK Heads, Non-TopK Heads
        row_configs = [
            ('All Heads', act_all, f'Baseline (All Heads)\nGT: {gt}, Pred: {pred}'),
            (f'TopK Heads (k={topk})', act_topk, f'TopK Heads (k={topk})\nGT: {gt}, Pred: {pred}'),
            ('Non-TopK Heads', act_non_topk, f'Remaining Heads\nGT: {gt}, Pred: {pred}')
        ]
        
        # 根據層數調整圖像大小
        fig_width = max(3 * cols, 20)  # 至少 20 寬度，每層至少 3 單位
        fig_height = 4 * rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 為每種配置和每層繪製
        for row_idx, (config_name, act_dict, _) in enumerate(row_configs):
            for col_idx, layer_name in enumerate(layer_names):
                if layer_name in act_dict:
                    act_data = act_dict[layer_name]
                    
                    # 處理新的數據結構
                    if isinstance(act_data, dict):
                        act = act_data['map'].detach().cpu().numpy()  # 上採樣版本
                        H_act, W_act = act_data['raw_shape']  # 原始維度
                    else:
                        # 向後兼容
                        act = act_data[0].detach().cpu().numpy()
                        H_act, W_act = act.shape
                    
                    act_norm = normalize_act(act)
                    
                    ax = axes[row_idx, col_idx]
                    im = ax.imshow(act_norm, cmap='jet')
                    # 第一行顯示層名稱和原始空間維度（即使圖像是上採樣的）
                    if row_idx == 0:
                        ax.set_title(f'{layer_name}\n原始: {H_act}×{W_act}', fontsize=10, pad=5)
                    else:
                        ax.set_title('', fontsize=8)
                    ax.axis('off')
                    # 只在最後一列添加 colorbar
                    if col_idx == cols - 1:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 在左側添加配置名稱
        for row_idx, (_, _, label_text) in enumerate(row_configs):
            fig.text(0.02, 0.75 - row_idx * 0.33, label_text, 
                    fontsize=12, rotation=90, ha='center', va='center', weight='bold')
        
        fig.suptitle(f"Sample {sample_idx} {status} - All Layers Activation Comparison\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}", 
                    fontsize=16, y=0.995, weight='bold')
        plt.tight_layout(rect=[0.05, 0, 1, 0.98])  # 為左側標籤留空間
        plt.savefig(f'{save_dir}/sample_{sample_idx}_all_layers_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 創建一個更緊湊的版本（如果層數太多，可以分組顯示）
        if num_layers > 8:
            # 分兩組顯示
            mid_point = num_layers // 2
            for group_idx, (start_idx, end_idx) in enumerate([(0, mid_point), (mid_point, num_layers)]):
                group_layers = layer_names[start_idx:end_idx]
                group_cols = len(group_layers)
                group_rows = 3
                
                fig, axes = plt.subplots(group_rows, group_cols, figsize=(3*group_cols, 4*group_rows))
                if group_rows == 1:
                    axes = axes.reshape(1, -1)
                elif group_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for row_idx, (config_name, act_dict, _) in enumerate(row_configs):
                    for col_idx, layer_name in enumerate(group_layers):
                        if layer_name in act_dict:
                            act_data = act_dict[layer_name]
                            
                            # 處理新的數據結構
                            if isinstance(act_data, dict):
                                act = act_data['map'].detach().cpu().numpy()  # 上採樣版本
                                H_act, W_act = act_data['raw_shape']  # 原始維度
                            else:
                                # 向後兼容
                                act = act_data[0].detach().cpu().numpy()
                                H_act, W_act = act.shape
                            
                            act_norm = normalize_act(act)
                            
                            ax = axes[row_idx, col_idx]
                            im = ax.imshow(act_norm, cmap='jet')
                            if row_idx == 0:
                                ax.set_title(f'{layer_name}\n原始: {H_act}×{W_act}', fontsize=10)
                            ax.axis('off')
                            if col_idx == group_cols - 1:
                                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                for row_idx, (_, _, label_text) in enumerate(row_configs):
                    fig.text(0.02, 0.75 - row_idx * 0.33, label_text, 
                            fontsize=12, rotation=90, ha='center', va='center', weight='bold')
                
                fig.suptitle(f"Sample {sample_idx} {status} - Layers {start_idx}-{end_idx-1} Activation Comparison\nGT: {gt}, Pred: {pred}, Conf: {confidence:.3f}", 
                            fontsize=14, y=0.995)
                plt.tight_layout(rect=[0.05, 0, 1, 0.98])
                plt.savefig(f'{save_dir}/sample_{sample_idx}_all_layers_comparison_group{group_idx+1}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"Sample {sample_idx} 視覺化完成")

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
    
    # 載入權重
    weight_path = "runs/train/exp108/VIT_best.pth"
    print(f"載入權重: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device)
    if ckpt is not None and 'model_weights' in ckpt:
        print("先在包裝前載入權重到原始模型，避免 hook 後鍵名不一致")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_weights'], strict=False)
        print(f"原始模型載入結果 - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # 判斷是否需要包一層
    if isinstance(model, VIT_PartialLRP_PatchMerging):
        analyzer = model
    else:
        # 如果 model 是 VIT 包裝器，需要提取其 backbone
        if hasattr(model, 'backbone'):
            print("檢測到 VIT 包裝器，提取 backbone 用於 Partial LRP")
            vit_backbone = model.backbone
        else:
            vit_backbone = model
        
        analyzer = VIT_PartialLRP_PatchMerging(
            vit_backbone,
            topk_heads=None,  # 先不設定，稍後動態設定
            head_weighting='normalize',
            eps=0.01
        ).to(device).eval()
    
    # 從 dataloader 蒐集樣本
    print("蒐集樣本...")
    N = 4  # 處理 4 個樣本
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
    
    # 使用收集到的樣本建立 head importance 參考
    topk = 3
    reference_batch = X[:min(4, X.size(0))]
    print(f"使用 {reference_batch.size(0)} 個樣本建立 head importance 參考...")
    reference_analysis = analyzer.analyze_head_importance(reference_batch, save_plots=False)
    reference_layer_to_heads = analyzer.build_head_mask_from_importance(reference_analysis, topk=topk)
    
    # 清理參考批次相關的記憶體
    del reference_batch, reference_analysis
    analyzer._clear_memory_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 限制評估樣本數以避免 OOM（評估前 1000 個樣本）
    max_eval_samples = 1000
    
    print("計算 Baseline (All Heads) accuracy...")
    baseline_acc = evaluate_head_config_accuracy(
        analyzer, test_loader, device, head_config='all', max_samples=max_eval_samples
    )
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    
    # 清理 cache
    analyzer._clear_memory_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # print("計算 TopK Heads accuracy（對每個樣本重新計算 topk heads）...")
    # topk_acc = evaluate_head_config_accuracy(
    #     analyzer, test_loader, device, head_config='topk', topk=topk, 
    #     max_samples=max_eval_samples, compute_per_sample=True
    # )
    # print(f"TopK Heads Accuracy: {topk_acc*100:.2f}%")
    
    # # 清理 cache
    # analyzer._clear_memory_cache()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    
    # print("計算 Remove TopK Heads accuracy（對每個樣本重新計算 topk heads）...")
    # non_topk_acc = evaluate_head_config_accuracy(
    #     analyzer, test_loader, device, head_config='non_topk', topk=topk, 
    #     max_samples=max_eval_samples, compute_per_sample=True
    # )
    # print(f"Remove TopK Heads Accuracy: {non_topk_acc*100:.2f}%")
    
    # 最終清理
    analyzer._clear_memory_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 創建保存目錄
    save_dir = "./vit_patch_merging_visualization"
    os.makedirs(save_dir, exist_ok=True)
    print(f"結果將保存到: {save_dir}")
    
    # 處理每個樣本
    topk = 5
    for i in range(min(N, X.size(0))):
        print(f"\n處理樣本 {i}...")
        visualize_sample(analyzer, X, y, i, save_dir, topk=topk)
    
    print(f"\n所有視覺化完成！結果保存在: {save_dir}")

if __name__ == "__main__":
    main()

