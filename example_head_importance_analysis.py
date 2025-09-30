#!/usr/bin/env python3
"""
範例：使用 VIT_with_Partial_LRP 分析 head 重要性
展示如何取得每層 attention head 的重要性分數，以及 input 的哪些部分比較重要
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.VIT_with_Partial_LRP import VIT_with_Partial_LRP
from dataloader import get_dataloader
from config import config
import models

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
    try:
        ckpt = torch.load(f"{config['save_dir']}/{config['model']['name']}_best.pth", map_location=device)
    except Exception:
        try:
            ckpt = torch.load(f"./pth/{config['dataset']}/{config['load_model_name']}.pth", map_location=device)
        except Exception:
            print("[warn] 無法找到 checkpoint，將以隨機權重進行分析")
            ckpt = None

    # 判斷是否需要包一層
    if isinstance(model, VIT_with_Partial_LRP):
        analyzer = model
    else:
        analyzer = VIT_with_Partial_LRP(model, topk_heads=None, head_weighting='normalize', eps=1e-6).to(device).eval()

    # 載入權重（避免層級不對齊，使用 strict=False）
    if ckpt is not None and 'model_weights' in ckpt:
        _keys = analyzer.load_state_dict(ckpt['model_weights'], strict=False)

    # 從 test_loader 取一個 batch，確保與訓練前處理一致
    print("擷取資料批次...")
    try:
        X, y = next(iter(test_loader))
    except StopIteration:
        X, y = next(iter(train_loader))
    X = X.to(device)
    batch_size = X.size(0)
    
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
        top_patches = patch_scores.topk(5).indices.tolist()
        top_scores = patch_scores.topk(5).values.tolist()
        
        print(f"圖片 {i} 最重要的 5 個 patches:")
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
    
    print("\n=== 分析完成 ===")
    print("你可以查看 ./head_analysis_results/ 資料夾中的視覺化結果")

if __name__ == "__main__":
    main()


