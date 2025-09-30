#!/usr/bin/env python3
"""
Partial LRP ViT 使用範例

這個腳本展示如何使用帶有Partial Layer-wise Relevance Propagation的Vision Transformer
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from models.VIT_with_Partial_LRP import PartialLRPViT


def main():
    """主要使用範例"""
    
    print("=== Partial LRP ViT 使用範例 ===\n")
    
    # 1. 創建模型
    print("1. 創建 torchvision ViT 與 Partial LRP 包裝器...")
    num_classes = 10
    vit = torchvision.models.vit_b_16(weights=None)
    # 調整分類頭輸出類別數
    if hasattr(vit, 'heads') and hasattr(vit.heads, 'head'):
        in_c = vit.heads.head.in_features
        vit.heads.head = nn.Linear(in_c, num_classes)
    else:
        raise RuntimeError("未找到 torchvision ViT 的分類頭結構，請檢查版本。")

    explainer = PartialLRPViT(vit_model=vit, topk_heads=4, head_weighting='normalize', eps=1e-6)
    print("模型與解釋器創建完成！")
    
    # 2. 創建測試數據
    print("\n2. 創建測試數據...")
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"輸入張量形狀: {input_tensor.shape}")
    
    # 3. 前向傳播
    print("\n3. 執行前向傳播...")
    vit.eval()
    with torch.no_grad():
        output = vit(input_tensor)
    print(f"模型輸出形狀: {output.shape}")
    print(f"預測類別: {output.argmax(dim=1).tolist()}")
    
    # 4. 計算LRP相關性
    print("\n4. 計算LRP相關性...")
    
    # 對第一個樣本計算相關性
    sample_input = input_tensor[0:1]  # 取第一個樣本
    target_class = output[0].argmax().item()  # 使用預測的類別
    
    print(f"目標類別: {target_class}")
    
    # 計算輸入層的相關性
    input_relevance = explainer.compute_lrp_relevance(sample_input, target_class=target_class)
    print(f"輸入層相關性形狀: {input_relevance.shape}")
    
    # 5. 獲取所有層的相關性
    print("\n5. 獲取所有層的相關性...")
    all_relevance = explainer.compute_lrp_relevance(sample_input, target_class=target_class, return_intermediate=True)
    print(f"相關性層數: {len(all_relevance)}")
    for layer_name, relevance in all_relevance.items():
        if isinstance(relevance, torch.Tensor):
            print(f"  {layer_name}: {relevance.shape}")
        else:
            print(f"  {layer_name}: {type(relevance)}")
    
    # 6. 視覺化相關性
    print("\n6. 生成相關性視覺化...")
    relevance_vis = explainer.visualize_relevance(
        sample_input, 
        target_class=target_class,
        save_path="lrp_relevance_visualization.png"
    )
    print(f"相關性視覺化形狀: {relevance_vis.shape}")
    print("視覺化已保存到: lrp_relevance_visualization.png")
    
    # 7. 比較不同LRP規則
    print("\n7. 比較不同LRP規則...")
    
    lrp_rules = ["epsilon"]
    relevance_comparison = {}
    
    for rule in lrp_rules:
        print(f"  計算 {rule} 規則的相關性...")
        # 目前示範 epsilon 規則，其他規則略
        relevance_rule = explainer.compute_lrp_relevance(sample_input, target_class=target_class)
        relevance_comparison[rule] = relevance_rule
        print(f"    {rule} 相關性形狀: {relevance_rule.shape}")
    
    # 8. 分析相關性統計
    print("\n8. 分析相關性統計...")
    for rule, relevance in relevance_comparison.items():
        relevance_abs = torch.abs(relevance)
        print(f"  {rule}:")
        print(f"    平均相關性: {relevance_abs.mean().item():.6f}")
        print(f"    最大相關性: {relevance_abs.max().item():.6f}")
        print(f"    最小相關性: {relevance_abs.min().item():.6f}")
        print(f"    標準差: {relevance_abs.std().item():.6f}")
    
    # 9. 額外測試：不同 head 選擇（top-k）
    print("\n9. 測試不同 top-k heads 設定...")
    for k in [None, 1, 2, 4]:
        alt = PartialLRPViT(vit_model=vit, topk_heads=k, head_weighting='normalize')
        Rk = alt.compute_lrp_relevance(sample_input, target_class=target_class)
        print(f"  topk_heads={k}: mean|R|={torch.abs(Rk).mean().item():.6f}")
    
    print("\n=== 範例執行完成！ ===")


def compare_with_original_vit():
    """與原始ViT進行比較"""
    
    print("\n=== 與原始ViT比較 ===")
    
    # 使用 torchvision 的 ViT 與相同頭輸出
    num_classes = 10
    vit = torchvision.models.vit_b_16(weights=None)
    in_c = vit.heads.head.in_features
    vit.heads.head = nn.Linear(in_c, num_classes)
    lrp_vit = PartialLRPViT(vit_model=vit)
    
    # 測試輸入
    test_input = torch.randn(1, 3, 224, 224)
    
    # 比較預測結果
    vit.eval()
    
    with torch.no_grad():
        original_output = vit(test_input)
        lrp_output = vit(test_input)
    
    print(f"原始ViT預測: {original_output.argmax().item()}")
    print(f"LRP ViT預測: {lrp_output.argmax().item()}")
    print(f"預測差異: {torch.abs(original_output - lrp_output).max().item():.6f}")
    
    # 計算LRP相關性
    target_class = lrp_output.argmax().item()
    relevance = lrp_vit.compute_lrp_relevance(test_input, target_class=target_class)
    
    print(f"LRP相關性計算成功，形狀: {relevance.shape}")
    print(f"相關性範圍: [{relevance.min().item():.6f}, {relevance.max().item():.6f}]")


if __name__ == "__main__":
    # 執行主要範例
    main()
    
    # 與原始ViT比較
    compare_with_original_vit()
