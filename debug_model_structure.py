#!/usr/bin/env python3
"""
Debug: 檢查 ViT 模型結構
"""

import torch
import torchvision.models as tv_models

def main():
    # 載入模型
    model = tv_models.vit_b_16(weights=None)
    
    print("=== ViT 模型結構 ===")
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    # 檢查是否有 heads 屬性
    if hasattr(model, 'heads'):
        print(f"\nheads 類型: {type(model.heads)}")
        print(f"heads 屬性: {[attr for attr in dir(model.heads) if not attr.startswith('_')]}")
        
        if hasattr(model.heads, 'head'):
            print(f"\nhead 類型: {type(model.heads.head)}")
            print(f"head 權重形狀: {model.heads.head.weight.shape}")
            print(f"head bias 形狀: {model.heads.head.bias.shape}")
    
    # 檢查是否有 head 屬性
    if hasattr(model, 'head'):
        print(f"\nhead 類型: {type(model.head)}")
        print(f"head 權重形狀: {model.head.weight.shape}")
        print(f"head bias 形狀: {model.head.bias.shape}")
    
    # 檢查 encoder 結構
    if hasattr(model, 'encoder'):
        print(f"\nencoder 類型: {type(model.encoder)}")
        print(f"encoder 屬性: {[attr for attr in dir(model.encoder) if not attr.startswith('_')]}")
        
        if hasattr(model.encoder, 'ln'):
            print(f"encoder.ln 類型: {type(model.encoder.ln)}")
    
    # 測試前向傳播
    print("\n=== 測試前向傳播 ===")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        print(f"輸出形狀: {output.shape}")
        print(f"輸出範圍: [{output.min():.4f}, {output.max():.4f}]")

if __name__ == "__main__":
    main()
