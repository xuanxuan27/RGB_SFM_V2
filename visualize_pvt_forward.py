#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT Forward 流程圖視覺化腳本
載入訓練好的 PVT 模型並生成 forward 流程圖
"""

import torch
import torch.nn as nn
import torchviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 導入模型
from models.PVT import PVT, PVTv2

def load_trained_model(model_path, config):
    """
    載入訓練好的 PVT 模型
    
    Args:
        model_path: 模型權重檔案路徑
        config: 模型配置參數
    
    Returns:
        載入權重後的模型
    """
    print(f"正在載入模型: {model_path}")
    
    # 創建模型實例
    model = PVT(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        input_size=config['input_size'],
        head_schedule=config.get('head_schedule', 'auto'),
        head_dim_target=config.get('head_dim_target', 64),
        max_heads=config.get('max_heads', None),
        drop_rate=config.get('drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0)
    )
    
    # 載入權重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 處理不同的權重格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 載入權重
        model.load_state_dict(state_dict, strict=False)
        print("模型權重載入成功！")
    else:
        print(f"警告: 找不到模型檔案 {model_path}")
        print("將使用隨機初始化的權重進行視覺化")
    
    model.eval()
    return model

def create_torchviz_graph(model, input_tensor, save_path):
    """
    使用 torchviz 創建計算圖
    
    Args:
        model: PVT 模型
        input_tensor: 輸入張量
        save_path: 保存路徑
    """
    print("正在生成 torchviz 計算圖...")
    
    try:
        # 前向傳播
        output = model(input_tensor)
        
        # 創建計算圖
        dot = torchviz.make_dot(output, params=dict(model.named_parameters()))
        
        # 保存為 PDF 和 PNG
        dot.render(save_path, format='pdf', cleanup=True)
        dot.render(save_path, format='png', cleanup=True)
        
        print(f"計算圖已保存至: {save_path}.pdf 和 {save_path}.png")
        
    except Exception as e:
        print(f"生成 torchviz 圖時發生錯誤: {e}")

def create_manual_architecture_diagram(save_path):
    """
    手動創建 PVT 架構圖
    
    Args:
        save_path: 保存路徑
    """
    print("正在生成手動架構圖...")
    
    # 創建圖形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定義顏色
    colors = {
        'input': '#E8F4FD',
        'patch_embed': '#B3D9FF',
        'stage': '#87CEEB',
        'attention': '#FFB6C1',
        'mlp': '#98FB98',
        'output': '#FFA07A'
    }
    
    # 繪製各個階段
    stages = [
        {'name': 'Input\n(3, 224, 224)', 'pos': (1, 10), 'color': colors['input']},
        {'name': 'Patch Embed 1\n(64, 56, 56)', 'pos': (2.5, 10), 'color': colors['patch_embed']},
        {'name': 'Stage 1\n2 Blocks\n8 Heads', 'pos': (4, 10), 'color': colors['stage']},
        {'name': 'Patch Embed 2\n(128, 28, 28)', 'pos': (5.5, 10), 'color': colors['patch_embed']},
        {'name': 'Stage 2\n2 Blocks\n2 Heads', 'pos': (7, 10), 'color': colors['stage']},
        {'name': 'Patch Embed 3\n(320, 14, 14)', 'pos': (2.5, 7), 'color': colors['patch_embed']},
        {'name': 'Stage 3\n2 Blocks\n5 Heads', 'pos': (4, 7), 'color': colors['stage']},
        {'name': 'Patch Embed 4\n(512, 7, 7)', 'pos': (5.5, 7), 'color': colors['patch_embed']},
        {'name': 'Stage 4\n2 Blocks\n8 Heads', 'pos': (7, 7), 'color': colors['stage']},
        {'name': 'Global Avg Pool\n(512, 1, 1)', 'pos': (8.5, 7), 'color': colors['output']},
        {'name': 'Classification Head\n(30 classes)', 'pos': (8.5, 4), 'color': colors['output']}
    ]
    
    # 繪製方塊
    for stage in stages:
        x, y = stage['pos']
        width, height = 1.2, 0.8
        
        # 創建圓角矩形
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # 添加文字
        ax.text(x, y, stage['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold', wrap=True)
    
    # 繪製箭頭連接
    arrows = [
        ((1.6, 10), (1.9, 10)),  # Input -> Patch Embed 1
        ((3.1, 10), (3.4, 10)),  # Patch Embed 1 -> Stage 1
        ((4.6, 10), (4.9, 10)),  # Stage 1 -> Patch Embed 2
        ((6.1, 10), (6.4, 10)),  # Patch Embed 2 -> Stage 2
        ((7, 9.6), (2.5, 7.4)),  # Stage 2 -> Patch Embed 3
        ((3.1, 7), (3.4, 7)),    # Patch Embed 3 -> Stage 3
        ((4.6, 7), (4.9, 7)),    # Stage 3 -> Patch Embed 4
        ((6.1, 7), (6.4, 7)),    # Patch Embed 4 -> Stage 4
        ((7.6, 7), (7.9, 7)),    # Stage 4 -> Global Avg Pool
        ((8.5, 6.6), (8.5, 4.4)) # Global Avg Pool -> Classification
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加詳細說明
    details = [
        "PVT (Pyramid Vision Transformer) 架構說明:",
        "• 4 個階段，每個階段包含 Patch Embedding 和 Transformer Blocks",
        "• 使用 Spatial-Reduction Attention (SRA) 降低計算複雜度",
        "• 多尺度特徵提取: 56×56 → 28×28 → 14×14 → 7×7",
        "• 通道數遞增: 64 → 128 → 320 → 512",
        "• 注意力頭數自動計算，確保通道數能被整除"
    ]
    
    for i, detail in enumerate(details):
        ax.text(0.5, 2.5 - i*0.3, detail, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # 添加標題
    ax.text(5, 11.5, 'PVT Forward 流程圖', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_architecture.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}_architecture.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"架構圖已保存至: {save_path}_architecture.png 和 {save_path}_architecture.pdf")

def analyze_model_structure(model, input_tensor):
    """
    分析模型結構並打印詳細信息
    
    Args:
        model: PVT 模型
        input_tensor: 輸入張量
    """
    print("\n=== PVT 模型結構分析 ===")
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"總參數數量: {total_params:,}")
    print(f"可訓練參數數量: {trainable_params:,}")
    
    # 分析各層結構
    print("\n=== 各層詳細信息 ===")
    
    # 分析 backbone 結構
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        print(f"Backbone 類型: {type(backbone).__name__}")
        
        if hasattr(backbone, 'dims'):
            print(f"各階段通道數: {backbone.dims}")
        if hasattr(backbone, 'heads'):
            print(f"各階段注意力頭數: {backbone.heads}")
        if hasattr(backbone, 'blocks'):
            print(f"各階段 Block 數量: {backbone.blocks}")
        if hasattr(backbone, 'srs'):
            print(f"各階段 SR 比例: {backbone.srs}")
    
    # 前向傳播測試
    print("\n=== 前向傳播測試 ===")
    with torch.no_grad():
        output = model(input_tensor)
        print(f"輸入形狀: {input_tensor.shape}")
        print(f"輸出形狀: {output.shape}")
        print(f"輸出範圍: [{output.min():.4f}, {output.max():.4f}]")
        
        # 檢查是否有 NaN 或 Inf
        if torch.isnan(output).any():
            print("警告: 輸出包含 NaN 值!")
        if torch.isinf(output).any():
            print("警告: 輸出包含 Inf 值!")

def main():
    """主函數"""
    print("PVT Forward 流程圖視覺化工具")
    print("=" * 50)
    
    # 模型配置（從 config.py 讀取）
    model_config = {
        'in_channels': 3,
        'out_channels': 30,
        'input_size': (224, 224),
        'head_schedule': 'auto',
        'head_dim_target': 64,
        'max_heads': 16,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1
    }
    
    # 模型路徑
    model_path = "/home/xuan/RGB_SFM_V2/runs/train/exp64/PVT_best.pth"
    
    # 創建輸出目錄
    output_dir = "/home/xuan/RGB_SFM_V2/visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型
    model = load_trained_model(model_path, model_config)
    
    # 創建測試輸入
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 分析模型結構
    analyze_model_structure(model, input_tensor)
    
    # 生成 torchviz 計算圖
    torchviz_path = os.path.join(output_dir, "pvt_computation_graph")
    create_torchviz_graph(model, input_tensor, torchviz_path)
    
    # 生成手動架構圖
    manual_path = os.path.join(output_dir, "pvt_architecture")
    create_manual_architecture_diagram(manual_path)
    
    print("\n視覺化完成！")
    print(f"輸出檔案保存在: {output_dir}")
    print("包含以下檔案:")
    print("- pvt_computation_graph.pdf/png (torchviz 計算圖)")
    print("- pvt_architecture.pdf/png (手動架構圖)")

if __name__ == "__main__":
    main()
