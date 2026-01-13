#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版 PVT Stage 特徵可視化腳本
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from pathlib import Path

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
matplotlib.rcParams['font.family'] = 'sans-serif'

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 導入模型
from models.PVT import PVT

def load_model_and_visualize():
    """載入模型並進行可視化"""
    
    # 模型配置
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
    
    print("正在載入 PVT 模型...")
    
    # 創建模型
    model = PVT(**model_config)
    
    # 嘗試載入權重
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("✓ 模型權重載入成功！")
        except Exception as e:
            print(f"⚠ 載入權重時發生錯誤: {e}")
            print("使用隨機初始化的權重")
    else:
        print(f"⚠ 找不到模型檔案: {model_path}")
        print("使用隨機初始化的權重")
    
    model.eval()
    
    # 創建測試輸入
    print("創建測試輸入...")
    test_input = torch.randn(1, 3, 224, 224)
    
    # 提取特徵
    print("提取各 stage 特徵...")
    with torch.no_grad():
        backbone = model.backbone
        stage_features = []
        stage_info = []
        
        B = test_input.shape[0]
        x = test_input
        
        # 追蹤上一層的 (C, H, W)
        prev_C, prev_H, prev_W = None, None, None
        seq = None
        
        for i in range(4):
            if i == 0:
                x_img = x
            else:
                x_img = seq.transpose(1, 2).reshape(B, prev_C, prev_H, prev_W)
            
            # Patch Embedding
            seq, H, W = backbone.patch_embeds[i](x_img)
            
            # 經過多層 Block
            for blk in backbone.stages[i]:
                seq = blk(seq, H, W)
            
            # 正規化並轉換為特徵圖
            seq_out = backbone.norms[i](seq)
            feat = seq_out.transpose(1, 2).reshape(B, backbone.dims[i], H, W)
            
            # 存儲特徵和信息
            stage_features.append(feat[0])  # 取第一個樣本
            stage_info.append({
                'stage': i + 1,
                'channels': backbone.dims[i],
                'height': H,
                'width': W,
                'heads': backbone.heads[i] if hasattr(backbone, 'heads') else 'N/A',
                'sr_ratio': backbone.srs[i] if hasattr(backbone, 'srs') else 'N/A'
            })
            
            prev_C, prev_H, prev_W = backbone.dims[i], H, W
            last_seq = seq_out
    
    # 打印信息
    print("\n=== 各 Stage 信息 ===")
    for info in stage_info:
        print(f"Stage {info['stage']}: C={info['channels']}, H={info['height']}, W={info['width']}, "
              f"Heads={info['heads']}, SR={info['sr_ratio']}")
    
    # 創建可視化
    print("\n正在生成可視化圖表...")
    
    # 創建輸出目錄
    output_dir = "/home/xuan/RGB_SFM_V2/visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 所有 stage 對比圖
    num_stages = len(stage_features)
    fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
    
    if num_stages == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('PVT 各 Stage 特徵圖對比', fontsize=16, fontweight='bold')
    
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        C, H, W = feat.shape
        
        # 上排：所有通道平均值
        mean_feat = feat.mean(dim=0).cpu().numpy()
        im1 = axes[0, i].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Stage {info["stage"]}\n(C={C}, H={H}, W={W})')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 下排：通道統計
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        axes[1, i].plot(channel_means, 'b-', alpha=0.7, linewidth=2)
        axes[1, i].set_title(f'通道平均值\n(Heads={info["heads"]}, SR={info["sr_ratio"]})')
        axes[1, i].set_xlabel('通道索引')
        axes[1, i].set_ylabel('平均值')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存對比圖
    comparison_path = os.path.join(output_dir, "pvt_stages_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ 對比圖已保存至: {comparison_path}")
    plt.show()
    
    # 2. 每個 stage 的詳細圖
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        C, H, W = feat.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Stage {info["stage"]} 詳細特徵圖 (C={C}, H={H}, W={W})', 
                     fontsize=14, fontweight='bold')
        
        # 所有通道平均值
        mean_feat = feat.mean(dim=0).cpu().numpy()
        im1 = axes[0, 0].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('所有通道平均值')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 前16個通道的網格
        num_channels_to_show = min(16, C)
        channel_grid = feat[:num_channels_to_show].cpu().numpy()
        grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
        channel_heatmap = np.zeros((grid_size * H, grid_size * W))
        
        for j in range(num_channels_to_show):
            row = j // grid_size
            col = j % grid_size
            channel_heatmap[row*H:(row+1)*H, col*W:(col+1)*W] = channel_grid[j]
        
        im2 = axes[0, 1].imshow(channel_heatmap, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'前 {num_channels_to_show} 個通道')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 通道統計
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        channel_stds = feat.std(dim=(1, 2)).cpu().numpy()
        
        axes[1, 0].plot(channel_means, 'b-', label='平均值', alpha=0.7, linewidth=2)
        axes[1, 0].plot(channel_stds, 'r-', label='標準差', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('通道統計')
        axes[1, 0].set_xlabel('通道索引')
        axes[1, 0].set_ylabel('數值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 特徵分布
        all_values = feat.flatten().cpu().numpy()
        axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('特徵值分布')
        axes[1, 1].set_xlabel('特徵值')
        axes[1, 1].set_ylabel('頻率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存詳細圖
        detail_path = os.path.join(output_dir, f"stage_{info['stage']}_detail.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stage {info['stage']} 詳細圖已保存至: {detail_path}")
        plt.show()
    
    print(f"\n🎉 所有可視化完成！")
    print(f"📁 輸出目錄: {output_dir}")
    print("📊 生成的檔案:")
    print("  - pvt_stages_comparison.png (所有 stage 對比)")
    print("  - stage_X_detail.png (各 stage 詳細特徵圖)")

if __name__ == "__main__":
    load_model_and_visualize()
