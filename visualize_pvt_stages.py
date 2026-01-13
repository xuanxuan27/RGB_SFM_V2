#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT 各 Stage 特徵圖可視化腳本
載入訓練好的 PVT 模型，提取每個 stage 的特徵並可視化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import seaborn as sns
from torchvision import transforms
from PIL import Image

# # 添加專案根目錄到 Python 路徑
# project_root = Path(__file__).parent
# sys.path.append(str(project_root))

# 導入模型和數據集
from models.PVT import PVT, PVTv2
from dataloader.Colored_MNIST import Colored_MNIST

def load_trained_model(model_path, config):
    """
    載入訓練好的 PVT 模型
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

def load_sample_data(data_root, num_samples=4):
    """
    載入 Colored_MNIST 樣本數據
    """
    print("正在載入 Colored_MNIST 數據...")
    
    # 定義數據變換
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # 載入數據集
    dataset = Colored_MNIST(root=data_root, train=False, transform=transform)
    
    # 獲取樣本
    samples = []
    labels = []
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        samples.append(img)
        labels.append(label)
    
    # 堆疊成批次
    batch = torch.stack(samples)
    labels = torch.stack(labels)
    
    print(f"載入了 {len(samples)} 個樣本")
    return batch, labels

def extract_stage_features(model, input_batch):
    """
    提取 PVT 每個 stage 的特徵圖
    """
    print("正在提取各 stage 特徵...")
    
    with torch.no_grad():
        # 獲取 backbone
        backbone = model.backbone
        
        # 存儲各 stage 特徵
        stage_features = []
        stage_info = []
        
        B = input_batch.shape[0]
        x = input_batch
        
        # 追蹤上一層的 (C, H, W)
        prev_C, prev_H, prev_W = None, None, None
        seq = None
        
        for i in range(4):
            if i == 0:
                x_img = x  # (B, 3, H, W)
            else:
                # 把上一個 stage 的序列還原成 feature map
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
            stage_features.append(feat)
            stage_info.append({
                'stage': i + 1,
                'channels': backbone.dims[i],
                'height': H,
                'width': W,
                'heads': backbone.heads[i] if hasattr(backbone, 'heads') else 'N/A',
                'sr_ratio': backbone.srs[i] if hasattr(backbone, 'srs') else 'N/A'
            })
            
            # 記錄給下一個 stage 使用
            prev_C, prev_H, prev_W = backbone.dims[i], H, W
            last_seq = seq_out
        
        return stage_features, stage_info

def visualize_single_stage(features, stage_info, sample_idx=0, save_path=None):
    """
    可視化單個 stage 的特徵圖
    """
    feat = features[sample_idx]  # (C, H, W)
    info = stage_info
    
    C, H, W = feat.shape
    
    # 創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Stage {info["stage"]} 特徵圖 (C={C}, H={H}, W={W}, Heads={info["heads"]}, SR={info["sr_ratio"]})', 
                 fontsize=16, fontweight='bold')
    
    # 1. 所有通道的平均值
    mean_feat = feat.mean(dim=0).cpu().numpy()
    im1 = axes[0, 0].imshow(mean_feat, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('所有通道平均值')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. 前幾個通道的熱力圖
    num_channels_to_show = min(16, C)
    channel_grid = feat[:num_channels_to_show].cpu().numpy()
    
    # 重新排列為網格
    grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
    channel_heatmap = np.zeros((grid_size * H, grid_size * W))
    
    for i in range(num_channels_to_show):
        row = i // grid_size
        col = i % grid_size
        channel_heatmap[row*H:(row+1)*H, col*W:(col+1)*W] = channel_grid[i]
    
    im2 = axes[0, 1].imshow(channel_heatmap, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'前 {num_channels_to_show} 個通道')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. 通道統計
    channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
    channel_stds = feat.std(dim=(1, 2)).cpu().numpy()
    
    axes[1, 0].plot(channel_means, 'b-', label='平均值', alpha=0.7)
    axes[1, 0].plot(channel_stds, 'r-', label='標準差', alpha=0.7)
    axes[1, 0].set_title('通道統計')
    axes[1, 0].set_xlabel('通道索引')
    axes[1, 0].set_ylabel('數值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 特徵分布直方圖
    all_values = feat.flatten().cpu().numpy()
    axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('特徵值分布')
    axes[1, 1].set_xlabel('特徵值')
    axes[1, 1].set_ylabel('頻率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stage {info['stage']} 特徵圖已保存至: {save_path}")
    
    plt.show()

def visualize_all_stages(stage_features, stage_info, sample_idx=0, save_path=None):
    """
    可視化所有 stage 的特徵圖對比
    """
    print("正在生成所有 stage 對比圖...")
    
    num_stages = len(stage_features)
    fig, axes = plt.subplots(2, num_stages, figsize=(5*num_stages, 10))
    
    if num_stages == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('PVT 各 Stage 特徵圖對比', fontsize=16, fontweight='bold')
    
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        feat_sample = feat[sample_idx]  # (C, H, W)
        C, H, W = feat_sample.shape
        
        # 上排：所有通道平均值
        mean_feat = feat_sample.mean(dim=0).cpu().numpy()
        im1 = axes[0, i].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Stage {info["stage"]}\n(C={C}, H={H}, W={W})')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 下排：通道統計
        channel_means = feat_sample.mean(dim=(1, 2)).cpu().numpy()
        axes[1, i].plot(channel_means, 'b-', alpha=0.7)
        axes[1, i].set_title(f'通道平均值\n(Heads={info["heads"]}, SR={info["sr_ratio"]})')
        axes[1, i].set_xlabel('通道索引')
        axes[1, i].set_ylabel('平均值')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"所有 stage 對比圖已保存至: {save_path}")
    
    plt.show()

def visualize_patch_evolution(stage_features, stage_info, sample_idx=0, patch_pos=(0, 0), save_path=None):
    """
    可視化特定 patch 位置在各 stage 的演變
    """
    print(f"正在生成 patch ({patch_pos[0]}, {patch_pos[1]}) 的演變圖...")
    
    num_stages = len(stage_features)
    fig, axes = plt.subplots(1, num_stages, figsize=(4*num_stages, 4))
    
    if num_stages == 1:
        axes = [axes]
    
    fig.suptitle(f'Patch ({patch_pos[0]}, {patch_pos[1]}) 在各 Stage 的演變', 
                 fontsize=14, fontweight='bold')
    
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        feat_sample = feat[sample_idx]  # (C, H, W)
        C, H, W = feat_sample.shape
        
        # 確保 patch 位置在範圍內
        h_idx = min(patch_pos[0], H-1)
        w_idx = min(patch_pos[1], W-1)
        
        # 提取該 patch 的所有通道特徵
        patch_features = feat_sample[:, h_idx, w_idx].cpu().numpy()
        
        # 繪製通道特徵
        axes[i].bar(range(len(patch_features)), patch_features, alpha=0.7)
        axes[i].set_title(f'Stage {info["stage"]}\n({h_idx}, {w_idx})')
        axes[i].set_xlabel('通道索引')
        axes[i].set_ylabel('特徵值')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Patch 演變圖已保存至: {save_path}")
    
    plt.show()

def main():
    """主函數"""
    print("PVT 各 Stage 特徵圖可視化工具")
    print("=" * 50)
    
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
    
    # 路徑設置
    model_path = "/home/xuan/RGB_SFM_V2/runs/train/exp64/PVT_best.pth"
    data_root = "/home/xuan/RGB_SFM_V2/data"  # 根據您的數據路徑調整
    output_dir = "/home/xuan/RGB_SFM_V2/visualization_output"
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型
    model = load_trained_model(model_path, model_config)
    
    # 載入樣本數據
    try:
        batch, labels = load_sample_data(data_root, num_samples=4)
    except Exception as e:
        print(f"載入數據時發生錯誤: {e}")
        print("使用隨機數據進行演示...")
        batch = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 30, (4,))
    
    # 提取各 stage 特徵
    stage_features, stage_info = extract_stage_features(model, batch)
    
    # 打印 stage 信息
    print("\n=== 各 Stage 信息 ===")
    for info in stage_info:
        print(f"Stage {info['stage']}: C={info['channels']}, H={info['height']}, W={info['width']}, "
              f"Heads={info['heads']}, SR={info['sr_ratio']}")
    
    # 可視化每個 stage
    print("\n=== 生成各 Stage 特徵圖 ===")
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        save_path = os.path.join(output_dir, f"stage_{info['stage']}_features.png")
        visualize_single_stage(feat, info, sample_idx=0, save_path=save_path)
    
    # 可視化所有 stage 對比
    print("\n=== 生成所有 Stage 對比圖 ===")
    all_stages_path = os.path.join(output_dir, "all_stages_comparison.png")
    visualize_all_stages(stage_features, stage_info, sample_idx=0, save_path=all_stages_path)
    
    # 可視化 patch 演變
    print("\n=== 生成 Patch 演變圖 ===")
    patch_evolution_path = os.path.join(output_dir, "patch_evolution.png")
    visualize_patch_evolution(stage_features, stage_info, sample_idx=0, 
                             patch_pos=(0, 0), save_path=patch_evolution_path)
    
    print("\n視覺化完成！")
    print(f"輸出檔案保存在: {output_dir}")
    print("包含以下檔案:")
    print("- stage_X_features.png (各 stage 詳細特徵圖)")
    print("- all_stages_comparison.png (所有 stage 對比圖)")
    print("- patch_evolution.png (patch 演變圖)")

if __name__ == "__main__":
    main()
