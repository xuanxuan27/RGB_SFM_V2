#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB_SFMCNN_V2 模型空間合併分析工具
分析每一層的 SFM filter 累積效果和空間合併機制
"""

import torch
import math
from models.RGB_SFMCNN_V2 import RGB_SFMCNN_V2

def analyze_sfm_filters(arch_config):
    """
    分析 RGB_SFMCNN_V2 模型中每一層的空間合併效果
    
    Args:
        arch_config: 模型架構配置字典
    """
    print("=" * 80)
    print("RGB_SFMCNN_V2 空間合併分析")
    print("=" * 80)
    
    # 提取配置參數
    mode = arch_config['args']['mode']
    SFM_filters = arch_config['args']['SFM_filters']
    channels = arch_config['args']['channels']
    SFM_methods = arch_config['args']['SFM_methods']
    strides = arch_config['args']['strides']
    
    print(f"模型模式: {mode}")
    print(f"SFM 方法: {SFM_methods}")
    print()
    
    # 分析 RGB 分支
    if mode in ['rgb', 'both']:
        print("🔴 RGB 分支分析:")
        print("-" * 50)
        analyze_branch("RGB", SFM_filters[0], channels[0], strides[0])
        print()
    
    # 分析 Gray 分支
    if mode in ['gray', 'both']:
        print("⚫ Gray 分支分析:")
        print("-" * 50)
        analyze_branch("Gray", SFM_filters[1], channels[1], strides[1])
        print()

def analyze_branch(branch_name, sfm_filters, channels, strides):
    """
    分析單個分支的空間合併效果
    
    Args:
        branch_name: 分支名稱 (RGB 或 Gray)
        sfm_filters: 該分支的 SFM filter 配置
        channels: 該分支的通道配置
        strides: 該分支的步長配置
    """
    print(f"層數: {len(sfm_filters)}")
    print(f"SFM Filters: {sfm_filters}")
    print(f"Channels: {channels}")
    print(f"Strides: {strides}")
    print()
    
    # 計算累積的空間合併效果
    cumulative_filter = (1, 1)  # 初始值
    cumulative_stride = 1       # 初始值
    
    print("各層詳細分析:")
    print("層次 | SFM Filter | 累積 Filter | 通道數 | 步長 | 累積步長 | 空間縮放")
    print("-" * 80)
    
    for i in range(len(sfm_filters)):
        # 當前層的 SFM filter
        current_filter = sfm_filters[i]
        
        # 計算累積 filter (相乘)
        cumulative_filter = (cumulative_filter[0] * current_filter[0], 
                           cumulative_filter[1] * current_filter[1])
        
        # 計算累積步長
        cumulative_stride *= strides[i]
        
        # 通道數
        channel_count = channels[i][0] * channels[i][1]
        
        # 空間縮放比例
        spatial_scale = cumulative_filter[0] * cumulative_filter[1] * cumulative_stride
        
        print(f" {i:2d}  | {str(current_filter):12s} | {str(cumulative_filter):12s} | "
              f"{channel_count:6d} | {strides[i]:4d} | {cumulative_stride:8d} | {spatial_scale:8d}")
    
    print()
    print(f"最終累積效果:")
    print(f"  - 累積 SFM Filter: {cumulative_filter}")
    print(f"  - 累積步長: {cumulative_stride}")
    print(f"  - 總空間縮放: {cumulative_filter[0] * cumulative_filter[1] * cumulative_stride}")

def explain_sfm_mechanism():
    """
    詳細解釋 SFM (Spatial Feature Merging) 空間合併機制
    """
    print("=" * 80)
    print("SFM 空間合併機制詳細說明")
    print("=" * 80)
    
    print("""
🔍 SFM (Spatial Feature Merging) 工作原理:

1. 基本概念:
   - SFM 是一種空間特徵合併技術
   - 將相鄰的空間區域合併成單一特徵
   - 通過 filter 大小控制合併範圍

2. 數學原理:
   - 輸入: (batch, channels, height, width)
   - Filter: (filter_h, filter_w) 例如 (2, 2)
   - 輸出尺寸計算:
     output_height = floor((height - (filter_h - 1) - 1) / filter_h + 1)
     output_width = floor((width - (filter_w - 1) - 1) / filter_w + 1)

3. 合併方法:
   a) alpha_mean: 加權平均合併
      - 使用線性遞減的權重 (alpha_min 到 alpha_max)
      - 對 filter 區域內的值進行加權平均
   
   b) max: 最大值合併
      - 取 filter 區域內的最大值

4. 累積效果:
   - 每層的 SFM filter 會累積相乘
   - 例如: (1,1) → (2,2) → (1,3) → (1,1)
   - 累積效果: (1,1) → (2,2) → (2,6) → (2,6)
   - 這意味著最終的空間感受野會擴大

5. 與步長的關係:
   - 步長控制卷積的採樣間隔
   - SFM filter 控制空間合併範圍
   - 兩者共同決定最終的空間縮放效果
""")

def demonstrate_sfm_effect():
    """
    演示 SFM 的實際效果
    """
    print("=" * 80)
    print("SFM 效果演示")
    print("=" * 80)
    
    # 模擬輸入
    batch_size, channels, height, width = 1, 1, 8, 8
    input_tensor = torch.arange(height * width).float().reshape(1, 1, height, width)
    
    print(f"原始輸入尺寸: {input_tensor.shape}")
    print("原始數據:")
    print(input_tensor[0, 0].numpy())
    print()
    
    # 模擬 SFM 效果
    from models.RGB_SFMCNN_V2 import SFM
    
    # 測試不同的 filter 大小
    filters_to_test = [(1, 1), (2, 2), (1, 3)]
    
    for filter_size in filters_to_test:
        print(f"SFM Filter: {filter_size}")
        sfm = SFM(filter=filter_size, method="alpha_mean")
        
        with torch.no_grad():
            output = sfm(input_tensor)
        
        print(f"輸出尺寸: {output.shape}")
        print("輸出數據:")
        print(output[0, 0].numpy())
        print("-" * 40)

if __name__ == "__main__":
    # 使用範例配置
    example_config = {
        "name": 'RGB_SFMCNN_V2',
        "args": {
            "mode": "both",
            "SFM_filters": [[(1, 1), (2, 2), (1, 3), (1, 1)],
                           [(2, 2), (1, 3), (1, 1)]],
            "channels": [[(10, 10), (15, 15), (25, 25), (35, 35)],
                        [(7, 10), (15, 15), (35, 35)]],
            "strides": [[1, 4, 1, 1],
                       [4, 1, 1]],
            "SFM_methods": [["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"],
                           ["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"]]
        }
    }
    
    # 執行分析
    analyze_sfm_filters(example_config)
    print("\n")
    explain_sfm_mechanism()
    print("\n")
    demonstrate_sfm_effect()
