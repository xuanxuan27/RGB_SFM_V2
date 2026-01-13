#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT Patch Merging 分析腳本
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch

def create_patch_merging_diagram():
    """創建 PVT Patch Merging 分析圖"""
    
    # 創建圖形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PVT Patch Merging: Kernel Size & Stride Analysis', fontsize=16, fontweight='bold')
    
    # 1. 各 Stage 的參數配置
    ax1 = axes[0, 0]
    stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    kernel_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    paddings = [3, 1, 1, 1]
    input_sizes = ['224×224', '56×56', '28×28', '14×14']
    output_sizes = ['56×56', '28×28', '14×14', '7×7']
    
    x = np.arange(len(stages))
    width = 0.25
    
    bars1 = ax1.bar(x - width, kernel_sizes, width, label='Kernel Size', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x, strides, width, label='Stride', color='lightcoral', alpha=0.8)
    bars3 = ax1.bar(x + width, paddings, width, label='Padding', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Size')
    ax1.set_title('Patch Embedding Parameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 空間尺寸變化
    ax2 = axes[0, 1]
    ax2.plot(stages, [224, 56, 28, 14, 7], 'o-', linewidth=3, markersize=8, color='red', label='Spatial Size')
    ax2.set_title('Spatial Dimension Reduction')
    ax2.set_ylabel('Size (H×W)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加尺寸標籤
    sizes = [224, 56, 28, 14, 7]
    for i, size in enumerate(sizes):
        ax2.text(i, size + 5, f'{size}×{size}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 重疊度分析
    ax3 = axes[1, 0]
    overlap_ratios = []
    for i in range(4):
        # 重疊度 = (kernel_size - stride) / kernel_size
        overlap = (kernel_sizes[i] - strides[i]) / kernel_sizes[i]
        overlap_ratios.append(overlap)
    
    bars = ax3.bar(stages, overlap_ratios, color='orange', alpha=0.8)
    ax3.set_title('Patch Overlap Ratio')
    ax3.set_ylabel('Overlap Ratio')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar, ratio in zip(bars, overlap_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 計算複雜度分析
    ax4 = axes[1, 1]
    # 計算每個 stage 的參數數量（近似）
    param_counts = []
    for i in range(4):
        # Conv2d 參數 = in_channels * out_channels * kernel_size^2
        if i == 0:
            in_ch = 3
        else:
            in_ch = [64, 128, 320][i-1]
        out_ch = [64, 128, 320, 512][i]
        params = in_ch * out_ch * kernel_sizes[i] ** 2
        param_counts.append(params / 1000)  # 轉換為 K
    
    bars = ax4.bar(stages, param_counts, color='purple', alpha=0.8)
    ax4.set_title('Patch Embedding Parameters (K)')
    ax4.set_ylabel('Parameter Count (×1000)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count:.1f}K', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/xuan/RGB_SFM_V2/visualization_output/pvt_patch_merging_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ PVT Patch Merging 分析圖已保存")

def create_detailed_patch_visualization():
    """創建詳細的 patch 可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PVT Patch Merging: Detailed Visualization', fontsize=16, fontweight='bold')
    
    # 1. Stage 1: 7x7 kernel, stride=4
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 16)
    ax1.set_title('Stage 1: Kernel=7×7, Stride=4, Padding=3')
    ax1.set_aspect('equal')
    
    # 繪製輸入網格 (224x224 -> 16x16 for visualization)
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            rect = Rectangle((i, j), 7, 7, linewidth=2, edgecolor='blue', 
                           facecolor='lightblue', alpha=0.3)
            ax1.add_patch(rect)
    
    # 繪製輸出位置
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            circle = patches.Circle((i+3.5, j+3.5), 0.5, color='red', alpha=0.8)
            ax1.add_patch(circle)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Input: 224×224 → Output: 56×56')
    
    # 2. Stage 2: 3x3 kernel, stride=2
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 12)
    ax2.set_title('Stage 2: Kernel=3×3, Stride=2, Padding=1')
    ax2.set_aspect('equal')
    
    # 繪製輸入網格
    for i in range(0, 12, 2):
        for j in range(0, 12, 2):
            rect = Rectangle((i, j), 3, 3, linewidth=2, edgecolor='green', 
                           facecolor='lightgreen', alpha=0.3)
            ax2.add_patch(rect)
    
    # 繪製輸出位置
    for i in range(0, 12, 2):
        for j in range(0, 12, 2):
            circle = patches.Circle((i+1.5, j+1.5), 0.3, color='red', alpha=0.8)
            ax2.add_patch(circle)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Input: 56×56 → Output: 28×28')
    
    # 3. 重疊度比較
    ax3 = axes[1, 0]
    stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    kernel_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    
    overlap_pixels = []
    for i in range(4):
        overlap = kernel_sizes[i] - strides[i]
        overlap_pixels.append(overlap)
    
    bars = ax3.bar(stages, overlap_pixels, color=['skyblue', 'lightgreen', 'lightgreen', 'lightgreen'], alpha=0.8)
    ax3.set_title('Overlap Pixels per Dimension')
    ax3.set_ylabel('Overlap Pixels')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, overlap in zip(bars, overlap_pixels):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{overlap}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 設計原理說明
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = [
        "PVT Patch Merging 設計原理:",
        "",
        "🎯 Stage 1 (7×7, stride=4):",
        "   • 大 kernel 捕獲更多上下文信息",
        "   • 高重疊度 (3 pixels) 保持細節",
        "   • 224×224 → 56×56 (4倍下採樣)",
        "",
        "🎯 Stage 2-4 (3×3, stride=2):",
        "   • 小 kernel 專注局部特徵",
        "   • 低重疊度 (1 pixel) 提高效率",
        "   • 2倍下採樣保持金字塔結構",
        "",
        "⚡ 設計優勢:",
        "   • 漸進式特徵提取",
        "   • 平衡計算效率與特徵質量",
        "   • 保持空間層次結構"
    ]
    
    for i, line in enumerate(explanation):
        ax4.text(0.05, 0.95 - i*0.05, line, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7) if i == 0 else None,
                fontweight='bold' if i == 0 else 'normal')
    
    plt.tight_layout()
    plt.savefig('/home/xuan/RGB_SFM_V2/visualization_output/pvt_patch_detailed_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ PVT Patch 詳細可視化已保存")

def main():
    """主函數"""
    print("PVT Patch Merging 分析")
    print("=" * 50)
    
    # 創建分析圖
    create_patch_merging_diagram()
    
    # 創建詳細可視化
    create_detailed_patch_visualization()
    
    print("\n🎉 分析完成！")
    print("📁 輸出檔案:")
    print("  - pvt_patch_merging_analysis.png (參數分析)")
    print("  - pvt_patch_detailed_visualization.png (詳細可視化)")

if __name__ == "__main__":
    main()

