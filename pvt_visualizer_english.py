#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT Stage Feature Visualization Script (English Version)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import model
from models.PVT import PVT

def load_model_and_visualize():
    """Load model and perform visualization"""
    
    # Model configuration
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
    
    # Model path
    model_path = "/home/xuan/RGB_SFM_V2/runs/train/exp64/PVT_best.pth"
    
    print("Loading PVT model...")
    
    # Create model
    model = PVT(**model_config)
    
    # Try to load weights
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
            print("✓ Model weights loaded successfully!")
        except Exception as e:
            print(f"⚠ Error loading weights: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"⚠ Model file not found: {model_path}")
        print("Using randomly initialized weights")
    
    model.eval()
    
    # Create test input
    print("Creating test input...")
    test_input = torch.randn(1, 3, 224, 224)
    
    # Extract features
    print("Extracting stage features...")
    with torch.no_grad():
        backbone = model.backbone
        stage_features = []
        stage_info = []
        
        B = test_input.shape[0]
        x = test_input
        
        # Track previous layer (C, H, W)
        prev_C, prev_H, prev_W = None, None, None
        seq = None
        
        for i in range(4):
            if i == 0:
                x_img = x
            else:
                x_img = seq.transpose(1, 2).reshape(B, prev_C, prev_H, prev_W)
            
            # Patch Embedding
            seq, H, W = backbone.patch_embeds[i](x_img)
            
            # Through multiple Blocks
            for blk in backbone.stages[i]:
                seq = blk(seq, H, W)
            
            # Normalize and convert to feature map
            seq_out = backbone.norms[i](seq)
            feat = seq_out.transpose(1, 2).reshape(B, backbone.dims[i], H, W)
            
            # Store features and info
            stage_features.append(feat[0])  # Take first sample
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
    
    # Print information
    print("\n=== Stage Information ===")
    for info in stage_info:
        print(f"Stage {info['stage']}: C={info['channels']}, H={info['height']}, W={info['width']}, "
              f"Heads={info['heads']}, SR={info['sr_ratio']}")
    
    # Create visualization
    print("\nGenerating visualization charts...")
    
    # Create output directory
    output_dir = "/home/xuan/RGB_SFM_V2/visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. All stages comparison
    num_stages = len(stage_features)
    fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
    
    if num_stages == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('PVT Stage Feature Maps Comparison', fontsize=16, fontweight='bold')
    
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        C, H, W = feat.shape
        
        # Top row: Average of all channels
        mean_feat = feat.mean(dim=0).cpu().numpy()
        im1 = axes[0, i].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Stage {info["stage"]}\n(C={C}, H={H}, W={W})')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Bottom row: Channel statistics
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        axes[1, i].plot(channel_means, 'b-', alpha=0.7, linewidth=2)
        axes[1, i].set_title(f'Channel Means\n(Heads={info["heads"]}, SR={info["sr_ratio"]})')
        axes[1, i].set_xlabel('Channel Index')
        axes[1, i].set_ylabel('Mean Value')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(output_dir, "pvt_stages_comparison_en.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {comparison_path}")
    plt.show()
    
    # 2. Detailed plots for each stage
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        C, H, W = feat.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Stage {info["stage"]} Detailed Features (C={C}, H={H}, W={W})', 
                     fontsize=14, fontweight='bold')
        
        # Average of all channels
        mean_feat = feat.mean(dim=0).cpu().numpy()
        im1 = axes[0, 0].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Average of All Channels')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Grid of first 16 channels
        num_channels_to_show = min(16, C)
        channel_grid = feat[:num_channels_to_show].cpu().numpy()
        grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
        channel_heatmap = np.zeros((grid_size * H, grid_size * W))
        
        for j in range(num_channels_to_show):
            row = j // grid_size
            col = j % grid_size
            channel_heatmap[row*H:(row+1)*H, col*W:(col+1)*W] = channel_grid[j]
        
        im2 = axes[0, 1].imshow(channel_heatmap, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'First {num_channels_to_show} Channels')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Channel statistics
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        channel_stds = feat.std(dim=(1, 2)).cpu().numpy()
        
        axes[1, 0].plot(channel_means, 'b-', label='Mean', alpha=0.7, linewidth=2)
        axes[1, 0].plot(channel_stds, 'r-', label='Std', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('Channel Statistics')
        axes[1, 0].set_xlabel('Channel Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature distribution
        all_values = feat.flatten().cpu().numpy()
        axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Feature Value Distribution')
        axes[1, 1].set_xlabel('Feature Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed plot
        detail_path = os.path.join(output_dir, f"stage_{info['stage']}_detail_en.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stage {info['stage']} detailed plot saved to: {detail_path}")
        plt.show()
    
    print(f"\n🎉 All visualizations completed!")
    print(f"📁 Output directory: {output_dir}")
    print("📊 Generated files:")
    print("  - pvt_stages_comparison_en.png (All stages comparison)")
    print("  - stage_X_detail_en.png (Detailed feature maps for each stage)")

if __name__ == "__main__":
    load_model_and_visualize()
