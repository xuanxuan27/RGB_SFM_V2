#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多樣本 PVT 可視化腳本
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from torchvision import transforms

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import model and dataset
from models.PVT import PVT
from dataloader.Colored_MNIST import Colored_MNIST

def load_model_and_data():
    """Load trained model and Colored MNIST data"""
    
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
    
    # Load Colored MNIST data
    print("Loading Colored MNIST data...")
    data_root = "/home/xuan/RGB_SFM_V2/data"
    
    # Define transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    try:
        # Load dataset
        dataset = Colored_MNIST(root=data_root, train=False, transform=transform)
        
        # Get more samples
        samples = []
        labels = []
        for i in range(min(12, len(dataset))):  # 獲取 12 個樣本
            img, label = dataset[i]
            samples.append(img)
            # Convert numpy array to tensor if needed
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).float()
            labels.append(label)
        
        # Stack into batch
        batch = torch.stack(samples)
        labels = torch.stack(labels)
        
        print(f"✓ Loaded {len(samples)} samples from Colored MNIST")
        
    except Exception as e:
        print(f"⚠ Error loading Colored MNIST: {e}")
        print("Using random data for demonstration...")
        batch = torch.randn(12, 3, 224, 224)
        labels = torch.randint(0, 30, (12,))
    
    return model, batch, labels

def extract_stage_features(model, input_batch):
    """Extract features from each PVT stage"""
    print("Extracting stage features...")
    
    with torch.no_grad():
        backbone = model.backbone
        stage_features = []
        stage_info = []
        
        B = input_batch.shape[0]
        x = input_batch
        
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
            stage_features.append(feat)
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
    
    return stage_features, stage_info

def create_multi_sample_comparison(stage_features, stage_info, num_samples=8):
    """Create multi-sample comparison visualization"""
    
    # Create output directory
    output_dir = "/home/xuan/RGB_SFM_V2/visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print stage information
    print("\n=== Stage Information ===")
    for info in stage_info:
        print(f"Stage {info['stage']}: C={info['channels']}, H={info['height']}, W={info['width']}, "
              f"Heads={info['heads']}, SR={info['sr_ratio']}")
    
    # 1. Multi-sample comparison for each stage
    for stage_idx, (feat, info) in enumerate(zip(stage_features, stage_info)):
        C, H, W = feat.shape[1:]
        
        # Create grid for multiple samples
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Stage {info["stage"]} - Multiple Samples (C={C}, H={H}, W={W})', 
                     fontsize=16, fontweight='bold')
        
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for sample_idx in range(num_samples):
            if sample_idx < len(feat):
                # Get average feature map for this sample
                mean_feat = feat[sample_idx].mean(dim=0).cpu().numpy()
                
                im = axes[sample_idx].imshow(mean_feat, cmap='viridis', aspect='auto')
                axes[sample_idx].set_title(f'Sample {sample_idx+1}')
                axes[sample_idx].axis('off')
                
                # Add colorbar for the first subplot
                if sample_idx == 0:
                    plt.colorbar(im, ax=axes[sample_idx], fraction=0.046, pad=0.04)
            else:
                axes[sample_idx].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save multi-sample plot
        multi_sample_path = os.path.join(output_dir, f"stage_{info['stage']}_multi_samples.png")
        plt.savefig(multi_sample_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stage {info['stage']} multi-sample plot saved to: {multi_sample_path}")
        plt.show()
    
    # 2. Cross-stage comparison for specific samples
    selected_samples = [0, 1, 2, 3]  # 選擇 4 個樣本進行跨 stage 比較
    
    for sample_idx in selected_samples:
        if sample_idx < len(stage_features[0]):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Sample {sample_idx+1} - Cross Stage Comparison', 
                         fontsize=16, fontweight='bold')
            
            for stage_idx, (feat, info) in enumerate(zip(stage_features, stage_info)):
                feat_sample = feat[sample_idx]  # (C, H, W)
                C, H, W = feat_sample.shape
                
                # Top row: Average feature maps
                mean_feat = feat_sample.mean(dim=0).cpu().numpy()
                im1 = axes[0, stage_idx].imshow(mean_feat, cmap='viridis', aspect='auto')
                axes[0, stage_idx].set_title(f'Stage {info["stage"]}\n(C={C}, H={H}, W={W})')
                axes[0, stage_idx].axis('off')
                plt.colorbar(im1, ax=axes[0, stage_idx], fraction=0.046, pad=0.04)
                
                # Bottom row: Channel statistics
                channel_means = feat_sample.mean(dim=(1, 2)).cpu().numpy()
                axes[1, stage_idx].plot(channel_means, 'b-', alpha=0.7, linewidth=2)
                axes[1, stage_idx].set_title(f'Channel Means\n(Heads={info["heads"]})')
                axes[1, stage_idx].set_xlabel('Channel Index')
                axes[1, stage_idx].set_ylabel('Mean Value')
                axes[1, stage_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save cross-stage comparison
            cross_stage_path = os.path.join(output_dir, f"cross_stage_comparison_sample_{sample_idx+1}.png")
            plt.savefig(cross_stage_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cross-stage comparison for sample {sample_idx+1} saved to: {cross_stage_path}")
            plt.show()
    
    print(f"\n🎉 Multi-sample visualizations completed!")
    print(f"📁 Output directory: {output_dir}")
    print("📊 Generated files:")
    print("  - stage_X_multi_samples.png (Multi-sample comparison for each stage)")
    print("  - cross_stage_comparison_sample_X.png (Cross-stage comparison for specific samples)")

def main():
    """Main function"""
    print("Multi-Sample PVT Visualization")
    print("=" * 50)
    
    # Load model and data
    model, batch, labels = load_model_and_data()
    
    # Extract stage features
    stage_features, stage_info = extract_stage_features(model, batch)
    
    # Create multi-sample visualizations
    create_multi_sample_comparison(stage_features, stage_info, num_samples=8)

if __name__ == "__main__":
    main()
