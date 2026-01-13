#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT Stage Visualization for Multiple Datasets
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from torchvision import transforms
# 讀取 exp72 的 config
import sys
sys.path.append('/home/xuan/RGB_SFM_V2/runs/train/exp64')
from config import *
import models
from dataloader import get_dataloader

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import model and dataset
from models.PVT import PVT

def _clean_state_dict_keys(state_dict: dict) -> dict:
    """移除 DataParallel 等帶有 'module.' 前綴的權重鍵名。"""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    return cleaned

def load_model_and_data(model_path=None):
    """從 config 讀取設定，建立模型並載入權重，使用 get_dataloader 取得資料（與 train.py 對齊）"""

    # 建立模型
    model_name = config['model']['name']
    model_args = dict(config['model']['args'])
    model_cls = getattr(getattr(models, model_name), model_name)
    model = model_cls(**model_args)

    # 載入權重（寫死使用 exp72/PVT_best.pth）
    if model_path is None:
        model_path = "/home/xuan/RGB_SFM_V2/runs/train/exp64/PVT_best.pth"
        print(f"Using model path: {model_path}")
    else:
        print(f"Using model path: {model_path}")
        
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Using 'model_state_dict' key")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("Using 'state_dict' key")
                elif 'model_weights' in checkpoint:
                    state_dict = checkpoint['model_weights']
                    print("Using 'model_weights' key")
                else:
                    state_dict = checkpoint
                    print("Using checkpoint directly as state_dict")
            else:
                state_dict = checkpoint
                print("Checkpoint is not a dict, using directly")
            
            state_dict = _clean_state_dict_keys(state_dict)
            print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
            
            # 檢查模型參數數量
            model_params = sum(p.numel() for p in model.parameters())
            state_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"Model parameters: {model_params}, State dict parameters: {state_params}")
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # 只顯示前5個
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # 只顯示前5個
            
            print("✓ Model weights loaded successfully!")
        except Exception as e:
            print(f"⚠ Error loading weights: {e}\nUsing randomly initialized weights")
    else:
        print(f"⚠ Model file not found or unspecified: {model_path}\nUsing randomly initialized weights")

    model.eval()

    # 取得 dataloader（與 train.py 一致）
    # 檢查模型是否需要 ImageNet 標準化
    model_name = config['model']['name']
    use_imagenet_norm = model_name in ['PVT', 'VIT', 'RGB_SFM_VIT']  # 這些模型通常需要 ImageNet 標準化
    
    train_loader, test_loader = get_dataloader(
        dataset=config['dataset'],
        root=config['root'] + '/data/',
        batch_size=config['batch_size'],
        input_size=config['input_shape'],
        use_pretrained_vit=use_imagenet_norm
    )
    
    print(f"Using ImageNet normalization: {use_imagenet_norm}")

    # 取一個 batch
    try:
        batch, labels = next(iter(test_loader))
    except Exception:
        batch, labels = next(iter(train_loader))
    
    # 打印標籤信息用於調試
    print(f"\n=== Batch Information ===")
    print(f"Batch shape: {batch.shape}")
    print(f"Labels type: {type(labels)}")
    if isinstance(labels, torch.Tensor):
        print(f"Labels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        if labels.numel() > 0:
            print(f"First label sample: {labels[0]}")
            if labels.dim() > 1:
                print(f"First label shape: {labels[0].shape}")
    elif isinstance(labels, (list, tuple)):
        print(f"Labels length: {len(labels)}")
        if len(labels) > 0:
            print(f"First label type: {type(labels[0])}")
            print(f"First label: {labels[0]}")
    print("=" * 50)

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
        
        # Get model predictions
        predictions = model(input_batch)
        pred_probs = torch.softmax(predictions, dim=1)
        pred_classes = torch.argmax(predictions, dim=1)
    
    return stage_features, stage_info, pred_classes, pred_probs

def visualize_stages(stage_features, stage_info, sample_idx=0, labels=None, input_batch=None, pred_classes=None, pred_probs=None, dataset_type='CIFAR10'):
    """Visualize PVT stage features"""
    
    # Create output directory
    output_dir = "/home/xuan/RGB_SFM_V2/colored_mnist_visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get ground truth label for this sample
    gt_label = None
    gt_class = None
    
    if labels is not None:
        try:
            # 首先嘗試獲取單個樣本的標籤
            try:
                if isinstance(labels, torch.Tensor):
                    # 如果是 tensor，檢查維度
                    if labels.dim() == 1:
                        # 1D tensor，可能是批次標籤或 one-hot
                        if labels.shape[0] > sample_idx:
                            label_item = labels[sample_idx]
                        else:
                            label_item = labels[0]  # 如果索引超出範圍，使用第一個
                    elif labels.dim() == 2:
                        # 2D tensor，可能是批次 one-hot 編碼
                        if labels.shape[0] > sample_idx:
                            label_item = labels[sample_idx]
                        else:
                            label_item = labels[0]
                    else:
                        label_item = labels
                elif isinstance(labels, (list, tuple, np.ndarray)):
                    # 如果是列表或數組，嘗試索引
                    if len(labels) > sample_idx:
                        label_item = labels[sample_idx]
                    else:
                        label_item = labels[0] if len(labels) > 0 else None
                else:
                    label_item = labels
            except (IndexError, TypeError):
                # 如果無法索引，直接使用 labels
                label_item = labels
            
            if label_item is None:
                gt_label = None
                gt_class = None
            # 如果是 tensor
            elif isinstance(label_item, torch.Tensor):
                # 轉換為 numpy 以便處理
                if label_item.numel() == 0:
                    gt_label = None
                    gt_class = None
                else:
                    label_np = label_item.cpu().numpy()
                    if label_item.dim() == 0:  # scalar tensor
                        gt_class = int(label_item.item())
                        gt_label = gt_class
                    elif label_item.dim() == 1:  # one-hot or vector
                        gt_label = label_np
                        # 檢查是否是 one-hot (只有一個元素為 1，其他為 0)
                        unique_vals = np.unique(label_np)
                        if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals and np.sum(label_np == 1) == 1:
                            gt_class = int(np.argmax(label_np))
                        else:
                            # 如果不是標準 one-hot，嘗試找到最大值的索引
                            gt_class = int(np.argmax(label_np))
                    else:
                        gt_label = label_np
                        gt_class = None
            # 如果是 numpy array
            elif isinstance(label_item, np.ndarray):
                gt_label = label_item
                if label_item.ndim == 0:  # scalar
                    gt_class = int(label_item.item())
                elif label_item.ndim == 1:
                    # 檢查是否是 one-hot
                    unique_vals = np.unique(label_item)
                    if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals and np.sum(label_item == 1) == 1:
                        gt_class = int(np.argmax(label_item))
                    else:
                        gt_class = int(np.argmax(label_item))
                else:
                    gt_class = None
            # 如果是 list
            elif isinstance(label_item, (list, tuple)):
                # 轉換為 numpy array
                gt_label = np.array(label_item, dtype=np.float32)
                # 檢查是否是 one-hot
                if len(gt_label) > 0:
                    unique_vals = np.unique(gt_label)
                    if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals and np.sum(gt_label == 1) == 1:
                        gt_class = int(np.argmax(gt_label))
                    else:
                        gt_class = int(np.argmax(gt_label))
            # 如果是基本類型（int, float）
            elif isinstance(label_item, (int, float, np.integer, np.floating)):
                gt_class = int(label_item)
                gt_label = gt_class
            else:
                # 嘗試轉換為字符串顯示
                gt_label = str(label_item)
                gt_class = None
                
        except Exception as e:
            print(f"Warning: Error processing label for sample {sample_idx}: {e}")
            print(f"  Label type: {type(labels)}, Label item type: {type(label_item) if 'label_item' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            gt_label = None
            gt_class = None
    
    # Print stage information
    print("\n=== Stage Information ===")
    for info in stage_info:
        print(f"Stage {info['stage']}: C={info['channels']}, H={info['height']}, W={info['width']}, "
              f"Heads={info['heads']}, SR={info['sr_ratio']}")
    
    # Print ground truth and prediction
    if gt_label is not None:
        try:
            if isinstance(gt_label, (list, np.ndarray, tuple)):
                if len(gt_label) > 10:
                    # 如果標籤太長，只顯示前幾個和後幾個
                    print(f"Ground Truth Label (first 5, last 5): [{gt_label[:5]} ... {gt_label[-5:]}]")
                else:
                    print(f"Ground Truth Label: {gt_label}")
            else:
                print(f"Ground Truth Label: {gt_label}")
        except Exception as e:
            print(f"Ground Truth Label: {gt_label} (display error: {e})")
        
        if gt_class is not None:
            try:
                if isinstance(gt_class, (int, float, np.integer, np.floating)):
                    print(f"Ground Truth Class: {gt_class}")
                else:
                    print(f"Ground Truth Class: {gt_class} (type: {type(gt_class)})")
            except Exception as e:
                print(f"Ground Truth Class: {gt_class} (error: {e})")
    else:
        print("Ground Truth Label: None (not available)")
    
    if pred_classes is not None:
        pred_class = pred_classes[sample_idx].item()
        print(f"Predicted Class: {pred_class}")
        
        if pred_probs is not None:
            pred_prob = pred_probs[sample_idx, pred_class].item()
            print(f"Prediction Confidence: {pred_prob:.4f}")
            
            # Check if prediction is correct
            if gt_class is not None:
                try:
                    # 確保 gt_class 是整數
                    if isinstance(gt_class, (np.ndarray, list)):
                        gt_class = int(gt_class[0]) if len(gt_class) > 0 else None
                    elif isinstance(gt_class, (np.integer, np.floating)):
                        gt_class = int(gt_class)
                    
                    if gt_class is not None and isinstance(gt_class, (int, float)):
                        is_correct = (int(pred_class) == int(gt_class))
                        print(f"Prediction {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    else:
                        print("Cannot compare prediction: ground truth class is not a valid integer")
                except Exception as e:
                    print(f"Cannot compare prediction: {e}")
            else:
                print("Cannot compare prediction: ground truth class is not available")
    
    # 0. Show original input image
    if input_batch is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        img_sample = input_batch[sample_idx].clone()
        
        # 顯示圖像數值範圍用於調試
        print(f"Image value range: [{img_sample.min():.3f}, {img_sample.max():.3f}]")
        
        # 檢查是否需要反標準化（檢查數值範圍）
        if img_sample.min() < -0.5 or img_sample.max() > 1.5:
            # 需要反標準化
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_sample = img_sample * IMAGENET_STD + IMAGENET_MEAN
            img_sample = torch.clamp(img_sample, 0, 1)
            print(f"Applied ImageNet denormalization")
        else:
            # 已經在 [0,1] 範圍內，直接使用
            img_sample = torch.clamp(img_sample, 0, 1)
            print(f"Image already in [0,1] range, no denormalization needed")
        
        # Convert to numpy and transpose for matplotlib
        img_np = img_sample.permute(1, 2, 0).cpu().numpy()
        
        # 使用更好的插值方法顯示
        ax.imshow(img_np, interpolation='bilinear')
        title = f'Original Input Image (Sample {sample_idx + 1})'
        if gt_class is not None:
            try:
                if isinstance(gt_class, (np.ndarray, list)):
                    gt_display = int(gt_class[0]) if len(gt_class) > 0 else str(gt_class)
                else:
                    gt_display = int(gt_class) if isinstance(gt_class, (int, float, np.integer)) else str(gt_class)
                title += f' - GT: {gt_display}'
            except:
                title += f' - GT: {gt_class}'
        if pred_classes is not None:
            try:
                pred_class = pred_classes[sample_idx].item()
                title += f' - Pred: {pred_class}'
            except:
                pass
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save original image
        original_path = os.path.join(output_dir, f"original_input_{dataset_type.lower()}_sample_{sample_idx+1}.png")
        plt.savefig(original_path, dpi=300, bbox_inches='tight')
        print(f"✓ Original input image saved to: {original_path}")
        plt.show()
    
    # 1. All stages comparison
    num_stages = len(stage_features)
    fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
    
    if num_stages == 1:
        axes = axes.reshape(2, 1)
    
    # Create title with ground truth and prediction
    title = 'PVT Stage Feature Maps Comparison'
    if gt_class is not None:
        try:
            if isinstance(gt_class, (np.ndarray, list)):
                gt_display = int(gt_class[0]) if len(gt_class) > 0 else str(gt_class)
            else:
                gt_display = int(gt_class) if isinstance(gt_class, (int, float, np.integer)) else str(gt_class)
            title += f' (GT: {gt_display}'
            if pred_classes is not None:
                try:
                    pred_class = pred_classes[sample_idx].item()
                    title += f', Pred: {pred_class}'
                except:
                    pass
            title += ')'
        except:
            pass
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        feat_sample = feat[sample_idx]  # (C, H, W)
        C, H, W = feat_sample.shape
        
        # Top row: Absolute average of all channels
        mean_feat = feat_sample.abs().mean(dim=0).cpu().numpy()
        im1 = axes[0, i].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Stage {info["stage"]}\n(C={C}, H={H}, W={W})')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Bottom row: Channel statistics
        channel_means = feat_sample.mean(dim=(1, 2)).cpu().numpy()
        axes[1, i].plot(channel_means, 'b-', alpha=0.7, linewidth=2)
        axes[1, i].set_title(f'Channel Means\n(Heads={info["heads"]}, SR={info["sr_ratio"]})')
        axes[1, i].set_xlabel('Channel Index')
        axes[1, i].set_ylabel('Mean Value')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(output_dir, f"pvt_{dataset_type.lower()}_comparison_sample_{sample_idx+1}.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {comparison_path}")
    plt.show()
    
    # 2. Detailed plots for each stage
    for i, (feat, info) in enumerate(zip(stage_features, stage_info)):
        feat_sample = feat[sample_idx]  # (C, H, W)
        C, H, W = feat_sample.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Create title with ground truth and prediction
        stage_title = f'Stage {info["stage"]} Detailed Features (C={C}, H={H}, W={W})'
        if gt_class is not None:
            try:
                if isinstance(gt_class, (np.ndarray, list)):
                    gt_display = int(gt_class[0]) if len(gt_class) > 0 else str(gt_class)
                else:
                    gt_display = int(gt_class) if isinstance(gt_class, (int, float, np.integer)) else str(gt_class)
                stage_title += f' - GT: {gt_display}'
                if pred_classes is not None:
                    try:
                        pred_class = pred_classes[sample_idx].item()
                        stage_title += f', Pred: {pred_class}'
                    except:
                        pass
            except:
                pass
        fig.suptitle(stage_title, fontsize=14, fontweight='bold')
        
        # 1. Original input image
        if input_batch is not None:
            img_sample = input_batch[sample_idx].clone()
            
            # 檢查是否需要反標準化（檢查數值範圍）
            if img_sample.min() < -0.5 or img_sample.max() > 1.5:
                IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_sample = img_sample * IMAGENET_STD + IMAGENET_MEAN
                img_sample = torch.clamp(img_sample, 0, 1)
            
            img_np = img_sample.permute(1, 2, 0).cpu().numpy()
            axes[0, 0].imshow(img_np, interpolation='nearest')
            axes[0, 0].set_title('Original Input Image')
            axes[0, 0].axis('off')
        
        # 2. Grid of first 16 channels
        num_channels_to_show = min(16, C)
        channel_grid = feat_sample[:num_channels_to_show].cpu().numpy()
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
        
        # 3. Average of all channels (signed)
        mean_feat = feat_sample.mean(dim=0).cpu().numpy()
        im3 = axes[1, 0].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Average of All Channels (Signed)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 4. Average of all channels (absolute)
        mean_feat_abs = feat_sample.abs().mean(dim=0).cpu().numpy()
        im4 = axes[1, 1].imshow(mean_feat_abs, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Average of All Channels (Absolute)')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save detailed plot
        detail_path = os.path.join(output_dir, f"stage_{info['stage']}_{dataset_type.lower()}_detail_sample_{sample_idx+1}.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stage {info['stage']} detailed plot saved to: {detail_path}")
        plt.show()
    
    print(f"\n🎉 All visualizations completed!")
    print(f"📁 Output directory: {output_dir}")
    print("📊 Generated files:")
    print(f"  - original_input_{dataset_type.lower()}_sample_X.png (Original input image with GT and prediction)")
    print(f"  - pvt_{dataset_type.lower()}_comparison_sample_X.png (All stages comparison)")
    print(f"  - stage_X_{dataset_type.lower()}_detail_sample_X.png (Detailed feature maps for each stage)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PVT Stage Visualization')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model')
    
    args = parser.parse_args()
    
    print("PVT Stage Visualization")
    print("=" * 50)
    print(f"Using dataset (from config): {config['dataset']}")
    print(f"Model path: {args.model_path or config.get('load_model_name', 'None')}")
    
    # Load model and data
    model, batch, labels = load_model_and_data(model_path=args.model_path)
    
    # Extract stage features
    stage_features, stage_info, pred_classes, pred_probs = extract_stage_features(model, batch)
    
    # Visualize stages for multiple samples
    for sample_idx in range(min(4, batch.shape[0])):  # 可視化前 4 個樣本
        print(f"\n=== 可視化樣本 {sample_idx + 1} ===")
        visualize_stages(stage_features, stage_info, sample_idx=sample_idx, labels=labels, 
                        input_batch=batch, pred_classes=pred_classes, pred_probs=pred_probs, dataset_type=config['dataset'])

if __name__ == "__main__":
    main()
