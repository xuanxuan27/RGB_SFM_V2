import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

from models.PVTv2_B0 import PVTv2_B0
from dataloader import get_dataloader
import config

def analyze():
    # 1. 先載入 checkpoint 來確定輸出通道數
    checkpoint = torch.load("/home/xuan/RGB_SFM_V2/runs/train/exp125/PVTv2_B0_best.pth", map_location='cpu', weights_only=False)
    
    # 從 checkpoint 中獲取模型權重
    if 'model_weights' in checkpoint:
        state_dict = checkpoint['model_weights']
    else:
        state_dict = checkpoint
    
    # 從 state_dict 中推斷輸出通道數（檢查 model.head.weight 的形狀）
    if 'model.head.weight' in state_dict:
        out_channels = state_dict['model.head.weight'].shape[0]
        print(f"從 checkpoint 推斷出輸出通道數: {out_channels}")
    else:
        # 如果找不到，使用默認值
        out_channels = 10
        print(f"無法從 checkpoint 推斷輸出通道數，使用默認值: {out_channels}")
    
    # 2. 建立相同結構的模型（使用正確的輸出通道數）[cite: 735, 942]
    model = PVTv2_B0(out_channels=out_channels) 
    
    # 3. 載入訓練好的權重
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 4. 從 config 獲取 dataset 並載入測試數據
    print(f"載入 dataset: {config.config['dataset']}")
    _, test_dataloader = get_dataloader(
        dataset=config.config['dataset'],
        root=config.config['root'] + '/data/',
        batch_size=1,  # 只取一張圖片
        input_size=config.config['input_shape'],
        use_pretrained_vit=True
    )
    
    # 從測試數據集中取一張圖片
    img, label = next(iter(test_dataloader))
    print(f"測試圖片形狀: {img.shape}, 標籤: {label}") 

    # 5. 提取各階段特徵 (修改後的提取邏輯) [cite: 208, 813]
    with torch.no_grad():
        x = img  # img 已經從 dataloader 獲取，形狀為 (1, 3, H, W)
        if model.channel_adapter is not None:
            x = model.channel_adapter(x)
        
        # 逐一走過四個階段 [cite: 209, 954]
        stage_outputs = []
        
        # 第一層 Patch Embed [cite: 822, 1024]
        x = model.model.patch_embed(x)
        
        for i, stage in enumerate(model.model.stages):
            x = stage(x)
            stage_outputs.append(x) # 儲存 Stage 1, 2, 3, 4 的輸出 [cite: 825, 851, 884, 914]
            print(f"Stage {i+1} 輸出形狀: {x.shape}")

    # 6. 可視化 (觀察特徵放大現象)
    visualize_stages(stage_outputs)
    visualize_expansion(img, stage_outputs)

    # 7. 抓取單層注意力圖
    try:
        spatial_attn, topk_indices = get_pvt_topk_attention(model, img)
        visualize_topk(img, spatial_attn, topk_indices)
        print(f"Top-k Attention visualization completed")
    except Exception as e:
        print(f"Warning: Could not extract attention weights: {e}")
        print("Skipping single layer attention visualization...")
    
    # 8. 抓取並可視化所有層的注意力圖
    try:
        print("\nAnalyzing all layers attention...")
        all_results = analyze_all_layers(model, img, k=5)
        visualize_all_layers_attention(img, all_results)
        print("All layers attention visualization completed")
    except Exception as e:
        print(f"Warning: Could not analyze all layers attention: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping all layers attention visualization...")

    # # 8. 抓取通道重要性
    # channel_importance = get_channel_importance(model, img)
    # visualize_channel_importance(channel_importance)
    # print(f"Channel Importance已儲存至: {save_path}")

def visualize_stages(outputs):
    # 這裡可以寫一段 matplotlib 程式碼將 outputs 轉成熱圖顯示
    os.makedirs("pvt_v2_b0_analysis/stage_outputs", exist_ok=True)
    
    for i, output in enumerate(outputs):
        # output 形狀: (1, C, H, W)
        feat = output[0]  # 移除 batch 維度: (C, H, W)
        C, H, W = feat.shape
        
        # Create 2x2 subplots to display different views
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stage {i+1} Feature Maps (C={C}, H={H}, W={W})', fontsize=16, fontweight='bold')
        
        # 1. Average of all channels
        mean_feat = feat.mean(dim=0).cpu().numpy()
        im1 = axes[0, 0].imshow(mean_feat, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Average of All Channels')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Grid of first few channels (show up to 16 channels)
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
        
        # 3. Channel statistics (mean and std)
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        channel_stds = feat.std(dim=(1, 2)).cpu().numpy()
        
        axes[1, 0].plot(channel_means, 'b-', label='Mean', alpha=0.7, linewidth=2)
        axes[1, 0].plot(channel_stds, 'r-', label='Std', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('Channel Statistics')
        axes[1, 0].set_xlabel('Channel Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature distribution histogram
        all_values = feat.flatten().cpu().numpy()
        axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Feature Value Distribution')
        axes[1, 1].set_xlabel('Feature Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"pvt_v2_b0_analysis/stage_outputs/stage_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization for Stage {i+1}")

def visualize_expansion(img, stage_outputs, save_path="pvt_v2_b0_analysis/expansion_comparison.png"):
    """
    將原圖與 4 個階段的平均熱圖並排顯示，展示特徵範圍的擴張。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 準備繪圖
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 顯示原始圖片 (img 形狀是 [1, 3, H, W])
    orig_img = img[0].permute(1, 2, 0).cpu().numpy()
    # 反正規化 (假設您使用了標準的 ImageNet 歸一化)
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 顯示四個階段的特徵演化 [cite: 813]
    titles = ["Stage 1 (H/4)", "Stage 2 (H/8)", "Stage 3 (H/16)", "Stage 4 (H/32)"]
    
    for i, feat in enumerate(stage_outputs):
        # 取得單張圖片特徵並計算通道平均值
        heatmap = feat[0].mean(dim=0).cpu().numpy()
        
        # 關鍵：將不同解析度的熱圖放大回原始尺寸 (224x224) 進行對比
        # 您會發現 Stage 1 的熱圖很精細，Stage 4 會變得很寬大（感受野放大） [cite: 451, 1060]
        axes[i+1].imshow(heatmap, cmap='jet')
        axes[i+1].set_title(titles[i])
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"感受野放大對比圖已儲存至: {save_path}")

# 儲存 Hook 抓到的注意力圖
attention_store = {}

def create_attention_wrapper(original_attn, layer_name=None, attention_dict=None):
    """
    創建一個包裝器來替換原來的attention層，使其返回attention weights
    Args:
        original_attn: 原始的attention層
        layer_name: 層的名稱（用於多層時區分）
        attention_dict: 保存attention的字典（如果為None，使用全局attention_store）
    """
    class AttentionWrapper(torch.nn.Module):
        def __init__(self, original_attn, layer_name=None, attention_dict=None):
            super().__init__()
            self.original_attn = original_attn
            self.layer_name = layer_name
            self.attention_dict = attention_dict if attention_dict is not None else attention_store
            # 複製所有屬性
            for attr in dir(original_attn):
                if not attr.startswith('_') and attr != 'forward':
                    try:
                        setattr(self, attr, getattr(original_attn, attr))
                    except:
                        pass
        
        def forward(self, x, H, W):
            """
            攔截forward過程，計算並保存attention weights
            """
            try:
                B, N, C = x.shape
                num_heads = self.original_attn.num_heads
                head_dim = C // num_heads
                scale = head_dim ** -0.5
                
                # 計算 Q
                q = self.original_attn.q(x).reshape(B, N, num_heads, head_dim)
                q = q.transpose(1, 2)  # (B, heads, N, head_dim)
                
                # 計算 K, V（處理 SR）
                if hasattr(self.original_attn, 'sr') and self.original_attn.sr is not None:
                    x_ = x.transpose(1, 2).reshape(B, C, H, W)
                    x_ = self.original_attn.sr(x_)
                    Hr, Wr = x_.shape[-2:]
                    x_ = x_.flatten(2).transpose(1, 2)
                    if hasattr(self.original_attn, 'norm'):
                        x_ = self.original_attn.norm(x_)
                else:
                    x_ = x
                    Hr, Wr = H, W
                
                kv = self.original_attn.kv(x_)
                kv = kv.reshape(B, Hr * Wr, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv[0], kv[1]  # (B, heads, N', head_dim)
                
                # 計算 attention weights
                attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N')
                attn = attn.softmax(dim=-1)
                
                # 保存 attention weights（在dropout之前保存，這樣可以看到原始的attention分數）
                if self.layer_name is not None:
                    self.attention_dict[self.layer_name] = attn.detach()
                else:
                    self.attention_dict['weights'] = attn.detach()
                
                # 繼續原始forward
                attn = self.original_attn.attn_drop(attn)
                out = attn @ v  # (B, heads, N, head_dim)
                out = out.transpose(1, 2).reshape(B, N, C)
                out = self.original_attn.proj(out)
                out = self.original_attn.proj_drop(out)
                
                return out
            except Exception as e:
                # 如果wrapper失敗，回退到原始forward
                print(f"Warning: Attention wrapper failed for {self.layer_name}: {e}")
                return self.original_attn(x, H, W)
    
    return AttentionWrapper(original_attn, layer_name, attention_dict)

def hook_fn(module, input, output):
    """
    抓取 Attention 模組的輸出
    在 timm 的 PVTv2 中，attention 層只返回 output，不返回 attention weights
    所以我們需要使用 wrapper 來攔截 attention 計算過程
    """
    # 如果 output 是 tuple，通常第二個元素是注意力權重
    if isinstance(output, tuple):
        if len(output) >= 2:
            attention_store['weights'] = output[1]  # 第二個元素是注意力權重
        else:
            attention_store['weights'] = output[0]
    else:
        # timm的PVTv2只返回output，attention weights已經在wrapper中保存
        # 這裡不需要做任何事，因為attention weights已經在forward過程中被保存了
        pass

def get_pvt_topk_attention(model, img_tensor, k=5):
    attention_store.clear()
    
    # 1. 替換最後一個 stage 的 attention 層為 wrapper
    # 保存原始attention層的引用
    target_block = model.model.stages[3].blocks[-1]
    original_attn = target_block.attn
    
    # 創建並替換為wrapper
    wrapped_attn = create_attention_wrapper(original_attn, layer_name=None, attention_dict=attention_store)
    target_block.attn = wrapped_attn
    
    # 2. 前向傳播（wrapper會在forward過程中保存attention weights）
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # 3. 恢復原始attention層
    target_block.attn = original_attn
    
    # 4. 處理注意力矩陣
    if 'weights' not in attention_store:
        print("Warning: Could not extract attention weights. Using feature map instead.")
        # 使用 Stage 4 的特徵圖作為替代
        with torch.no_grad():
            x = img_tensor
            if model.channel_adapter is not None:
                x = model.channel_adapter(x)
            x = model.model.patch_embed(x)
            for stage in model.model.stages[:4]:
                x = stage(x)
        spatial_attn = x[0].mean(dim=0).flatten()  # (H*W,)
        topk_vals, topk_indices = torch.topk(spatial_attn, k=k)
        return spatial_attn, topk_indices.cpu().numpy()
    
    attn_weights = attention_store['weights']
    
    # 根據實際形狀處理注意力權重
    print(f"Debug: attn_weights shape: {attn_weights.shape}, dim: {attn_weights.dim()}")
    
    if attn_weights.dim() == 4:
        # (B, Head, N, N) - 取第一個 Batch，並在 Head 維度取平均
        attn_weights = attn_weights[0].mean(dim=0)  # (N, N)
    elif attn_weights.dim() == 3:
        # (Head, N, N) - 所有 head 的平均
        attn_weights = attn_weights.mean(dim=0)  # (N, N)
    elif attn_weights.dim() == 2:
        # (N, N) - 已經是 2D，直接使用
        pass
    elif attn_weights.dim() == 1:
        # (N,) - 已經是 1D，可能是空間注意力
        print("Warning: attn_weights is 1D, using as spatial attention directly.")
        spatial_attn = attn_weights
        topk_vals, topk_indices = torch.topk(spatial_attn, k=k)
        return spatial_attn, topk_indices.cpu().numpy()
    else:
        print(f"Warning: Unexpected attention shape {attn_weights.shape}, using feature map instead.")
        # 使用特徵圖作為替代
        with torch.no_grad():
            x = img_tensor
            if model.channel_adapter is not None:
                x = model.channel_adapter(x)
            x = model.model.patch_embed(x)
            for stage in model.model.stages[:4]:
                x = stage(x)
        spatial_attn = x[0].mean(dim=0).flatten()
        topk_vals, topk_indices = torch.topk(spatial_attn, k=k)
        return spatial_attn, topk_indices.cpu().numpy()
    
    # 現在 attn_weights 應該是 (N, N) 形狀
    if attn_weights.dim() != 2:
        raise ValueError(f"Expected 2D attention weights after processing, got {attn_weights.dim()}D")
    
    N = attn_weights.shape[0]  # 此時 N 應該是 49 (7x7) 或 50 (如果有 cls_token)

    # --- 關鍵修改：GAP 模式下的關注點提取 ---
    if N == 49:  # 沒有 cls_token 的情況
        # 沿著 Query 維度取平均，得到各個位置被「關注」的總體權重
        # 這代表：平均來說，所有 Patch 都看了哪些地方
        spatial_attn = attn_weights.mean(dim=0)  # (49,)
    elif N == 50:  # 有 cls_token 的情況
        # 取出 cls_token 對所有空間 Token 的注意力
        spatial_attn = attn_weights[0, 1:]  # (49,)
    else:
        # 其他情況，使用全局平均池化
        print(f"Warning: Unexpected N={N}, using mean pooling.")
        spatial_attn = attn_weights.mean(dim=0)  # (N,)
    # ----------------------------------------

    # 4. 找出 Top-k
    topk_vals, topk_indices = torch.topk(spatial_attn, k=k)
    
    return spatial_attn, topk_indices.cpu().numpy()

# 建立一個字典來儲存所有層的權重
all_attentions = {}
original_attentions = {}  # 保存原始attention層的引用

def get_all_attention_hooks(model, attention_dict):
    """
    為所有attention層創建wrapper並保存原始層的引用
    返回所有被替換的層的信息，以便後續恢復
    Args:
        model: PVTv2模型
        attention_dict: 用於保存attention weights的字典
    """
    replaced_layers = []
    
    # 遍歷 4 個 Stage
    for s_idx, stage in enumerate(model.model.stages):
        # 遍歷該 Stage 內所有的 Block
        for b_idx, block in enumerate(stage.blocks):
            layer_name = f"stage{s_idx}_block{b_idx}"
            
            # 保存原始attention層
            original_attn = block.attn
            original_attentions[layer_name] = original_attn
            
            # 創建並替換為wrapper（每個wrapper會將attention保存到attention_dict中對應的key）
            wrapped_attn = create_attention_wrapper(original_attn, layer_name=layer_name, attention_dict=attention_dict)
            block.attn = wrapped_attn
            
            replaced_layers.append((layer_name, block, original_attn))
            
    return replaced_layers

def analyze_all_layers(model, img_tensor, k=5):
    all_attentions.clear()
    original_attentions.clear()
    
    # 1. 替換所有層的attention為wrapper
    replaced_layers = get_all_attention_hooks(model, all_attentions)
    
    # 2. 前向傳播（每個wrapper會在forward過程中自動將attention保存到all_attentions字典中）
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # 3. 恢復所有原始attention層
    for layer_name, block, original_attn in replaced_layers:
        block.attn = original_attn
    
    # 4. 處理每一層的結果
    results = {}
    for name, attn in all_attentions.items():
        try:
            # 處理不同形狀的 attention weights
            if isinstance(attn, torch.Tensor):
                if attn.dim() == 4:
                    # (B, H, N, N) - 取第一個 batch，所有 head 的平均
                    avg_attn = attn[0].mean(dim=0)  # (N, N)
                elif attn.dim() == 3:
                    # (H, N, N) - 所有 head 的平均
                    avg_attn = attn.mean(dim=0)  # (N, N)
                elif attn.dim() == 2:
                    # (N, N) - 直接使用
                    avg_attn = attn
                elif attn.dim() == 1:
                    # (N,) - 已經是空間注意力
                    spatial_attn = attn
                    _, topk_indices = torch.topk(spatial_attn, k=min(k, spatial_attn.shape[0]))
                    results[name] = {
                        "spatial_attn": spatial_attn,
                        "topk_indices": topk_indices.cpu().numpy()
                    }
                    continue
                else:
                    print(f"Warning: Unexpected attention shape {attn.shape} for {name}, skipping...")
                    continue
                
                # 進行全域平均 (GAP 模式) 得到空間注意力
                spatial_attn = avg_attn.mean(dim=0)  # (N,)
                
                # 取得 Top-k
                _, topk_indices = torch.topk(spatial_attn, k=min(k, spatial_attn.shape[0]))
                
                results[name] = {
                    "spatial_attn": spatial_attn,
                    "topk_indices": topk_indices.cpu().numpy()
                }
            else:
                print(f"Warning: Attention for {name} is not a tensor (type: {type(attn)}), skipping...")
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
        
    return results

def visualize_all_layers_attention(img_tensor, all_results, input_size=224, save_dir="pvt_v2_b0_analysis/all_layers_attention"):
    """
    可視化所有層的 attention map
    Args:
        img_tensor: 輸入圖片 tensor，形狀為 (1, 3, H, W)
        all_results: analyze_all_layers 返回的結果字典
        input_size: 輸入圖片尺寸
        save_dir: 保存目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 將圖片 tensor 轉換為 numpy
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Stage 對應的空間分辨率
    stage_grid_sizes = {
        0: 56,  # Stage 1: 56x56
        1: 28,  # Stage 2: 28x28
        2: 14,  # Stage 3: 14x14
        3: 7    # Stage 4: 7x7
    }
    
    # 為每個 stage 創建一個大圖
    for stage_idx in range(4):
        stage_name = f"stage{stage_idx}"
        stage_layers = {k: v for k, v in all_results.items() if k.startswith(stage_name)}
        
        if not stage_layers:
            continue
        
        # 計算子圖布局（每個 stage 有 2 個 blocks）
        num_blocks = len(stage_layers)
        fig, axes = plt.subplots(2, num_blocks, figsize=(6 * num_blocks, 12))
        if num_blocks == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Stage {stage_idx + 1} Attention Maps', fontsize=16, fontweight='bold')
        
        for block_idx, (layer_name, result) in enumerate(sorted(stage_layers.items())):
            spatial_attn = result['spatial_attn']
            topk_indices = result['topk_indices']
            
            # 轉換為 numpy
            if isinstance(spatial_attn, torch.Tensor):
                spatial_attn = spatial_attn.cpu().numpy()
            
            # 確定 grid_size
            attn_size = spatial_attn.size
            grid_size = stage_grid_sizes[stage_idx]
            
            # 重塑為 2D
            if attn_size == grid_size * grid_size:
                heatmap_2d = spatial_attn.reshape(grid_size, grid_size)
            else:
                # 如果不是標準大小，嘗試插值
                side_len = int(np.sqrt(attn_size))
                if side_len * side_len == attn_size:
                    heatmap_2d = spatial_attn.reshape(side_len, side_len)
                    heatmap_2d = cv2.resize(heatmap_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
                else:
                    # 使用插值
                    temp_size = int(np.ceil(np.sqrt(attn_size)))
                    temp_2d = np.zeros(temp_size * temp_size)
                    temp_2d[:attn_size] = spatial_attn.flatten()
                    temp_2d = temp_2d.reshape(temp_size, temp_size)
                    heatmap_2d = cv2.resize(temp_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
            
            # 上排：原始圖片 + Top-k 標記
            img_with_boxes = img_bgr.copy()
            stride = input_size // grid_size
            
            for idx in topk_indices:
                if idx >= attn_size:
                    continue
                # 計算 2D 座標
                if attn_size == grid_size * grid_size:
                    row = idx // grid_size
                    col = idx % grid_size
                else:
                    # 需要映射
                    orig_side = int(np.sqrt(attn_size))
                    row = idx // orig_side
                    col = idx % orig_side
                    row = int(row * grid_size / orig_side)
                    col = int(col * grid_size / orig_side)
                
                x1 = col * stride
                y1 = row * stride
                x2 = x1 + stride
                y2 = y1 + stride
                
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            axes[0, block_idx].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            axes[0, block_idx].set_title(f'{layer_name}\nTop-{len(topk_indices)} Patches')
            axes[0, block_idx].axis('off')
            
            # 下排：Attention heatmap
            heatmap_resized = cv2.resize(heatmap_2d, (input_size, input_size))
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            im = axes[1, block_idx].imshow(heatmap_normalized, cmap='jet', aspect='auto')
            axes[1, block_idx].set_title('Attention Heatmap')
            axes[1, block_idx].axis('off')
            plt.colorbar(im, ax=axes[1, block_idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/stage_{stage_idx + 1}_attention.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Stage {stage_idx + 1} attention maps to {save_dir}/stage_{stage_idx + 1}_attention.png")
    
    # 創建一個總覽圖，顯示所有層的 attention heatmap
    create_summary_heatmap(all_results, save_dir)

def create_summary_heatmap(all_results, save_dir):
    """
    創建一個總覽圖，顯示所有層的 attention heatmap
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 24))
    fig.suptitle('All Layers Attention Heatmaps Summary', fontsize=16, fontweight='bold')
    
    stage_grid_sizes = {0: 56, 1: 28, 2: 14, 3: 7}
    
    for stage_idx in range(4):
        stage_name = f"stage{stage_idx}"
        stage_layers = {k: v for k, v in all_results.items() if k.startswith(stage_name)}
        
        for block_idx, (layer_name, result) in enumerate(sorted(stage_layers.items())):
            spatial_attn = result['spatial_attn']
            
            if isinstance(spatial_attn, torch.Tensor):
                spatial_attn = spatial_attn.cpu().numpy()
            
            attn_size = spatial_attn.size
            grid_size = stage_grid_sizes[stage_idx]
            
            # 重塑為 2D
            if attn_size == grid_size * grid_size:
                heatmap_2d = spatial_attn.reshape(grid_size, grid_size)
            else:
                side_len = int(np.sqrt(attn_size))
                if side_len * side_len == attn_size:
                    heatmap_2d = spatial_attn.reshape(side_len, side_len)
                    heatmap_2d = cv2.resize(heatmap_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
                else:
                    temp_size = int(np.ceil(np.sqrt(attn_size)))
                    temp_2d = np.zeros(temp_size * temp_size)
                    temp_2d[:attn_size] = spatial_attn.flatten()
                    temp_2d = temp_2d.reshape(temp_size, temp_size)
                    heatmap_2d = cv2.resize(temp_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
            
            im = axes[stage_idx, block_idx].imshow(heatmap_2d, cmap='jet', aspect='auto')
            axes[stage_idx, block_idx].set_title(f'{layer_name}\n({grid_size}x{grid_size})')
            axes[stage_idx, block_idx].axis('off')
            plt.colorbar(im, ax=axes[stage_idx, block_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_layers_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary heatmap to {save_dir}/all_layers_summary.png")

def visualize_topk(img_tensor, spatial_attn, topk_indices, input_size=224, save_path="pvt_v2_b0_analysis/topk_attention.png"):
    """
    可視化 Top-k 注意力區域
    Args:
        img_tensor: 輸入圖片 tensor，形狀為 (1, 3, H, W)
        spatial_attn: 空間注意力權重，可能是 tensor 或 numpy array
        topk_indices: Top-k 索引
        input_size: 輸入圖片尺寸
        save_path: 保存路徑
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 將 tensor 轉換為 numpy 並處理
    if isinstance(spatial_attn, torch.Tensor):
        spatial_attn = spatial_attn.cpu().numpy()
    
    # 將圖片 tensor 轉換為 numpy (B, C, H, W) -> (H, W, C)
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    # 反正規化 (ImageNet normalization)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    # 轉換為 BGR (OpenCV 格式)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    
    # 確定 spatial_attn 的形狀並重塑
    attn_size = spatial_attn.size
    
    # 嘗試推斷 grid_size（根據常見的 PVTv2 stage 輸出大小）
    # Stage 1: 56x56 = 3136, Stage 2: 28x28 = 784, Stage 3: 14x14 = 196, Stage 4: 7x7 = 49
    possible_sizes = {
        49: 7,    # Stage 4
        196: 14,  # Stage 3
        784: 28,  # Stage 2
        3136: 56  # Stage 1
    }
    
    if attn_size in possible_sizes:
        grid_size = possible_sizes[attn_size]
        heatmap_2d = spatial_attn.reshape(grid_size, grid_size)
    else:
        # 如果不是標準大小，嘗試找到最接近的完全平方數
        side_len = int(np.sqrt(attn_size))
        if side_len * side_len == attn_size:
            # 是完全平方數
            grid_size = side_len
            heatmap_2d = spatial_attn.reshape(grid_size, grid_size)
        else:
            # 不是完全平方數，使用插值
            print(f"Warning: spatial_attn size {attn_size} is not a perfect square. Using interpolation.")
            # 找到最接近的完全平方數
            side_len = int(np.sqrt(attn_size))
            # 先重塑為最接近的形狀，然後插值到標準大小
            # 嘗試找到最接近的矩形形狀
            factors = []
            for i in range(1, int(np.sqrt(attn_size)) + 1):
                if attn_size % i == 0:
                    factors.append((i, attn_size // i))
            
            if factors:
                # 選擇最接近正方形的矩形
                h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                heatmap_2d = spatial_attn.reshape(h, w)
                # 插值到標準大小（使用 Stage 4 的 7x7）
                grid_size = 7
                heatmap_2d = cv2.resize(heatmap_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
            else:
                # 如果無法分解，直接插值
                grid_size = 7
                # 先重塑為 1D，然後插值
                heatmap_1d = spatial_attn.flatten()
                # 創建一個臨時的 2D 數組用於插值
                temp_size = int(np.ceil(np.sqrt(attn_size)))
                temp_2d = np.zeros(temp_size * temp_size)
                temp_2d[:attn_size] = heatmap_1d
                temp_2d = temp_2d.reshape(temp_size, temp_size)
                heatmap_2d = cv2.resize(temp_2d, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
    
    stride = input_size // grid_size
    
    # 繪製 Top-k 方框
    # 如果進行了插值，需要映射原始索引到新的 grid_size
    original_size = attn_size
    if original_size != grid_size * grid_size:
        # 需要將原始索引映射到新的 grid
        for orig_idx in topk_indices:
            if orig_idx >= original_size:
                continue  # 跳過無效索引
            
            # 計算原始索引對應的 2D 座標（在原始大小下）
            if original_size in possible_sizes:
                orig_grid_size = possible_sizes[original_size]
                orig_row = orig_idx // orig_grid_size
                orig_col = orig_idx % orig_grid_size
                # 映射到新的 grid_size
                row = int(orig_row * grid_size / orig_grid_size)
                col = int(orig_col * grid_size / orig_grid_size)
            else:
                # 如果原始大小不是標準大小，使用比例映射
                orig_side = int(np.sqrt(original_size)) if int(np.sqrt(original_size))**2 == original_size else int(np.sqrt(original_size)) + 1
                orig_row = orig_idx // orig_side
                orig_col = orig_idx % orig_side
                row = int(orig_row * grid_size / orig_side)
                col = int(orig_col * grid_size / orig_side)
            
            # 計算在原圖上的像素座標
            x1 = col * stride
            y1 = row * stride
            x2 = x1 + stride
            y2 = y1 + stride
            
            # 畫出關注點方框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Top", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        # 標準情況，直接使用索引
        for idx in topk_indices:
            if idx >= original_size:
                continue  # 跳過無效索引
            # 將 1D 索引轉回 2D 座標
            row = idx // grid_size
            col = idx % grid_size
            
            # 計算在原圖上的像素座標
            x1 = col * stride
            y1 = row * stride
            x2 = x1 + stride
            y2 = y1 + stride
            
            # 畫出關注點方框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Top", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 繪製完整熱圖對比
    heatmap_resized = cv2.resize(heatmap_2d, (input_size, input_size))
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    heatmap_img = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    result = cv2.addWeighted(overlay, 0.6, heatmap_img, 0.4, 0)
    
    # 保存結果
    cv2.imwrite(save_path, result)
    print(f"Top-k Attention Heatmap saved to: {save_path}")

if __name__ == "__main__":
    analyze()
