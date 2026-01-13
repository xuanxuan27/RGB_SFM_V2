import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from config import config, arch
from dataloader import get_dataloader
from models.RGB_SFM_PVT import RGB_SFM_PVT
from torchvision.utils import save_image
import json


def _to_numpy(t: torch.Tensor):
    return t.detach().float().cpu().numpy()


def _save_heatmap(img, save_path, title=None, cmap='viridis'):
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.tight_layout(pad=0)
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


@torch.no_grad()
def visualize_forward_responses(
    out_root: str = None,
    num_samples: int = 8,
    num_channels_per_stage: int = 4,
    model_ckpt: str = "runs/train/exp86/RGB_SFM_PVT_best.pth",
):
    device = config['device']

    # dataloader
    train_loader, test_loader = get_dataloader(
        dataset=config['dataset'],
        root=config['root'] + '/data/',
        batch_size=config['batch_size'],
        input_size=config['input_shape']
    )

    # 準備資料
    batch = next(iter(test_loader))
    images, labels = batch
    images = images[:num_samples].to(device)

    # 構建 RGB_SFM_PVT
    rm_kwargs = dict(arch['args'])
    model = RGB_SFM_PVT(
        in_channels=rm_kwargs.get('in_channels', 3),
        out_channels=rm_kwargs.get('out_channels', 30),
        input_size=config.get('input_shape', (224, 224)),
        freeze_sfm=False,
        Conv2d_kernel=rm_kwargs.get('Conv2d_kernel'),
        channels=rm_kwargs.get('channels'),
        SFM_filters=rm_kwargs.get('SFM_filters'),
        strides=rm_kwargs.get('strides'),
        conv_method=rm_kwargs.get('conv_method'),
        initial=rm_kwargs.get('initial'),
        rbfs=rm_kwargs.get('rbfs'),
        SFM_methods=rm_kwargs.get('SFM_methods'),
        paddings=rm_kwargs.get('paddings'),
        fc_input=rm_kwargs.get('fc_input'),
        device=str(device) if isinstance(device, torch.device) else device,
        activate_params=rm_kwargs.get('activate_params'),
        color_filter=rm_kwargs.get('color_filter', 'new_100'),
        expand_pvt_in_channels=rm_kwargs.get('expand_pvt_in_channels', True),
        apply_imagenet_norm=rm_kwargs.get('apply_imagenet_norm', False),
        interp_mode=rm_kwargs.get('interp_mode', 'bilinear'),
        head_schedule=rm_kwargs.get('head_schedule', 'auto'),
        head_dim_target=rm_kwargs.get('head_dim_target', 64),
        max_heads=rm_kwargs.get('max_heads', 16),
        drop_rate=rm_kwargs.get('drop_rate', 0.0),
        drop_path_rate=rm_kwargs.get('drop_path_rate', 0.1),
    ).to(device)

    # 載入已訓練權重（若存在）
    if model_ckpt and os.path.exists(model_ckpt):
        ckpt = torch.load(model_ckpt, map_location=device)
        state = ckpt.get('model_weights', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[警告] 載入權重時鍵不匹配，missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"[提示] 找不到權重檔：{model_ckpt}，將使用隨機初始化模型")

    model.eval()

    # 輸出根路徑
    if out_root is None:
        out_root = f"./detect/{config['dataset']}/SFM_CNN_PVT_forward"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    # 先取得模型預測（使用整個 model forward 以拿到分類 logits）
    logits = model(images)
    preds = logits.argmax(dim=1).detach().cpu().tolist()
    gts = labels[:num_samples].detach().cpu().tolist()

    # 類別名稱（若已知資料集）
    class_names = None
    if str(config.get('dataset','')).upper() == 'CIFAR10':
        class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # 1) 取得 RM 各 stage 特徵（不經 fc）：使用 hooks 擷取每個子模組輸出
    stage_outputs = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            stage_outputs[name] = out.detach()
        return _hook

    # 對 RGB_convs 中的每個層級註冊 hook（順序視為 stage 順序）
    for idx, m in enumerate(model.rgb_sfm.RGB_convs):
        h = m.register_forward_hook(_make_hook(f'stage_{idx:02d}'))
        hooks.append(h)

    # 前向一次以填充 stage_outputs
    rm_feat = model.rgb_sfm.RGB_convs(images)  # 最終 RGB 分支輸出 [B, C_rm, H, W]

    # 解除 hook
    for h in hooks:
        h.remove()

    # 產生最終 RM 的通道平均（保留）
    rm_mean = rm_feat.mean(dim=1)  # [B, H, W]

    # 2) 銜接 PVT 的輸入（與 forward 對齊）
    rm_norm = model._normalize_per_sample(rm_feat)
    if model.rm_to_rgb is not None:
        # A 方案：1x1 壓到 3 通道 + resize + (可選) ImageNet norm
        x3 = model.rm_to_rgb(rm_norm)
        x3 = torch.nn.functional.interpolate(
            x3, size=model.input_size, mode=model.interp_mode, align_corners=False
        )
        if model.apply_imagenet_norm:
            x3 = (x3 - model._im_mean.to(x3.device)) / model._im_std.to(x3.device)
    else:
        # B 方案：直接改 PVT 第一層接 C_rm + resize
        x3 = torch.nn.functional.interpolate(
            rm_norm, size=model.input_size, mode=model.interp_mode, align_corners=False
        )

    # 3) 從 PVT backbone 取得最終 stage 的序列與特徵圖
    outs, (last_seq, H, W) = model.pvt.backbone.forward_features(x3)
    feat_map = last_seq.transpose(1, 2).reshape(x3.shape[0], model.pvt.backbone.dims[-1], H, W)
    pvt_mean = feat_map.mean(dim=1)  # [B, H, W]

    # 4) 儲存圖像
    for idx in range(images.shape[0]):
        sample_dir = os.path.join(out_root, f'sample_{idx:02d}')
        Path(sample_dir).mkdir(parents=True, exist_ok=True)

        # 原圖（存 PNG）
        img_path = os.path.join(sample_dir, 'input.png')
        save_image(images[idx].clamp(0,1), img_path)

        # 預測與 GT（存成 meta.json）
        # preds 是通過 argmax 得到的整數列表
        pred_idx = int(preds[idx])
        
        # gts 可能是整數（CIFAR10）或 one-hot 列表（Colored_MNIST）
        gt_item = gts[idx]
        if isinstance(gt_item, (list, tuple)):
            # 如果是列表，可能是 one-hot 編碼，找最大值索引
            try:
                # 嘗試轉換為 numpy array 以便處理
                gt_array = np.array(gt_item, dtype=np.float32)
                # 檢查是否是標準 one-hot（只有一個 1，其他都是 0）
                unique_vals = np.unique(gt_array)
                if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals and np.sum(gt_array == 1) == 1:
                    gt_idx = int(np.argmax(gt_array))
                else:
                    # 其他情況，使用 argmax
                    gt_idx = int(np.argmax(gt_array))
            except Exception as e:
                # 如果轉換失敗，嘗試直接使用列表的 index 方法
                try:
                    if 1 in gt_item:
                        gt_idx = int(gt_item.index(1))
                    else:
                        # 使用 numpy 的 argmax（需要先轉換）
                        gt_array = np.array(gt_item)
                        gt_idx = int(np.argmax(gt_array))
                except:
                    gt_idx = 0  # 如果都失敗，使用默認值
        elif isinstance(gt_item, (int, float, np.integer, np.floating)):
            # 已經是整數或浮點數
            gt_idx = int(gt_item)
        else:
            # 其他類型，嘗試轉換
            try:
                gt_idx = int(gt_item)
            except:
                gt_idx = 0  # 如果無法轉換，使用默認值 0
        meta = {
            'pred_index': pred_idx,
            'gt_index': gt_idx,
            'pred_name': (class_names[pred_idx] if class_names and 0 <= pred_idx < len(class_names) else str(pred_idx)),
            'gt_name': (class_names[gt_idx] if class_names and 0 <= gt_idx < len(class_names) else str(gt_idx))
        }
        with open(os.path.join(sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # RM 通道平均熱圖（最終輸出）
        _save_heatmap(
            _to_numpy(rm_mean[idx]),
            os.path.join(sample_dir, 'rm_mean.png'),
            title='RM mean',
            cmap='magma'
        )

        # 各 stage 的輸出可視化（通道平均與代表通道）
        for sname in sorted(stage_outputs.keys()):
            feat = stage_outputs[sname]  # [B, C, H, W]
            stage_dir = os.path.join(sample_dir, sname)
            Path(stage_dir).mkdir(parents=True, exist_ok=True)

            # 通道平均
            _save_heatmap(
                _to_numpy(feat[idx].mean(dim=0)),
                os.path.join(stage_dir, f'{sname}_mean.png'),
                title=f'{sname} mean',
                cmap='magma'
            )

            # 代表通道
            c_s = feat.shape[1]
            step_s = max(1, c_s // num_channels_per_stage)
            pick_s = [min(i * step_s, c_s - 1) for i in range(num_channels_per_stage)]
            for ci in pick_s:
                _save_heatmap(
                    _to_numpy(feat[idx, ci]),
                    os.path.join(stage_dir, f'{sname}_c{ci:04d}.png'),
                    title=f'{sname} c{ci}',
                    cmap='inferno'
                )

        # PVT 最終通道平均熱圖
        _save_heatmap(
            _to_numpy(pvt_mean[idx]),
            os.path.join(sample_dir, 'pvt_mean.png'),
            title='PVT last mean',
            cmap='viridis'
        )

        # 顯示數個代表性通道
        c_rm = rm_feat.shape[1]
        step_rm = max(1, c_rm // num_channels_per_stage)
        pick_rm = [min(i * step_rm, c_rm - 1) for i in range(num_channels_per_stage)]

        for ci in pick_rm:
            _save_heatmap(
                _to_numpy(rm_feat[idx, ci]),
                os.path.join(sample_dir, f'rm_c{ci:04d}.png'),
                title=f'RM c{ci}',
                cmap='inferno'
            )

        c_pvt = feat_map.shape[1]
        step_p = max(1, c_pvt // num_channels_per_stage)
        pick_p = [min(i * step_p, c_pvt - 1) for i in range(num_channels_per_stage)]

        for ci in pick_p:
            _save_heatmap(
                _to_numpy(feat_map[idx, ci]),
                os.path.join(sample_dir, f'pvt_c{ci:04d}.png'),
                title=f'PVT c{ci}',
                cmap='plasma'
            )

        # PVT 各 stage 輸出可視化
        for si, sfeat in enumerate(outs):
            # 期望為 [B, C, Hs, Ws]（若為序列，需還原空間；此處假設 outs 已為 feature map）
            stage_dir = os.path.join(sample_dir, f'pvt_stage_{si:02d}')
            Path(stage_dir).mkdir(parents=True, exist_ok=True)

            # 通道平均
            _save_heatmap(
                _to_numpy(sfeat[idx].mean(dim=1) if sfeat.dim() == 3 else sfeat[idx].mean(dim=0)),
                os.path.join(stage_dir, f'pvt_stage_{si:02d}_mean.png'),
                title=f'pvt_stage_{si:02d} mean',
                cmap='cividis'
            )

            # 代表通道
            if sfeat.dim() == 4:
                c_s = sfeat.shape[1]
                step_s = max(1, c_s // num_channels_per_stage)
                pick_s = [min(i * step_s, c_s - 1) for i in range(num_channels_per_stage)]
                for ci in pick_s:
                    _save_heatmap(
                        _to_numpy(sfeat[idx, ci]),
                        os.path.join(stage_dir, f'pvt_stage_{si:02d}_c{ci:04d}.png'),
                        title=f'pvt_stage_{si:02d} c{ci}',
                        cmap='viridis'
                    )

        # 生成與 pvt_colored_mnist_visualizer 類似的比較圖（各 stage 概覽）
        try:
            num_stages = len(outs)
            fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
            if num_stages == 1:
                axes = axes.reshape(2, 1)

            # 標題含 GT / Pred
            title = 'PVT Stage Feature Maps Comparison'
            if gts is not None:
                title += f' (GT: {gts[idx]}'
                if preds is not None:
                    title += f', Pred: {preds[idx]}'
                title += ')'
            fig.suptitle(title, fontsize=16, fontweight='bold')

            for si, sfeat in enumerate(outs):
                fs = sfeat[idx]  # [C, H, W]
                C, Hs, Ws = fs.shape

                mean_abs = fs.abs().mean(dim=0).cpu().numpy()
                im = axes[0, si].imshow(mean_abs, cmap='viridis', aspect='auto')
                axes[0, si].set_title(f'Stage {si+1}\n(C={C}, H={Hs}, W={Ws})')
                axes[0, si].axis('off')
                plt.colorbar(im, ax=axes[0, si], fraction=0.046, pad=0.04)

                ch_means = fs.mean(dim=(1,2)).cpu().numpy()
                axes[1, si].plot(ch_means, 'b-', alpha=0.7, linewidth=2)
                axes[1, si].set_title('Channel Means')
                axes[1, si].set_xlabel('Channel Index')
                axes[1, si].set_ylabel('Mean Value')
                axes[1, si].grid(True, alpha=0.3)

            plt.tight_layout()
            comp_path = os.path.join(sample_dir, 'pvt_stages_comparison.png')
            plt.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"[警告] 生成 PVT 比較圖失敗：{e}")

        # 生成每個 stage 的詳細圖（原圖 + 前16通道格子 + 簽名平均 + 絕對值平均）
        try:
            # 準備顯示原圖（反標準化檢查）
            img_show = images[idx].clone().detach().cpu()
            if img_show.min() < -0.5 or img_show.max() > 1.5:
                IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_show = img_show * IMAGENET_STD + IMAGENET_MEAN
            img_show = img_show.clamp(0,1)
            img_np = img_show.permute(1,2,0).numpy()

            for si, sfeat in enumerate(outs):
                fs = sfeat[idx]  # [C, H, W]
                C, Hs, Ws = fs.shape
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                stage_title = f'Stage {si+1} Detailed (C={C}, H={Hs}, W={Ws})'
                stage_title += f' - GT: {gts[idx]}'
                stage_title += f', Pred: {preds[idx]}'
                fig.suptitle(stage_title, fontsize=14, fontweight='bold')

                # 原圖
                axes[0,0].imshow(img_np, interpolation='nearest')
                axes[0,0].set_title('Original Input Image')
                axes[0,0].axis('off')

                # 前 16 通道格子
                import numpy as np
                num_channels_to_show = min(16, C)
                grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
                heat = np.zeros((grid_size*Hs, grid_size*Ws))
                fs_cpu = fs.cpu().numpy()
                for j in range(num_channels_to_show):
                    r = j // grid_size
                    c = j % grid_size
                    heat[r*Hs:(r+1)*Hs, c*Ws:(c+1)*Ws] = fs_cpu[j]
                im2 = axes[0,1].imshow(heat, cmap='viridis', aspect='auto')
                axes[0,1].set_title(f'First {num_channels_to_show} Channels')
                axes[0,1].axis('off')
                plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)

                # 簽名平均
                im3 = axes[1,0].imshow(fs.mean(dim=0).cpu().numpy(), cmap='viridis', aspect='auto')
                axes[1,0].set_title('Average of All Channels (Signed)')
                axes[1,0].axis('off')
                plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)

                # 絕對值平均
                im4 = axes[1,1].imshow(fs.abs().mean(dim=0).cpu().numpy(), cmap='viridis', aspect='auto')
                axes[1,1].set_title('Average of All Channels (Absolute)')
                axes[1,1].axis('off')
                plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)

                plt.tight_layout()
                detail_path = os.path.join(sample_dir, f'pvt_stage_{si:02d}_detail.png')
                plt.savefig(detail_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"[警告] 生成 PVT 詳細圖失敗：{e}")

    print(f"Saved forward response visualizations to: {out_root}")


if __name__ == '__main__':
    visualize_forward_responses()


