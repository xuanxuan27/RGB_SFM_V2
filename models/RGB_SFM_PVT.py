import torch
import torch.nn as nn
import torch.nn.functional as F

from .RGB_SFMCNN_V3 import RGB_SFMCNN_V3
from .PVT import PVT


class RGB_SFM_PVT(nn.Module):
    """
    將可解釋模型 (RM: RGB_SFMCNN_V3) 的特徵圖作為 PVT 輸入。
    預設：A 方案（1x1 將 C_rm 壓成 3 通道，再丟 PVT）
    可選：B 方案（expand_pvt_in_channels=True，改 PVT 第一層 patch_embed 接 C_rm）
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        input_size=(224, 224),
        freeze_sfm: bool = False,
        # 供 RM 構建的參數（原封不動透傳）
        Conv2d_kernel=None, channels=None, SFM_filters=None, strides=None,
        conv_method=None, initial=None, rbfs=None, SFM_methods=None,
        paddings=None, fc_input=None, device='cuda', activate_params=None,
        color_filter="new_100",
        # 串法選項
        expand_pvt_in_channels: bool = True,    # 方案B：改 PVT 第一層 in_channels = C_rm
        apply_imagenet_norm: bool = False,      # 在 rm_to_rgb 後對 3 通道做 ImageNet Normalize（做對照用）
        interp_mode: str = "bilinear",          # resize 差值模式：bicubic / bilinear
        # PVT 參數
        **pvt_kwargs                            # PVT 的其他參數（head_schedule, head_dim_target 等）
    ):
        super().__init__()
        self.input_size = input_size
        self.apply_imagenet_norm = apply_imagenet_norm
        self.interp_mode = interp_mode
        assert self.interp_mode in ("bicubic", "bilinear")

        # 1) 可解釋模型（僅 RGB 分支）
        self.rgb_sfm = RGB_SFMCNN_V3(
            in_channels=in_channels,
            out_channels=out_channels,  # 只是為了構造，實際不用其 fc
            Conv2d_kernel=Conv2d_kernel,
            channels=channels,
            SFM_filters=SFM_filters,
            strides=strides,
            conv_method=conv_method,
            initial=initial,
            rbfs=rbfs,
            SFM_methods=SFM_methods,
            paddings=paddings,
            fc_input=fc_input if fc_input is not None else 1,
            device=device,
            activate_params=activate_params,
            color_filter=color_filter,
            mode='rgb'
        )
        if freeze_sfm:
            for p in self.rgb_sfm.parameters():
                p.requires_grad = False

        # 2) 先用 dummy 取得 RM 輸出通道數 C_rm（確保裝置一致 & 避免訓練態開銷）
        with torch.no_grad():
            was_train = self.rgb_sfm.training
            self.rgb_sfm.eval()
            # 以 RM 目前參數所在裝置建立 dummy（避免 CPU/GPU 不一致）
            rm_dev = next(self.rgb_sfm.parameters()).device
            dummy = torch.zeros(1, in_channels, *self.input_size, device=rm_dev)
            rm_dummy = self.rgb_sfm.RGB_convs(dummy)      # [1, C_rm, H, W]
            self.C_rm = rm_dummy.shape[1]
            if was_train:
                self.rgb_sfm.train()

        # 3) 建 PVT + 換 head（先不移動裝置，交由外部 .to() 同步）
        # 根據方案決定 PVT 的 in_channels
        if expand_pvt_in_channels:
            # 方案B：改 PVT 第一層接 C_rm
            # 先創建 PVT（in_channels=3 以讓它正常初始化）
            self.pvt = PVT(
                in_channels=3,  # 先設為 3，稍後再修改
                out_channels=out_channels,
                input_size=input_size,
                **pvt_kwargs
            )
            
            # 禁用 channel_adapter（因為我們要直接修改 patch_embed）
            self.pvt.channel_adapter = None
            
            # 取得第一層 patch_embed
            first_patch_embed = self.pvt.backbone.patch_embeds[0]
            embed_dim = first_patch_embed.proj.out_channels
            kernel_size = first_patch_embed.proj.kernel_size
            stride = first_patch_embed.proj.stride
            padding = first_patch_embed.proj.padding
            
            # 創建新的 conv2d 層，接受 C_rm 通道
            new_conv = nn.Conv2d(
                self.C_rm, embed_dim, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                bias=False
            )
            
            # 初始化權重：將原始權重平均後複製到 C_rm 通道
            with torch.no_grad():
                w_old = first_patch_embed.proj.weight  # [embed_dim, 3, kernel_h, kernel_w]
                # 對 3 通道求平均，然後重複 C_rm 次
                w_mean = w_old.mean(dim=1, keepdim=True)  # [embed_dim, 1, kernel_h, kernel_w]
                new_conv.weight.copy_(w_mean.repeat(1, self.C_rm, 1, 1))
            
            first_patch_embed.proj = new_conv
            self.rm_to_rgb = None
        else:
            # 方案A：1x1 將 C_rm 壓成 3 通道
            self.rm_to_rgb = nn.Conv2d(self.C_rm, 3, kernel_size=1, bias=False).to(rm_dev)
            self.pvt = PVT(
                in_channels=3,
                out_channels=out_channels,
                input_size=input_size,
                **pvt_kwargs
            )

        # 5) 可選的 ImageNet normalize 參數（若要與預訓練分佈更接近，當作對照）
        self.register_buffer("_im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _normalize_per_sample(feat: torch.Tensor) -> torch.Tensor:
        """對每張圖的每個通道做標準化，讓 RM 特徵分佈更穩定。"""
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (feat - mean) / std

    def _maybe_imagenet_norm(self, x3: torch.Tensor) -> torch.Tensor:
        """可選的 ImageNet 正規化"""
        if self.apply_imagenet_norm:
            return (x3 - self._im_mean.to(x3.device)) / self._im_std.to(x3.device)
        return x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 取 RM 特徵（不 flatten、不經其 fc）
        rm = self.rgb_sfm.RGB_convs(x)                       # [B, C_rm, H, W]
        rm = self._normalize_per_sample(rm)                  # 對齊分佈（關鍵）

        # 2) 對接 PVT
        if self.rm_to_rgb is not None:
            # A 方案：1x1 壓到 3 通道（權重會跟著 .to() 移動）
            rm = self.rm_to_rgb(rm)                          # [B, 3, H, W]
            rm = F.interpolate(rm, size=self.input_size, mode=self.interp_mode, align_corners=False)
            rm = self._maybe_imagenet_norm(rm)               # 可選：做 ImageNet norm 對照
        else:
            # B 方案：直接改 PVT 第一層接 C_rm
            rm = F.interpolate(rm, size=self.input_size, mode=self.interp_mode, align_corners=False)

        # 3) PVT
        output = self.pvt(rm)
        return output

