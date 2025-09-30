import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.vision_transformer import ViT_B_16_Weights


class VIT(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10, input_size=(224, 224)):
        super().__init__()

        self.input_size = input_size

        # Simple adapter to 3-channel input if needed
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # Build a torchvision ViT (vit_b_16). Use default patch size 16.
        # Note: image_size is set from input_size[0] to keep square assumption.
        image_size = input_size[0]
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        try:
            backbone = tv_models.vit_b_16(weights=weights, image_size=image_size)
        except TypeError:
            # Fallback for older torchvision where image_size arg is not exposed
            backbone = tv_models.vit_b_16(weights=weights)

        # Replace classification head to match out_channels
        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Linear(in_features, out_channels)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapt channels if needed
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return self.backbone(x)


