import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.vision_transformer import ViT_B_16_Weights


class RegisterViT(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, input_size=(224,224), num_registers=1):
        super().__init__()

        self.num_registers = num_registers
        self.input_size = input_size

        # Channel adapter if needed
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # Load base ViT-B/16 from torchvision
        image_size = input_size[0]
        weights = ViT_B_16_Weights.IMAGENET1K_V1

        try:
            backbone = tv_models.vit_b_16(weights=weights, image_size=image_size)
        except TypeError:
            backbone = tv_models.vit_b_16(weights=weights)

        # Replace head
        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Linear(in_features, out_channels)

        self.backbone = backbone

        embed_dim = self.backbone.hidden_dim

        # ★ Register Tokens (learnable)
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, embed_dim))

        # ★ Register positional embedding
        self.register_pos = nn.Parameter(torch.randn(1, num_registers, embed_dim))

        # ★ Extend positional embedding (CLS + patches + R registers)
        old_pos_embed = self.backbone.encoder.pos_embedding   # [1, 197, C]
        old_len = old_pos_embed.shape[1]

        new_pos = torch.randn(1, old_len + num_registers, embed_dim)
        new_pos[:, :old_len, :] = old_pos_embed
        new_pos[:, old_len:, :] = self.register_pos
        self.backbone.encoder.pos_embedding = nn.Parameter(new_pos)

    def forward(self, x):
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        backbone = self.backbone
        B = x.shape[0]

        # 1. Only patch tokens (NO CLS)
        x = backbone._process_input(x)        # [B, 196, C]

        # 2. Add CLS token manually (your version needs this)
        cls_tok = backbone.class_token.expand(B, -1, -1)  # [B,1,C]
        x = torch.cat([cls_tok, x], dim=1)                # now [B,197,C]

        # print("step1 patch+cls:", x.shape)

        # 3. Append register tokens
        if self.num_registers > 0:
            reg = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([x, reg], dim=1)                # [B,197+R,C]

        # print("step2 after reg:", x.shape)
        # print("step2 pos:", backbone.encoder.pos_embedding.shape)

        # 4. Encoder (will add pos_embedding automatically)
        x = backbone.encoder(x)

        x = backbone.encoder.ln(x)
        cls = x[:, 0]
        return backbone.heads(cls)



