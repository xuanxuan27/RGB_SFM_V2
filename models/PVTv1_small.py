import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

# 建立 PVTv1 Small 模型
class PVTv1_small(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, input_size=(224, 224)):
        super(PVTv1_small, self).__init__()
        self.model = timm.create_model('twins_pcpvt_small', pretrained=True, num_classes=10)
        self.model.head = nn.Linear(self.model.head.in_features, out_channels)
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
    def forward(self, x, return_features=False):
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        if not return_features:
            return self.model(x)

def main():
    model = PVTv1_small()
    # print(model)
    for name, param in model.named_parameters():
        print(f"參數: {name}, 形狀: {param.shape}")

if __name__ == "__main__":
    main()