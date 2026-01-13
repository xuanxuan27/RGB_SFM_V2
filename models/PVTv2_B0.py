import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class PVTv2_B0(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, input_size=(224, 224)):
        super(PVTv2_B0, self).__init__()
        self.input_size = input_size
        # 建立 PVT v2-B0 模型並載入 ImageNet 預訓練權重
        self.model = timm.create_model('pvt_v2_b0', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, out_channels)
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
    def forward(self, x, return_features=False):
        # 1. 通道適配與尺寸調整
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        # 2. 如果不需要特徵圖，直接執行原有的 forward
        if not return_features:
            return self.model(x)

        # 3. 如果需要特徵圖，手動走過四個階段 [cite: 213, 954]
        stage_outputs = []
        
        # 初始 Patch Embedding (Stage 1 開始) [cite: 822, 1024]
        x = self.model.patch_embed(x)
        
        # 遍歷四個 Stage [cite: 825, 851, 884, 914]
        for stage in self.model.stages:
            x = stage(x)
            stage_outputs.append(x)
            
        # 最後的分類輸出 [cite: 296]
        out = self.model.norm(x)
        out = self.model.head(out)
        
        return out, stage_outputs


if __name__ == "__main__":
    model = PVTv2_B0(in_channels=3, out_channels=10, input_size=(224, 224))
    print(model)
    
