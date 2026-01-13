import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
from typing import Optional, Tuple

# ==========================================
# 1. 模型定義 (基於你原本的代碼，保持不變)
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        if isinstance(img_size, (int, float)):
            img_size = (int(img_size), int(img_size))
        H, W = img_size
        
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.patch_shape = (H // patch_size, W // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path)"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VIT_SmallPatch_ver2(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, input_size=(64, 64), patch_size=4, 
                 embed_dim=384, depth=9, num_heads=12, mlp_ratio=4., 
                 dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        
        # 針對 CIFAR10 調整預設參數：
        # embed_dim 從 768 降到 384 (減少參數量)
        # depth 從 12 降到 9
        
        if isinstance(input_size, int): input_size = (input_size, input_size)
        self.input_size = input_size
        
        self.patch_embed = PatchEmbedding(input_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_fn)

    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# ==========================================
# 2. Mixup 工具函數 (抗 Overfitting 關鍵)
# ==========================================

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 3. 訓練與驗證流程
# ==========================================

def main():
    # 設定
    BATCH_SIZE = 128
    EPOCHS = 100  # 建議至少 100，因為用了 Mixup 收斂會變慢
    LR = 1e-3
    WEIGHT_DECAY = 0.05  # AdamW 關鍵參數
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 數據準備 (Data Augmentation)
    print("Preparing Data...")
    
    # 訓練集增強：AutoAugment + Normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 強力增強
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 測試集：僅 Normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 模型初始化
    # 注意：input_size 設定為 (64, 64) 配合 CIFAR10
    # drop_path 設定為 0.1 或 0.2 增加正則化
    print("Creating Model...")
    model = VIT_SmallPatch_ver2(
        in_channels=3, 
        out_channels=10, 
        input_size=(64, 64), 
        patch_size=4,       # 64/4 = 16x16 = 256 patches (長度適中)
        embed_dim=384,      # 稍微縮小維度以適應小數據集
        depth=9,            # 稍微減少層數
        num_heads=12, 
        drop_path=0.2,      # 增加 stochastic depth
        dropout=0.1         # 增加 dropout
    ).to(device)

    # 3. 優化器與 Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Warmup + Cosine Scheduler
    # 前 5 epochs 線性增加 LR，之後 Cosine 衰減
    warmup_epochs = 5
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    # 4. 訓練迴圈
    print("Start Training...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- Mixup ---
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 使用 Mixup loss
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            # 梯度裁減 (防止梯度爆炸)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            
            # Mixup 下計算準確率比較複雜，這裡僅做參考
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

        # Update Scheduler
        scheduler.step()

        # 5. 驗證 (不使用 Mixup)
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        if acc > best_acc:
            best_acc = acc
            # torch.save(model.state_dict(), 'vit_cifar10_best.pth')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {train_loss/len(trainloader):.4f} | "
              f"Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

if __name__ == '__main__':
    main()