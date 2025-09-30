# Partial LRP Vision Transformer (ViT)

這個專案實現了一個帶有Partial Layer-wise Relevance Propagation (LRP) 機制的Vision Transformer，用於提供模型的可解釋性分析。

## 功能特點

### 1. Partial LRP機制
- **選擇性層分析**: 可以選擇性地對特定層應用LRP，而不是所有層
- **多種LRP規則**: 支援alpha-beta、epsilon和gamma規則
- **靈活配置**: 可以自定義LRP參數和目標層

### 2. 支援的LRP規則

#### Alpha-Beta規則
```python
# 正相關性: R+ = α * (a+ * R+) / (Σa+ + ε)
# 負相關性: R- = β * (a- * R-) / (Σa- + ε)
# 總相關性: R = R+ - R-
```

#### Epsilon規則
```python
# R = (a * R) / (Σa + ε)
```

#### Gamma規則
```python
# 類似於epsilon規則，但對正負值有不同處理
```

### 3. 可視化功能
- 生成相關性熱力圖
- 支援保存視覺化結果
- 提供統計分析功能

## 安裝需求

```bash
torch >= 1.9.0
torchvision >= 0.10.0
matplotlib >= 3.3.0
numpy >= 1.19.0
```

## 使用方法

### 基本使用

```python
from models.VIT_with_Partial_LRP import PartialLRPViT

# 創建模型
model = PartialLRPViT(
    in_channels=3,
    out_channels=10,
    input_size=(224, 224),
    lrp_layers=['encoder_layer_0', 'patch_embedding'],  # 選擇要應用LRP的層
    lrp_rule="alpha_beta",  # LRP規則
    alpha=1.0,
    beta=0.0,
    epsilon=1e-6
)

# 前向傳播
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

# 計算LRP相關性
relevance = model.compute_lrp_relevance(input_tensor, target_class=0)
```

### 高級使用

```python
# 獲取所有層的相關性
all_relevance = model.compute_lrp_relevance(
    input_tensor, 
    target_class=0, 
    return_intermediate=True
)

# 視覺化相關性
relevance_vis = model.visualize_relevance(
    input_tensor,
    target_class=0,
    save_path="relevance_heatmap.png"
)
```

## 參數說明

### 模型參數
- `in_channels`: 輸入通道數 (預設: 3)
- `out_channels`: 輸出類別數 (預設: 10)
- `input_size`: 輸入圖像尺寸 (預設: (224, 224))

### LRP參數
- `lrp_layers`: 要應用LRP的層名稱列表
  - `'patch_embedding'`: Patch embedding層
  - `'encoder_layer_0'`, `'encoder_layer_1'`, ...: Transformer encoder層
  - `'classifier'`: 分類頭
- `lrp_rule`: LRP規則 ("alpha_beta", "epsilon", "gamma")
- `alpha`: Alpha-beta規則中的alpha參數 (預設: 1.0)
- `beta`: Alpha-beta規則中的beta參數 (預設: 0.0)
- `epsilon`: Epsilon規則中的epsilon參數 (預設: 1e-6)

## 可用的層名稱

在ViT架構中，以下層可以應用LRP：

1. **Patch Embedding層**
   - `'patch_embedding'`: 將圖像patches轉換為embeddings

2. **Transformer Encoder層**
   - `'encoder_layer_0'`: 第一個encoder層
   - `'encoder_layer_1'`: 第二個encoder層
   - `'encoder_layer_2'`: 第三個encoder層
   - ... (依此類推，通常有12層)

3. **分類頭**
   - `'classifier'`: 最終的分類層

## 使用範例

### 範例1: 基本相關性分析

```python
import torch
from models.VIT_with_Partial_LRP import PartialLRPViT

# 創建模型
model = PartialLRPViT(
    in_channels=3,
    out_channels=10,
    lrp_layers=['encoder_layer_0', 'patch_embedding']
)

# 測試數據
input_tensor = torch.randn(1, 3, 224, 224)

# 計算相關性
relevance = model.compute_lrp_relevance(input_tensor, target_class=0)
print(f"相關性形狀: {relevance.shape}")
```

### 範例2: 比較不同LRP規則

```python
# 創建不同規則的模型
rules = ["alpha_beta", "epsilon", "gamma"]
relevance_results = {}

for rule in rules:
    model = PartialLRPViT(
        in_channels=3,
        out_channels=10,
        lrp_rule=rule
    )
    
    relevance = model.compute_lrp_relevance(input_tensor, target_class=0)
    relevance_results[rule] = relevance

# 比較結果
for rule, relevance in relevance_results.items():
    print(f"{rule}: 平均相關性 = {torch.abs(relevance).mean().item():.6f}")
```

### 範例3: 層級分析

```python
# 獲取所有層的相關性
all_relevance = model.compute_lrp_relevance(
    input_tensor, 
    return_intermediate=True
)

# 分析每層的相關性
for layer_name, relevance in all_relevance.items():
    if isinstance(relevance, torch.Tensor):
        relevance_abs = torch.abs(relevance)
        print(f"{layer_name}:")
        print(f"  形狀: {relevance.shape}")
        print(f"  平均相關性: {relevance_abs.mean().item():.6f}")
        print(f"  最大相關性: {relevance_abs.max().item():.6f}")
```

## 視覺化

模型提供了內建的視覺化功能：

```python
# 生成相關性熱力圖
relevance_vis = model.visualize_relevance(
    input_tensor,
    target_class=0,
    save_path="relevance_analysis.png"
)
```

這會生成一個包含原始圖像和相關性熱力圖的對比圖。

## 與原始ViT的比較

```python
from models.VIT import VIT
from models.VIT_with_Partial_LRP import PartialLRPViT

# 創建兩個模型
original_vit = VIT(in_channels=3, out_channels=10)
lrp_vit = PartialLRPViT(in_channels=3, out_channels=10)

# 比較預測結果
input_tensor = torch.randn(1, 3, 224, 224)

original_output = original_vit(input_tensor)
lrp_output = lrp_vit(input_tensor)

print(f"預測差異: {torch.abs(original_output - lrp_output).max().item():.6f}")
```

## 注意事項

1. **記憶體使用**: LRP計算需要額外的記憶體來儲存中間結果
2. **計算時間**: LRP計算會增加推理時間
3. **層選擇**: 選擇太多層進行LRP分析可能會導致計算複雜度過高
4. **數值穩定性**: 使用epsilon參數來避免除零錯誤

## 故障排除

### 常見問題

1. **記憶體不足**
   - 減少batch size
   - 選擇較少的LRP層
   - 使用較小的輸入圖像尺寸

2. **相關性為零**
   - 檢查目標類別是否正確
   - 確認LRP層配置是否合理
   - 調整epsilon參數

3. **視覺化問題**
   - 確保matplotlib已正確安裝
   - 檢查保存路徑的權限

## 參考文獻

1. Bach, S., et al. "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
2. Montavon, G., et al. "Layer-wise relevance propagation: an overview." Explainable AI: interpreting, explaining and visualizing deep learning. Springer, 2019. 193-209.

## 授權

此專案遵循與主專案相同的授權條款。
