# VIT_SmallPatch 與 vit_head_mask_analysis.py 使用說明

## 相容性

✅ **已修改 `VIT_SmallPatch` 以與 `vit_head_mask_analysis.py` 相容**

### 主要修改

1. **使用 `nn.MultiheadAttention`**：將自訂的 `MultiHeadAttention` 改為使用 PyTorch 的 `nn.MultiheadAttention`，這樣 `VIT_with_Partial_LRP` 才能自動找到並包裝注意力層。

2. **模型結構相容**：
   - ✅ 有 `norm` 屬性（最後的 LayerNorm）→ `VIT_with_Partial_LRP` 可以找到並 hook
   - ✅ 有 `head` 屬性（分類頭）→ `VIT_with_Partial_LRP` 可以找到
   - ✅ 使用 `nn.MultiheadAttention` → `VIT_with_Partial_LRP` 可以自動包裝

## 使用方法

### 1. 在 config.py 中設定模型

```python
arch = {
    "name": 'VIT_SmallPatch',
    "need_calculate_status": False,
    "args": {
        'in_channels': 3,
        "out_channels": 10,
        "input_size": (224, 224),
        "patch_size": 4,  # 或 2
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    }
}
```

### 2. 直接使用 vit_head_mask_analysis.py

腳本會自動：
- 從 `config.py` 載入模型
- 尋找權重檔案（預設：`runs/train/exp92/VIT_SmallPatch_best.pth`）
- 自動推斷 patch size
- 進行 head masking 分析

### 3. 修改權重檔案路徑（如果需要）

在 `vit_head_mask_analysis.py` 中修改 `_latest_best_checkpoint()` 函數：

```python
def _latest_best_checkpoint():
    # 修改為您的權重檔案路徑
    cand = sorted(glob.glob("runs/train/exp*/VIT_SmallPatch_best.pth"))
    if len(cand) > 0:
        return cand[-1]
    # 或直接指定路徑
    fixed = "runs/train/exp92/VIT_SmallPatch_best.pth"
    return fixed if os.path.exists(fixed) else None
```

## 注意事項

1. **權重載入**：確保權重檔案路徑正確，腳本會自動處理權重載入。

2. **Patch Size 推斷**：`_infer_patch_size()` 函數會自動從模型結構推斷 patch size，用於繪製 patch grid。

3. **記憶體使用**：
   - `patch_size=2` 會產生大量 patches（224×224 → 112×112 = 12,544 patches），需要較多 GPU 記憶體
   - `patch_size=4` 產生中等數量 patches（224×224 → 56×56 = 3,136 patches）

4. **批次大小**：如果遇到記憶體不足，可以減少 `vit_head_mask_analysis.py` 中的批次大小或樣本數量。

## 測試

執行以下命令測試模型是否正常：

```bash
python test_vit_smallpatch.py
```

然後執行分析腳本：

```bash
python vit_head_mask_analysis.py
```

## 輸出

分析結果會保存在 `./head_mask_results/` 目錄下，包括：
- 原始圖像
- Baseline 結果
- 保留 Top-K heads 的結果
- 移除 Top-K heads 的結果
- 三種情境的對照圖

