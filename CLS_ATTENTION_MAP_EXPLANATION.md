# CLS Attention Map 原理說明

## 1. 什麼是 CLS Attention Map？

CLS Attention Map 是視覺化 **CLS token（分類 token）在最後一層 Transformer 中「關注」哪些圖像區域**的熱圖。

## 2. Vision Transformer (ViT) 的基本結構

### Token 序列
當一張圖像輸入 ViT 時，會被切成多個 patch（例如 16×16 的 patch），然後：

```
輸入圖像 → [CLS token, patch_1, patch_2, ..., patch_N]
```

- **CLS token**：一個特殊的 token，位於序列最前面（index 0）
- **patch tokens**：對應圖像的每個 patch（index 1 到 N）

### Attention Mechanism（注意力機制）

在每個 Transformer layer 中，每個 token 都會計算它對**所有其他 token** 的注意力分數：

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

產生的 attention matrix 形狀是 `[B, H, N, N]`：
- `B`：batch size
- `H`：head 數量（多頭注意力）
- `N`：token 數量（1 個 CLS + N 個 patches）
- 最後兩個維度 `[N, N]`：每個 token 對其他 token 的注意力分數

## 3. Attention Matrix 的結構

假設有 197 個 tokens（1 個 CLS + 196 個 patches = 14×14 grid）：

```
Attention Matrix [N, N] = 
    CLS   patch_1  patch_2  ...  patch_196
CLS   [   0.05     0.02     ...    0.01   ]  ← CLS 對各 patch 的注意力
patch_1 [ 0.10     0.15     ...    0.08   ]
patch_2 [ 0.08     0.12     ...    0.10   ]
...
patch_196 [ 0.03   0.05     ...    0.20   ]
```

**重點**：第一行 `A[0, :]` 就是 **CLS token 對所有 token 的注意力分數**。

## 4. 為什麼要看「最後一層」的 CLS Attention？

### 層級演進
- **第一層**：CLS token 剛開始學習，注意力可能較分散
- **中間層**：逐漸聚焦到相關區域
- **最後一層**：CLS token 已經整合了所有資訊，準備做分類決策

**最後一層的 CLS attention** 最能反映「模型在做分類決策時，最關注圖像的哪些區域」。

## 5. 程式碼解析

讓我們看看 `get_cls_attention_map()` 的實作：

```python
# 步驟 1：取得最後一層的 attention matrix
A = last['attn_probs']  # [B, H, N, N]

# 步驟 2：提取 CLS → patches 的注意力
# A[:, :, 0, 1:] 意思是：
#   - 第 0 個 token（CLS）作為 query
#   - 第 1 到 N-1 個 token（所有 patches）作為 key
cls_attn = A[:, :, 0, 1:]  # [B, H, N_patches]

# 步驟 3：多頭平均
# 因為有多個 head，每個 head 可能關注不同面向
# 平均後得到整體的注意力分佈
cls_attn = cls_attn.mean(dim=1)  # [B, N_patches]

# 步驟 4：reshape 成 patch grid
# 假設 196 個 patches = 14×14 grid
grid_h = grid_w = int(cls_attn.shape[-1] ** 0.5)  # 14
attn_map = cls_attn.view(B, 1, grid_h, grid_w)  # [B, 1, 14, 14]

# 步驟 5：上採樣到原始圖像大小
# 從 14×14 放大到 224×224（或輸入圖像大小）
attn_map = F.interpolate(attn_map, size=self.input_size, mode='bicubic')

# 步驟 6：正規化到 [0, 1]
attn_map = attn_map / (attn_map.max() + 1e-6)
```

## 6. 視覺化意義

### 熱圖解讀
- **紅色/亮色區域**：CLS token 高度關注的區域（對分類決策重要）
- **藍色/暗色區域**：CLS token 較少關注的區域（對分類決策較不重要）

### 實際應用
1. **模型解釋性**：了解模型在做分類時「看」哪裡
2. **錯誤分析**：如果模型分類錯誤，可以看它關注了錯誤的區域
3. **資料品質檢查**：檢查模型是否關注到正確的物體特徵

## 7. CLS Attention vs LRP Relevance 的差異

| 特性 | CLS Attention | LRP Relevance |
|------|---------------|---------------|
| **計算方式** | Forward pass 的 attention 分數 | Backward pass 的 relevance 傳播 |
| **方向** | CLS → Patches（單向） | Output → Input（反向傳播） |
| **意義** | 「CLS 關注哪些區域」 | 「哪些區域對輸出有貢獻」 |
| **計算成本** | 低（只需一次 forward） | 高（需要完整的 LRP 計算） |

### 為什麼兩者可能不同？
- **Attention**：反映「資訊流動的方向」（CLS 從哪裡獲取資訊）
- **LRP Relevance**：反映「貢獻度」（哪些區域對最終決策有貢獻）

兩者互補，可以一起使用來更全面地理解模型行為。

## 8. 範例：一張貓的圖片

假設輸入一張貓的圖片：

```
原圖：一隻貓坐在草地上
     [貓的臉] [貓的身體] [草地] [背景]
```

**CLS Attention Map 可能顯示**：
- 高亮：貓的臉、貓的眼睛、貓的耳朵
- 中亮：貓的身體
- 低亮：草地、背景

這表示模型在做「貓」的分類時，主要關注貓的臉部特徵。

## 9. 注意事項

1. **多頭平均**：程式碼使用 `mean(dim=1)` 平均所有 head，也可以選擇只顯示特定 head
2. **正規化方式**：目前使用 `max()` 正規化，也可以使用其他方式（如 percentile）
3. **層級選擇**：目前只看最後一層，也可以視覺化中間層來觀察演進過程

## 10. 參考資料

這個方法源自 Vision Transformer 的經典論文：
- **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** (ViT, ICLR 2021)
- 論文中使用 CLS attention 來視覺化模型的關注區域

