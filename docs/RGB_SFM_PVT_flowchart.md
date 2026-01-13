## RGB_SFM_PVT 流程圖（使用 RGB_SFMCNN_V3 → PVT）

> 說明：先以可解釋 RM（RGB_SFMCNN_V3）抽特徵，再銜接 PVT（兩種串接方案）。

```mermaid
flowchart TD
    %% Inputs
    X["輸入影像 x<br/>shape: B×3×H×W"]

    %% RM branch (RGB_SFMCNN_V3)
    subgraph RM["RGB_SFMCNN_V3<br/>僅 RGB 分支"]
      direction LR
      S0["Stage 0<br/>Conv/RBF/SFM"]
      S1["Stage 1<br/>Conv/RBF/SFM"]
      S2["Stage 2<br/>Conv/RBF/SFM<br/>可能非對稱 SFM (1,3)"]
      S3["Stage 3<br/>Conv/RBF/SFM"]
      S0 --> S1 --> S2 --> S3
    end

    X --> RM
    RM --> RMOut["rm 特徵圖<br/>shape: B×C_rm×H_rm×W_rm"]
    RMOut --> Norm["逐樣本逐通道標準化<br/>Z-Score per H,W"]

    %% Branching to PVT
    subgraph ToPVT["對接 PVT 的兩種方案"]
      direction TB
      A["方案 A：1×1 將 C_rm→3"]
      B["方案 B：修改 PVT 第一層<br/>in_channels=C_rm"]
    end

    Norm --> A
    Norm --> B

    %% A path: 1x1 to 3ch → resize → (optional ImageNet Norm) → PVT
    A --> A1["1×1 Conv<br/>C_rm → 3"]
    A1 --> A2["Resize 到 input_size"]
    A2 -->|可選| A3["ImageNet Normalize"]
    A3 --> PVTInA["送入 PVT"]
    A2 --> PVTInA

    %% B path: modify patch_embed → resize → PVT
    B --> B1["Resize 到 input_size"]
    B1 --> B2["修改 PVT 第一層<br/>patch_embed 輸入通道=C_rm"]
    B2 --> PVTInB["送入 PVT"]

    %% PVT Backbone
    subgraph PVT["PVTv2 Backbone"]
      direction LR
      P0["Stage 1<br/>Patch Embedding → Blocks"]
      P1["Stage 2<br/>Patch Merging → Blocks"]
      P2["Stage 3<br/>Patch Merging → Blocks"]
      P3["Stage 4<br/>Patch Merging → Blocks"]
      P0 --> P1 --> P2 --> P3
    end

    PVTInA --> PVT
    PVTInB --> PVT

    PVT --> LastSeq["最後序列 last_seq<br/>含 H_4, W_4"]
    LastSeq --> Map["重排為特徵圖<br/>shape: B×C_4×H_4×W_4"]
    Map --> GAP["Global Avg Pool"]
    GAP --> Head["線性分類頭"]
    Head --> Logits["分類 logits"]
```

### 重點
- SFM 的空間合併僅發生在 RM（RGB_SFMCNN_V3）內，可能導致 `H_rm`、`W_rm` 不等（長方形）。
- 進入 PVT 前：
  - 方案 A：1×1 壓到 3 通道後 resize（可選 ImageNet Norm）。
  - 方案 B：改 PVT 第一層 `patch_embed` 的輸入通道為 `C_rm`，直接吃 RM 特徵（仍需 resize）。
- PVT 內部的 patch merging 保持原版 PVTv2 設計不變（與 SFM 無衝突）。

### 與可視化對應
- `pvt_forward_response_plot.py` 會輸出：
  - RM 各 Stage 的特徵熱圖（均值與若干通道）。
  - PVT 各 Stage 的特徵熱圖（均值、通道格子、通道統計）。
  - 最終預測與 Ground Truth，以及原圖。


