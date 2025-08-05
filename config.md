# 📄 模型設定說明 (`config.py`)

## 🧰 基本資訊

## 🧰 基本資訊

| 參數名稱 | 說明                       |
|----------|--------------------------|
| `project` | 在 wandb 上的專案名稱           |
| `name` | 實驗名稱                     |
| `group` | 日期                       |
| `tags` | 模型標籤                     |
| `description` | 實驗描述備註                   |
| `device` | 自動偵測是否可用 GPU             |
| `load_model_name` | 欲載入的預訓練模型名稱，以及訓練完最佳模型儲存處 |

## 🧠 模型架構（`arch`）

### RGB_SFMCNN_V2

| 參數 | 說明                                                                                                                                                                         |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name` | 模型名稱                                                                                                                                                                       |
| `need_calculate_status` | 是否需要計算 RM 分布指標                                                                                                                                                             |
| `in_channels` | 輸入通道數（RGB）                                                                                                                                                                 |
| `out_channels` | 分類數                                                                                                                                                                        |
| `mode` | 模型輸入模態：<br> • `rgb`: 僅使用 RGB 通道<br> • `gray`: 僅使用灰階通道<br> • `both`: 同時使用 RGB 與灰階                                                                                           |
| `Conv2d_kernel` | 各層卷積核大小，格式為 `[[(w,h),...], [(w,h),...]]`，分別對應 RGB 與 Gray 分支                                                                                                                |
| `SFM_methods` | 空間合併方法：<br> • `alpha_mean`: 加權平均合併<br> • `max`: 取最大值<br>                                                                                                                   |
| `SFM_filters` | 每層的 SFM 濾波器大小，例如 `(2,2)`                                                                                                                                                   |
| `channels` | 各層濾波器通道數量，最內層 (h, w) 對應繪圖時濾波器排列方式，而外層分別對應 RGB 與 Gray 分支                                                                                                                    |
| `strides` | 每層卷積的步長設定                                                                                                                                                                  |
| `paddings` | 卷積 padding 設定                                                                                                                                                              |
| `color_filter` | 色彩濾波器組類型：<br> • `new`: CIELAB空間均勻分布<br> • `old`: PCCS 色環                                                                                                                   |
| `conv_method` | 卷積相似度計算方式：<br> • `cosine`: 餘弦相似度<br> • `cdist`: 歐氏距離<br> • `dot_product`: 內積<br> • `squared_cdist`: 平方距離<br>                                                               |
| `initial` | 權重初始化方式：<br> • `kaiming`: He 初始化<br> • `uniform`: 均勻初始化<br>                                                                                                                |
| `rbfs` | RBF 類型與激活函數：<br>  • `triangle`: 三角形函數<br> • `gauss`: 高斯函數<br> • `sigmoid`: sigmoid<br> • `cReLU`: cReLU<br> • `cReLU_percent`: cReLU_percent <br> • `regularization`: 正規化  |
| `activate_params` | 各層 RBF 激活參數，格式為 `[[[w,p],...], [[w,p],...]]`，分別對應 RGB 與 Gray 分支 ，<br>`[w,p]` 對應 <br>•`w`: 三角形函數、高斯函數寬度，<br>•`p`:  cReLU_percent 篩選 %                                       |
| `fc_input` | 全連接層的輸入維度，大小為 (rbg last channels + gray last channels) * last layer shape                                                                                                                                                        |
| `device` | 使用設備                                                                                                                                                                       |



### 其他模型
- 主要準確率比較對象，包含 'ResNet' 'AlexNet' 'DenseNet' 'GoogLeNet'


## 🧪 訓練參數

| 參數                       | 說明                                                                                     |
|--------------------------|----------------------------------------------------------------------------------------|
| `save_dir`               | 模型儲存資料夾                                                                                |
| `plot_bar`               | 畫可解釋性圖用，是否繪製數值的 Bar                                                                    |
| `plot_CAM`               | plot_example_V2 中，是否使用 Grad_CAM 進行篩選                                                   |
| `dataset`                | 使用的資料集，例如 `Colored_MNIST`, `MultiColor_Shapes_Database` ，詳細可見`get_dataloader.py`       |
| `input_shape`            | 輸入影像的尺寸                                                                                |
| `batch_size`             | 訓練批次大小                                                                                 |
| `epoch`                  | 訓練回合數                                                                                  |
| `early_stop`             | 是否啟用 Early Stopping                                                                    |
| `patience`               | Early Stop 容忍次數                                                                        |
| `lr`                     | 初始學習率                                                                                  |
| `lr_scheduler`           | 學習率調整策略（如 `ReduceLROnPlateau`）                                                         |
| `optimizer`              | 使用的優化器（如 `Adam`, `SGD`）                                                                |
| `loss_fn`                | 驗證用損失函數，如 `CrossEntropyLoss`                                                           |
| `training_loss_fn`       | 訓練用損失函數，用於客製化 Loss Function（如 `MetricBaseLoss` 為針對理想濾波器分布的損失函數），詳細可見`loss_function.py` |
| `use_metric_based_loss`  | 訓練時是否使用 `training_loss_fn`，否則用 `loss_fn`                                               |
| `use_preprocessed_image` | 是否使用預先處理過的影像，主要用於視網膜資料集(RetinaMNIST224)前處理                                             |

## 🫀 心臟鈣化設定（`heart_calcification`）

| 參數 | 說明 |
|------|------|
| `grid_size` | 切割的網格大小 |
| `need_resize_height` | 是否依圖片高度 resize |
| `resize_height` | resize 高度 |
| `threshold` | 檢測信心閾值 |
| `enhance_method` | 增強方式（contrast、clahe 等） |
| `contrast_factor` | 對比係數 |
| `use_vessel_mask` | 是否使用血管遮罩 |
| `use_min_count` | 是否根據最小鈣化點數進行篩選 |
| `augment_positive` | 是否增強陽性樣本 |
| `augment_multiplier` | 陽性樣本增強倍數 |