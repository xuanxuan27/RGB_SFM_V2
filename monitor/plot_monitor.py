import matplotlib.pyplot as plt
import os
import numpy as np

from config import arch
from monitor.calculate_stats import calculate_RM, get_stats

def plot_channel_histograms(raw, plot_shape=(5, 6), save_dir='.', layer_num='layer', xlim=(0, 1), space_count=5):
    """
    繪製每個通道的直方圖，使用熱圖色彩，以自定義區間等分，不顯示文字標題

    參數:
    - raw: 原始數據張量
    - plot_shape: 子圖網格形狀，默認為 (5, 6)
    - save_dir: 保存目錄，默認為當前目錄
    - save_file: 保存文件名，默認為 'channel_histograms.png'
    - xlim: x軸範圍，默認為 (0, 1)
    - space_count: x 等分數，默認為 5
    """

    # 準備數據
    data = raw.detach().numpy()  # 使用 detach() 方法
    num_channels = data.shape[0]

    # 創建圖形，預留頂部空間
    plt.figure(figsize=(20, 17))  # 略微增加高度以容納標題
    # plt.suptitle(f"{layer_num} x range : {xlim}, space: {space_count}", fontsize=32, fontweight='bold', y=0.98)

    # 創建分區（5等分）
    bins = np.linspace(xlim[0], xlim[1], space_count + 1)

    # 創建子圖網格
    rows, cols = plot_shape
    for i in range(min(num_channels, rows * cols)):
        plt.subplot(rows, cols, i + 1)

        # 使用熱圖色彩映射，依據 bin 的位置變化顏色
        n, bins_edges, patches = plt.hist(data[i], bins=bins, edgecolor='black')

        # 為每個 bin 設置漸變顏色
        fracs = (bins_edges[:-1] + bins_edges[1:]) / 2
        norm = plt.Normalize(xlim[0], xlim[1])
        for frac, patch in zip(fracs, patches):
            color = plt.cm.viridis(norm(frac))
            patch.set_facecolor(color)

        plt.xlim(xlim[0], xlim[1])  # x軸範圍固定在指定區間
        # plt.xticks([])  # 移除x軸刻度標籤
        # plt.yticks([])  # 移除y軸刻度標籤

    # 調整子圖間距
    plt.tight_layout()

    # 確保保存目錄存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存圖像
    full_path = os.path.join(save_dir, f'{layer_num}_{space_count}_channel_histograms.png')
    plt.savefig(full_path, dpi=300)  # 提高分辨率
    plt.close()  # 關閉圖形以釋放內存

    print(f"Histogram saved to {full_path}")


def plot_layer_graph(model, layers, layer_num, images, is_gray=False, plot_shape=None, save_dir='./output', space_count = 10):
    if is_gray:
        input_images = model.gray_transform(images)
    else:
        input_images = images

    raw = calculate_RM(layers[layer_num], input_images)
    stats, global_stats = get_stats(raw)

    if plot_shape is None:
        plot_shape = (int(raw.shape[0] ** 0.5), int(raw.shape[0] ** 0.5))

    xlim = (global_stats['total_min'], global_stats['total_max'])

    plot_channel_histograms(raw, plot_shape=plot_shape, save_dir=save_dir, layer_num=layer_num, xlim=xlim,
                            space_count=space_count)


def plot_all_layers_graph(model, rgb_layers, gray_layers, images, save_dir='./output', space_count=10):
    # 使否使用輪廓層
    mode = arch['args']['mode']
    use_gray = mode in ['gray', 'both']
    channels = arch['args']['channels']
    for key, layer in rgb_layers.items():
        print(f"plotting {key} graph")
        index = int(key.split('_')[-1])
        plot_shape = channels[0][index]

        plot_layer_graph(model, rgb_layers, key, images, is_gray = False, plot_shape = plot_shape, save_dir = save_dir, space_count = space_count)

    if use_gray:
        for key, layer in gray_layers.items():
            print(f"plotting {key} graph")
            index = int(key.split('_')[-1])
            plot_shape = channels[1][index]

            plot_layer_graph(model, gray_layers, key, images, is_gray=True, plot_shape=plot_shape, save_dir=save_dir,
                             space_count=space_count)



