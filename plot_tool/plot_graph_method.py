import numpy as np
from matplotlib import pyplot as plt
import torch
from PIL import Image

from config import config


def tensor_to_numpy_image(tensor, save_path=None):
    """
    將形狀為 (C, H, W) 的 PyTorch tensor 轉成 numpy 圖片 (H, W, C)，支援儲存。
    若為灰階 (C=1)，自動 squeeze 成 (H, W)。
    """
    tensor = tensor.detach().cpu()
    if tensor.shape[0] == 1:  # 灰階
        np_img = tensor.squeeze().numpy() * 255
        mode = 'L'
    else:  # RGB
        np_img = tensor.permute(1, 2, 0).numpy() * 255
        mode = 'RGBA' if np_img.shape[2] == 4 else 'RGB'

    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img, mode=mode)

    if save_path:
        pil_img.save(save_path)

    return np.array(pil_img)  # 傳回 numpy 格式給 combine 使用

def plot_RM_then_save(layers, model, layer_num, plot_shape, img, save_path, is_gray = False, figs = None):
    fig = plot_RM_map(layers, model, layer_num, plot_shape, img, save_path, is_gray)

    if figs is not None:
        figs[layer_num] = fig

# 繪製反應 RM 圖
def plot_RM_map(layers,model,layer_num, plot_shape, img, save_path, is_gray = False):

    if is_gray:
        RM = layers[layer_num](model.gray_transform(img.unsqueeze(0)))[0]
    else:
        RM = layers[layer_num](img.unsqueeze(0))[0]

    # print(f"Layer{layer_num}_RM: {RM.shape}")

    RM_H, RM_W = RM.shape[1], RM.shape[2]
    return plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
             path=save_path + f'{layer_num}_RM')


def plot_map(rm, path=None, padding=1, pad_value=0.0, return_type="image", **kwargs):
    """
    繪製濾波器圖像的網格視覺化。

    參數:
        rm (torch.Tensor 或 numpy.ndarray): 五維張量，shape = (num_filter_rows, num_filter_cols, filter_height, filter_width, num_channels)

            - num_filter_rows: 濾波器排列的行數（可視為 filter map 的排版高度）
            - num_filter_cols: 濾波器排列的列數（可視為 filter map 的排版寬度）
            - filter_height:   每個濾波器圖像的實際高度（像素）
            - filter_width:    每個濾波器圖像的實際寬度（像素）
            - num_channels:    通道數（1 表示灰階，3 表示 RGB 彩色）

        grid_size (tuple): matplotlib subplot 的總 grid 尺寸。
        rowspan (int): 單個圖佔用的 grid row 數。
        colspan (int): 單個圖佔用的 grid col 數。
        path (str): 若指定路徑，則儲存圖片至該路徑；否則顯示圖片。
        **kwargs: 傳給 imshow() 的額外參數，如 cmap。

    回傳:
        fig: matplotlib 的 figure 物件。
    """
    plot_bar = config["plot_bar"]
    print(f"plot_bar {plot_bar}")
    dip = 200

    if isinstance(rm, torch.Tensor):
        rm = rm.cpu().numpy()

    rows, cols, h, w, c = rm.shape
    global_min = rm.min()
    global_max = rm.max()

    # 計算整張大圖的尺寸（包含 padding）
    H = rows * h + (rows - 1) * padding
    W = cols * w + (cols - 1) * padding

    # 建立白色背景的大畫布
    if c == 1:
        canvas = np.ones((H, W), dtype=rm.dtype)
        for row in range(rows):
            for col in range(cols):
                top = row * (h + padding)
                left = col * (w + padding)
                canvas[top:top + h, left:left + w] = rm[row, col, :, :, 0]
    else:
        canvas = np.ones((H, W, c), dtype=rm.dtype)
        for row in range(rows):
            for col in range(cols):
                top = row * (h + padding)
                left = col * (w + padding)
                canvas[top:top + h, left:left + w, :] = rm[row, col]

    # 根據圖片大小計算合適的色條文字大小
    label_fontsize = max(6, int(min(H, W) / 30))

    fig, ax = plt.subplots(figsize=(W / 40, H / 40), facecolor="white", dpi=dip)
    im = ax.imshow(canvas, vmin=global_min, vmax=global_max, **kwargs)
    ax.axis('off')

    if plot_bar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=label_fontsize)

    if path:
        plt.savefig(path, dpi=dip, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_combine_images(figs, save_path=None, spacing=0.05,
                        fixed_width=5, fixed_height=5,
                        show=False, title="Combined Images"):
    num_images = len(figs)
    # 計算整張圖的大小：每張圖 fixed_width，高度固定，寬度加上間距
    fig_width = num_images * fixed_width + (num_images - 1) * spacing
    fig_height = fixed_height

    # 先關閉 constrained_layout，改用 tight_layout + subplots_adjust
    fig, axes = plt.subplots(
        1, num_images,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1] * num_images}
    )

    if num_images == 1:
        axes = [axes]

    for ax, (key, fig_source) in zip(axes, figs.items()):
        ax.imshow(fig_source.canvas.buffer_rgba())
        ax.axis('off')

    # 調整左右上下邊界為 0，並設定圖間距
    fig.subplots_adjust(
        left=0,      # 左邊貼齊
        right=1,     # 右邊貼齊
        top=1,       # 上邊貼齊
        bottom=0,    # 下邊貼齊
        wspace=spacing / fixed_width,  # 圖之間水平間距
        hspace=0     # 無垂直間距
    )

    # （若你比較喜歡 tight_layout 的方式，也可以把上面兩行換成下面這行）
    # plt.tight_layout(pad=0, w_pad=spacing, h_pad=0)

    # 如果要加大標題或其它設定，可在這裡放：
    # fig.suptitle(title, fontsize=24, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        if not show:
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.show()

    return fig


def plot_combine_images_vertical(figs, save_path=None, spacing=0.05, fixed_width=5, fixed_height=1, show=False):
    num_images = len(figs)
    fig_width = fixed_width
    fig_height = num_images * fixed_height + (num_images - 1) * spacing
    print(f"fig_width: {fig_width}, {fig_height}")

    # 創建畫布，啟用 constrained_layout
    fig, axes = plt.subplots(
        num_images, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': [1] * num_images, 'hspace': spacing / fixed_height},
        constrained_layout=True
    )

    # 確保 axes 是列表
    if num_images == 1:
        axes = [axes]

    for i, ((key, fig_source), ax) in enumerate(zip(figs.items(), axes)):
        # 將 `fig_source` 等比例縮小 80%
        fig_source.set_size_inches(fig_source.get_size_inches() * 0.8)
        ax.imshow(fig_source.canvas.buffer_rgba())
        ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150)
        if not show:
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.show()

    return fig


def plot_heatmap(CI_values, save_path, width=15, height=15):
    """
    Generate and save a heatmap from the given values.

    Parameters:
    CI_values (list of list of float): The matrix values for the heatmap.
    save_path (str): The path to save the heatmap image.
    width (int): The width of the heatmap.
    height (int): The height of the heatmap.
    """
    # Convert the input list to a NumPy array and reshape to the specified width and height
    reshaped_CI_values = np.array(CI_values).reshape(height, width)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 9))  # Enlarge by 1.5 times

    # Display the heatmap
    cax = ax.matshow(reshaped_CI_values, cmap='viridis')

    # Add color bar
    fig.colorbar(cax)

    # Set up ticks and labels
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels([''] * width)
    ax.set_yticklabels([''] * height)

    # Render the values in the matrix
    for i in range(height):
        for j in range(width):
            ax.text(j, i, f"{reshaped_CI_values[i][j]:.2f}", va='center', ha='center', color='white', fontsize=7)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return fig
