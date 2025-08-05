import os
os.environ["OMP_NUM_THREADS"] = "2"


import numpy as np
from skimage.color import lab2rgb, rgb2lab
import warnings
import plotly.graph_objects as go
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import colorsys


# 1. 生成合理的 CIELab 網格點
def generate_cielab_grid_points(L_steps=10, a_steps=20, b_steps=20):
    """
    生成 CIELab 空間中的合理網格點。
    :param L_steps: int, L 軸上的步驟數量
    :param a_steps: int, a* 軸上的步驟數量
    :param b_steps: int, b* 軸上的步驟數量
    :return: numpy array, 形狀為 (N, 3) 的合理 CIELab 顏色點
    """
    L_values = np.linspace(0, 100, L_steps)
    a_values = np.linspace(-128, 127, a_steps)
    b_values = np.linspace(-128, 127, b_steps)

    grid_points = []
    for L in L_values:
        for a in a_values:
            for b in b_values:
                lab_point = np.array([[[L, a, b]]])  # 調整形狀為 (1, 1, 3)

                # 使用 RGB 反向驗證以確保 CIELab 點的合理性
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # 將 Lab 轉換為 RGB
                    rgb = lab2rgb(lab_point)
                    rgb_scaled = rgb * 255

                    # 檢查是否有警告（例如負值剪裁）
                    if len(w) > 0:
                        continue

                    # 確保 RGB 在 [0, 255] 範圍內
                    if not np.all((rgb_scaled >= 0) & (rgb_scaled <= 255)):
                        continue

                    # 進行反向驗證，將 RGB 轉換回 CIELab
                    lab_converted = rgb2lab(rgb).reshape(1, 1, 3)

                    # 檢查原始 Lab 與轉回的 Lab 是否一致（避免漂移）
                    if not np.allclose(lab_point, lab_converted, atol=2.0):
                        continue

                    grid_points.append([L, a, b])

    return np.array(grid_points)

# 2. 使用 K-means 聚類生成代表點並確保包含黑色和白色
def generate_representative_colors(all_colors, n_colors=30):
    """
    使用 K-means 聚類選出 n_colors 個代表點，並確保包含黑色與白色。
    :param all_colors: numpy array, 所有顏色點
    :param n_colors: int, 代表點數量
    :return: numpy array, 形狀為 (n_colors, 3) 的代表點
    """
    kmeans = KMeans(n_clusters=n_colors - 2, random_state=42)  # 減去兩個代表點留給黑色和白色
    kmeans.fit(all_colors)
    representative_colors = kmeans.cluster_centers_

    # 手動添加黑色和白色的 Lab 值
    black = np.array([0, 0, 0])   # 黑色 (L=0, a=0, b=0)
    white = np.array([100, 0, 0]) # 白色 (L=100, a=0, b=0)

    # 合併代表點
    representative_colors = np.vstack([representative_colors, black, white])

    return representative_colors

# 3. 使用 plotly 繪製 CIELab 色彩空間網格圖和代表點
def plot_cielab_grid_with_representatives(lab_points, representative_colors):
    """
    使用 Plotly 繪製 CIELab 色彩空間的網格圖與代表點
    :param lab_points: numpy array, CIELab 顏色點
    :param representative_colors: numpy array, 代表點
    """
    # 提取 L, a, b 值
    L = lab_points[:, 0]
    a = lab_points[:, 1]
    b = lab_points[:, 2]

    # 將 Lab 轉換為 RGB 用於顏色顯示
    colors = lab2rgb(lab_points.reshape(-1, 1, 3)).reshape(-1, 3)
    colors_hex = ['rgb({:.0f},{:.0f},{:.0f})'.format(r*255, g*255, b*255) for r, g, b in colors]

    # 代表點的 L, a, b 值
    rep_L = representative_colors[:, 0]
    rep_a = representative_colors[:, 1]
    rep_b = representative_colors[:, 2]
    rep_colors = lab2rgb(representative_colors.reshape(-1, 1, 3)).reshape(-1, 3)
    rep_colors_hex = ['rgb({:.0f},{:.0f},{:.0f})'.format(r*255, g*255, b*255) for r, g, b in rep_colors]

    # 使用 Plotly 繪製交互式 3D 圖形
    fig = go.Figure()

    # 繪製網格點
    fig.add_trace(go.Scatter3d(
        x=a, y=b, z=L,
        mode='markers',
        marker=dict(
            size=5,
            color=colors_hex,  # 使用 RGB 顏色來著色每個點
            opacity=0.5
        ),
        name="Grid Points"
    ))

    # 繪製代表點
    fig.add_trace(go.Scatter3d(
        x=rep_a, y=rep_b, z=rep_L,
        mode='markers',
        marker=dict(
            size=10,
            color=rep_colors_hex,  # 使用 RGB 顏色來著色每個點
            opacity=1.0,
            symbol='diamond'
        ),
        name="Color Filters"
    ))

    # 設置圖形屬性
    fig.update_layout(
        scene=dict(
            xaxis_title="a* (Green-Red)",
            yaxis_title="b* (Blue-Yellow)",
            zaxis_title="L* (Lightness)"
        ),
        title="CIELab Color Space Interactive Grid Visualization with Color Filters"
    )

    fig.show()

# 3. 將代表顏色點轉換為 RGB 並以陣列方式列出（依據規則排序）
def convert_to_rgb_array(lab_points):
    # 將 CIELab 轉換為 RGB，並轉換為 HSV 進行色環排序
    rgb_points = lab2rgb(lab_points.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv_points = [colorsys.rgb_to_hsv(*rgb) for rgb in rgb_points]

    # 計算每個顏色的色相（Hue），並將色相分成 60 度的單位
    angle_range = 360 // 60
    hue_bins = [int((hsv[0] * angle_range +0.5) % angle_range) for hsv in hsv_points]

    # 創建一個列表，其中包含每個顏色的 (Hue bin, L value, index) 三元組
    sorting_keys = [(hue_bins[i], -lab_points[i, 0], i) for i in range(len(lab_points))]

    # 根據 (Hue bin, -L value) 進行排序，白色在最前，黑色在最後
    sorted_indices = sorted(range(len(lab_points)), key=lambda i: (sorting_keys[i][0], sorting_keys[i][1]))
    white_index = np.where((lab_points == [100, 0, 0]).all(axis=1))[0][0]
    black_index = np.where((lab_points == [0, 0, 0]).all(axis=1))[0][0]

    sorted_indices.remove(white_index)
    sorted_indices.remove(black_index)
    sorted_indices = [white_index] + sorted_indices + [black_index]

    lab_points_sorted = lab_points[sorted_indices]
    colors_rgb = lab2rgb(lab_points_sorted.reshape(-1, 1, 3)).reshape(-1, 3) * 255
    colors_rgb = np.clip(colors_rgb, 0, 255).astype(int)
    # 將排序反轉
    return colors_rgb.tolist()

# 4. 將 RGB 顏色陣列轉換為 CIELab 座標
def convert_rgb_to_lab(rgb_points):
    """
    將 RGB 顏色陣列轉換為 CIELab 座標
    :param rgb_points: list of lists, 每個顏色點的 RGB 值
    :return: numpy array, 形狀為 (N, 3) 的 CIELab 顏色點
    """
    rgb_array = np.array(rgb_points) / 255.0  # 將 RGB 值歸一化到 [0, 1]
    lab_array = rgb2lab(rgb_array.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab_array


# 5. 可視化代表顏色點
def plot_representative_colors(colors_rgb, width, height):
    """
    將代表顏色點以顏色方格的方式可視化
    :param colors_rgb: list of lists, 每個顏色點的 RGB 值
    """


    # 創建一個空的顏色矩陣
    color_matrix = np.zeros((height, width,  3), dtype=int)

    # for i, color in enumerate(colors_rgb):
    #     row =  height - 1 - i // width
    #     col =  width - 1 - i %  width
    #     color_matrix[row, col] = color

    for i, color in enumerate(colors_rgb):
        row = i // width
        col = i % width
        color_matrix[row, col] = color

    # 可視化顏色矩陣
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(color_matrix / 255.0)
    ax.axis('off')
    plt.show()



# 主程式執行
if __name__ == "__main__":
    # 生成合理的 CIELab 空間網格點
    all_colors = generate_cielab_grid_points(L_steps=10, a_steps=20, b_steps=20)

    # 使用 K-means 聚類生成 30 個代表點
    representative_colors = generate_representative_colors(all_colors, n_colors=300)

    # # # RGB 顏色陣列
    # rgb_points = [
    #     [185, 31, 87], [208, 47, 72], [221, 68, 59], [233, 91, 35],
    #     [230, 120, 0], [244, 157, 0], [241, 181, 0], [238, 201, 0],
    #     [210, 193, 0], [168, 187, 0], [88, 169, 29], [0, 161, 90],
    #     [0, 146, 110], [0, 133, 127], [0, 116, 136], [0, 112, 155],
    #     [0, 96, 156], [0, 91, 165], [26, 84, 165], [83, 74, 160],
    #     [112, 63, 150], [129, 55, 138], [143, 46, 124], [173, 46, 108],
    #     [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [0, 0, 0], [255, 255, 255]
    # ]
    # # # 將 RGB 顏色陣列轉換為 CIELab 座標
    # representative_colors = convert_rgb_to_lab(rgb_points)

    # 繪製交互式網格圖與代表點
    plot_cielab_grid_with_representatives(all_colors, representative_colors)

    # 將代表點轉換為 RGB 陣列並輸出
    representative_colors_rgb = convert_to_rgb_array(representative_colors)
    print("代表顏色點的 RGB 陣列：")
    print(representative_colors_rgb)

    # 可視化代表顏色點
    width = 20
    height = 15
    plot_representative_colors(representative_colors_rgb, width, height )
