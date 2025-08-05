import shutil

import torchvision
from matplotlib import pyplot as plt
from pytorch_grad_cam import (GradCAM, HiResCAM, GradCAMPlusPlus,
                              AblationCAM, ScoreCAM, EigenCAM, LayerCAM)

from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, display_image_comparison, \
    check_then_preprocess_images
from load_tools import load_model_and_data
from models.RGB_SFMCNN_V2 import get_feature_extraction_layers
from plot_tool.plot_cam_method import  generate_cam_visualizations
from plot_tool.plot_graph_method import plot_combine_images, plot_combine_images_vertical, plot_map
from ci_getter import *
import time
import matplotlib


# 設定是否繪製 CAM
PLOT_CAM = False

# 設定是否使用預處理後的影像
use_preprocessed_image = config['use_preprocessed_image']

# 使用 non-interactive 的後端以支援儲存圖片
matplotlib.use('Agg')

# 載入模型與資料集
checkpoint_filename = config["load_model_name"]
test_data = True # 測試模型準確度
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename, test_data=test_data)

mode = arch['args']['mode'] # 模式

# 路徑
save_root = f'./detect/{config["dataset"]}/{checkpoint_filename}/example'
print(save_root)
if os.path.exists(save_root):
    shutil.rmtree(save_root)  # 刪除資料夾及其內容
    os.makedirs(save_root)  # 重新建立資料夾
else:
    os.makedirs(save_root)  # 重新建立資料夾

# 提取 RGB 與 Gray 分支的 feature extraction 層
rgb_layers, gray_layers = get_feature_extraction_layers(model)

print(rgb_layers.keys())

if PLOT_CAM:
    use_gray = mode in ['gray', 'both']
    rgb_layers_cam, gray_layers_cam = get_basic_target_layers(model, use_gray)

    if mode in ['rgb']:
        target_layers_cam = rgb_layers_cam
    if mode in ['gray']:
        target_layers_cam = rgb_layers_cam
    if mode in ['both']:
        target_layers_cam = rgb_layers_cam | gray_layers_cam

    print(target_layers_cam)

# 視情況進行視網膜影像預處理
preprocess_images = check_then_preprocess_images(images)

# 取得所有層的 Critical Inputs
force_regenerate=False
CIs, CI_values = get_CIs(model, preprocess_images)

# 設定需處理的資料筆數
example_num = 450

gray_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
    ])

def process_layer(
    image: torch.Tensor,
    layer_name: str,
    use_gray: bool,
    model,
    layers,
    plot_shape,
    CIs,
    RM_CIs,
    arch_args,
    RM_save_path: str,
    RM_CI_save_path: str,
    RM_figs: dict,
    RM_CI_figs: dict,
) -> None:
    """
    處理單層的 RM, FM, CI 可視化並儲存圖檔。
    """
    print(f"--- Processing {layer_name}...")
    t0 = time.time()

    in_channels = 1 if use_gray else arch_args['in_channels']
    input_image = model.gray_transform(image.unsqueeze(0)) if use_gray else image.unsqueeze(0)

    RM_Conv = layers[layer_name + '_after_Conv'](input_image)[0]
    RM_H, RM_W =RM_Conv.shape[1], RM_Conv.shape[2]

    fig_rm_conv = plot_map(
        RM_Conv.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
        path=f'{RM_save_path}/{layer_name}_RM_Conv'
    )
    RM_figs[layer_name+ '_after_Conv'] = fig_rm_conv


    RM = layers[layer_name](input_image)[0]
    RM_H, RM_W = RM.shape[1], RM.shape[2]

    t1 = time.time()
    print(f"Complete RM calculate - time: {t1 - t0:.3f} sec")

    fig_rm = plot_map(
        RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
        path=f'{RM_save_path}/{layer_name}_RM'
    )
    RM_figs[layer_name] = fig_rm


    t2 = time.time()
    print(f"Complete plot RM - time: {t2 - t1:.3f} sec")

    top_idx = torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()
    CI_H, CI_W = CIs[layer_name].shape[2], CIs[layer_name].shape[3]
    RM_CI = CIs[layer_name][top_idx].reshape(RM_H, RM_W, CI_H, CI_W, in_channels)
    RM_CIs[layer_name] = RM_CI

    t3 = time.time()
    print(f"Complete calculate RM_CI - time: {t3 - t2:.3f} sec")

    RM_CI_figs[layer_name] = plot_map(RM_CI, path=f'{RM_CI_save_path}/{layer_name}_RM_CI', cmap='gray' if use_gray else None)

    t4 = time.time()
    print(f"Complete plot RM_CI - time: {t4 - t3:.3f} sec")
    print(f"Finished {layer_name} - time: {t4 - t0:.3f} sec")



def process_image(image, label, test_id):
    """
    處理單一圖像的 RM, RM-CI 生成與儲存
    """
    print(f"處理編號: {test_id}")

    save_path = f'{save_root}/{label.argmax().item()}/example_{test_id}/'
    RM_save_path = f'{save_path}/RMs/'
    RM_CI_save_path = f'{save_path}/RM_CIs/'
    os.makedirs(RM_save_path, exist_ok=True)
    os.makedirs(RM_CI_save_path, exist_ok=True)

    use_gray =  mode in ['gray', 'both']  # 使否使用輪廓層

    RM_CIs = {}
    RM_figs = {}
    RM_CI_figs = {}

    fig_origin = plt.figure(figsize=(5, 5), facecolor="white")
    # # **關鍵：把 axes 填滿整個 figure**
    # fig_origin.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(image.permute(1, 2, 0).detach().numpy())
    plt.axis('off')

    plt.savefig(save_path + f'origin_{test_id}.png', bbox_inches='tight', pad_inches=0)
    RM_CI_figs['Origin'] = fig_origin

    # 若啟用預處理，進行視網膜圖像強化
    if use_preprocessed_image:
        scaled_img, blurred_img, high_contrast_img, final_img = preprocess_retinal_tensor_image(image)
        display_image_comparison(save_path=save_path + f'preprocess.png', origin_img=image, final_img=final_img)
        fig_pre = plt.figure(figsize=(5, 5), facecolor="white")
        plt.imshow(final_img.permute(1, 2, 0).detach().numpy())
        plt.axis('off')
        plt.savefig(save_path + f'preprocess_{test_id}.png', bbox_inches='tight', pad_inches=0)
        RM_CI_figs['Preprocess'] = fig_pre
        image = final_img

    # 灰階圖
    if use_gray:
        fig_gray = plt.figure(figsize=(5, 5), facecolor="white")
        gray_image = gray_transform(image.unsqueeze(0))[0]
        plt.imshow(gray_image.squeeze().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(save_path + f'gray_{test_id}.png', bbox_inches='tight', pad_inches=0)
        # RM_CI_figs['Gray'] = fig_gray

    # 切割原圖，並顯示其分區反應
    segments = split(image.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0][0],
                     stride=(arch['args']['strides'][0][0], arch['args']['strides'][0][0]))[0]
    origin_split_img = plot_map(segments.permute(1, 2, 3, 4, 0), path=save_path + f'origin_split_{test_id}.png')
    RM_figs['Origin_Split'] = origin_split_img

    channels = arch['args']['channels']

    if mode in ['rgb', 'both']:
        # 處理 RGB 分支所有層
        for i in range(len(model.RGB_convs)):
            layer_name = f'RGB_convs_{i}'
            print(layer_name)
            plot_shape = channels[0][i]
            process_layer(image, layer_name, use_gray=False, model=model, layers=rgb_layers, plot_shape=plot_shape,
                          CIs=CIs, RM_CIs=RM_CIs,
                          arch_args=arch['args'], RM_save_path=RM_save_path, RM_CI_save_path=RM_CI_save_path,
                          RM_figs=RM_figs, RM_CI_figs=RM_CI_figs)

    if mode in ['gray', 'both']:
        # 處理 Gray 分支所有層
        for i in range(len(model.Gray_convs)):
            layer_name = f'Gray_convs_{i}'
            plot_shape = channels[1][i]
            process_layer(image, layer_name, use_gray=True, model=model, layers=gray_layers, plot_shape=plot_shape,
                          CIs=CIs, RM_CIs=RM_CIs,
                          arch_args=arch['args'], RM_save_path=RM_save_path, RM_CI_save_path=RM_CI_save_path,
                          RM_figs=RM_figs, RM_CI_figs=RM_CI_figs)


    # 合併圖
    t1 = time.time()

    plot_combine_images(RM_figs, RM_save_path + f'combine')
    t2 = time.time()
    print(f"Finished plot combine RM fig - time: {t2 - t1:.3f} sec")

    RM_CI_combine_fig = plot_combine_images(RM_CI_figs, RM_CI_save_path + f'combine')
    t3 = time.time()
    print(f"Finished plot combine RM_CI fig - time: {t3 - t2:.3f} sec")

    # 如果啟用 CAM，則繪製所有 CAM 方法的對應圖像
    if PLOT_CAM:
        # cam_methods = [GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, AblationCAM,
        #                ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, KPCA_CAM]

        cam_methods = [GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM,
                       ScoreCAM, EigenCAM, LayerCAM]

        cam_figs = {}
        RM_CI_figs = {'raw': RM_CI_combine_fig}


        for method in cam_methods:
            print(f"\nDrawing {method.__name__}...")
            start_time = time.time()  # ⏱️ 開始計時

            cam_fig, RM_CI_fig = generate_cam_visualizations(
                model=model,
                label=label.argmax().item(),
                image=image,
                origin_img=fig_origin,
                RM_CIs=RM_CIs,
                save_path=RM_CI_save_path,
                method=method
            )

            elapsed = time.time() - start_time  # ⏱️ 結束時間
            print(f"{method.__name__} took {elapsed:.2f} seconds")

            cam_figs[method.__name__] = cam_fig
            RM_CI_figs[method.__name__] = RM_CI_fig

        plot_combine_images_vertical(cam_figs, RM_CI_save_path + f'cam/cams_combine')
        plot_combine_images_vertical(RM_CI_figs, RM_CI_save_path + f'/{method.__name__}_combine')

    plt.close('all')


# # 針對整個資料集
for test_id in range(example_num):
    process_image(images[test_id], labels[test_id], test_id)

