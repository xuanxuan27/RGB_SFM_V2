import torch
from file_tools import increment_path
from pathlib import Path
import os

project = "paper experiment"
name = 'RGB_SFMCNN_V2_CIFAR10-both-1-5-1-1-new_100_color'
group = "07/17"
tags = ['RGB_SFMCNN_V2']
description = """

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model_name = 'both-1-5-1-1-new_100_color'  # "RGB_SFMCNN_V2_best" 'only-rgb-25-channels'

arch = {
    "name": 'RGB_SFMCNN_V2',
    "need_calculate_status": False,
    "args": {
        "in_channels": 3,
        "out_channels": 10,
        "mode" : "both", #  'rgb', 'gray', or 'both'
        "Conv2d_kernel": [[ (1, 1), (5, 5), (1, 1), (1, 1)],
                          [ (5, 5),  (1, 1), (1, 1)]],
        # SFM_methods: "alpha_mean" "max" "none"
        "SFM_methods": [["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"],
                        ["alpha_mean", "alpha_mean", "alpha_mean", "alpha_mean"]],
        "SFM_filters": [[  (1, 1), (2, 2),  (1, 3),  (1, 1)],
                        [ (2, 2),  (1, 3),  (1, 1)]],
        # 對應到畫圖時，該 channel 的形狀
        "channels": [[(10, 10), (15, 15), (25, 25),  (35, 35)],
                     [(7, 10), (15, 15), (35, 35)]],
        "strides": [[1, 4, 1, 1],
                    [4, 1, 1]],
        "paddings": [[0, 0, 0, 0],
                     [0, 0, 0]],
        # color_filter : "new_10" "new_30" "new_100"  "old_30"
        "color_filter" : "new_100",

        # conv_method:  "cdist", "dot_product" "squared_cdist" "cosine" "none"
        "conv_method" : [["none", "cosine", "cosine", "cosine"],
                         ["cosine", "cosine", "cosine", "cosine"]],
        # initial: "kaiming" "uniform"
        "initial": [["none", "kaiming", "kaiming", "kaiming"],
                    ["kaiming", "kaiming", "kaiming", "kaiming"]],
        # rbfs: "triangle" "gauss" 'sigmoid' 'cReLU' 'cReLU_percent' 'regularization'
        "rbfs": [[["triangle", 'cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']],
                                 [['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent'], ['cReLU_percent']]],
        "activate_params": [[[1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]],
                            [[-1, 0.3], [-1, 0.4], [-1, 0.5], [-1, 0.5]]],
        "fc_input": (1225 + 1225) * 1 * 3,  # (rbg last channels + gray last channels) * last layer shape
        "device": device

    }
}

# arch = {
#     "name": 'ResNet',
#     "need_calculate_status" : False,
#     "args":{
#         'layers':18,
#         'in_channels':3,
#         "out_channels": 8
#     }
# }

# arch = {
#     "name": 'AlexNet',
#     "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 8,
#         "input_size": (28, 28)
#     }
# }
#

# arch = {
#     "name": 'DenseNet',
# "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 5
#     }
# }

# arch = {
#     "name": 'GoogLeNet',
# "need_calculate_status" : False,
#     "args":{
#         'in_channels':3,
#         "out_channels": 5
#     }
# }

create_dir = False
if create_dir:
    save_dir = increment_path('runs/train/exp', exist_ok=False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
else:
    save_dir = 'runs/train/exp'

print(save_dir)

lr_scheduler = {
    "name": "ReduceLROnPlateau",
    "args": {
        "patience": 100
    }
}

optimizer = {
    "name": "Adam",
    "args": {

    }
}

config = {
    "device": device,
    "root": os.path.dirname(__file__),
    "save_dir": save_dir,
    "load_model_name": load_model_name,
    "model": arch,
    "plot_bar": True,
    "plot_CAM" : False,
    # "dataset":
    # 'Colored_MNIST', 'Colored_FashionMNIST', 'MultiColor_Shapes_Database'
    # "PathMNIST", "BloodMNIST", "CIFAR10"
    # "RetinaMNIST_224"
    "dataset": 'CIFAR10',
    "input_shape": (28, 28),
    "batch_size": 256,
    "epoch": 200,
    "early_stop": False,
    "patience": 50,  # How many epochs without progress, early stop
    "lr": 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    'use_metric_based_loss': False,
    "loss_fn": 'CrossEntropyLoss',  # 'CustomLoss', # 'CustomLoss' "CrossEntropyLoss" 'MetricBaseLoss'
    "training_loss_fn": 'MetricBaseLoss',
    "use_preprocessed_image": False,
    # Heart calcification detection
    "heart_calcification": {
        "grid_size": 45,  # Image cutting size
        "need_resize_height": True,  # Whether to resize based on image height
        "resize_height": 900,  # Resize size
        "threshold": 0.5,
        # For calcification point bounding box, determine if it's a calcification point, shrink the bounding box
        "enhance_method": 'none',
        # Data contrast enhancement method 'contrast' 'normalize' 'histogram_equalization' 'scale_and_offset' 'clahe' 'none'
        "contrast_factor": 1.5,  # Contrast factor, default is 1.0 (no change)
        "use_vessel_mask": False,  # Use vessel mask
        "use_min_count" : False,
        "augment_positive":  False,
        "augment_multiplier" : 10
    },
}