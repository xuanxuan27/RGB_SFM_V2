import shutil

import wandb
import numpy as np

from torch import optim, nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from dataloader import get_dataloader
from config import *
import models
from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, preprocess_retinal_tensor_batch
from file_tools import increment_path
from loss.loss_function import get_loss_function, MetricBaseLoss
from models.RGB_SFMCNN_V2 import get_feature_extraction_layers, get_basic_target_layers
from monitor.monitor_method import get_all_layers_stats
from memory_monitor import MemoryMonitor

# 1) 先設數學/精度偏好（提速、降峰值）
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2) 啟用 SDPA 的高效 kernel（全域生效）
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception as e:
    print(f"[warn] SDPA kernel toggle not available: {e}")

def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, eval_loss_fn, optimizer, scheduler, epoch, device,
          training_loss_fn, use_metric_based_loss=False):
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    best_train_acc = 0
    best_valid_loss =  float('inf')
    count = 0
    patience = config['patience']
    # 使用影像前處理
    use_preprocessed_image= config['use_preprocessed_image']
    checkpoint = {}

    # 需要計算 RM 分布指標
    need_calculate_status = arch["need_calculate_status"]
    if need_calculate_status:
        # 使否使用輪廓層
        mode = arch['args']['mode']
        use_gray = mode in ['gray', 'both']
        rgb_layers, gray_layers = get_basic_target_layers(model, use_gray=use_gray)


    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # 初始化記憶體監控
    memory_monitor = MemoryMonitor()
    memory_monitor.print_memory_status("(訓練開始)")
    
    # with torch.autograd.set_detect_anomaly(True):
    for e in range(epoch):
        print(f"------------------------------EPOCH {e}------------------------------")
        model.train()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
        losses = 0
        correct = 0
        size = 0
        # X, y = next(iter(train_dataloader))
        grad_accum_steps = config.get('grad_accum_steps', 1)
        for batch, (X, y) in progress:
            X = X.to(device); y= y.to(device)
            # 使用影像愈處理
            if use_preprocessed_image:
                X = preprocess_retinal_tensor_batch(X, final_size=config['input_shape'])

            with autocast('cuda', enabled=(device.type == 'cuda')):
                pred = model(X)
                # 轉換 one-hot 標籤為類別索引（CrossEntropy 需要 1D long）
                if y.dim() > 1:
                    y = y.argmax(1)
                y = y.long()
                # 檢查類別範圍，但不截斷（用於 VIT 30 類別訓練）
                C = pred.shape[1]
                if y.min() < 0 or y.max() >= C:
                    print(f"Warning: 標籤超出範圍 [{y.min()}, {y.max()}], 模型輸出類別數: {C}")
                    print("VIT 訓練不截斷標籤，請檢查模型配置是否正確")

                # if use_metric_based_loss:
                #     loss = training_loss_fn(pred, y, model, rgb_layers, gray_layers, X)
                # else:
                #     loss = eval_loss_fn(pred, y)

                # 縮減記憶體使用量
                if use_metric_based_loss:
                    if hasattr(model, "enable_explanations"): model.enable_explanations(True)
                    if hasattr(model, "enable_return_attn"):  model.enable_return_attn(True)
                    
                    with torch.no_grad():               # 只要這些分數不需要對模型反傳
                        mb_metric = training_loss_fn(pred, y, model, rgb_layers, gray_layers, X)
                    
                    # 關回去，避免後面 batch/步驟持續佔顯存
                    if hasattr(model, "enable_explanations"): model.enable_explanations(False)
                    if hasattr(model, "enable_return_attn"):  model.enable_return_attn(False)
                    
                    loss = eval_loss_fn(pred, y) + config.get('metric_coef', 0.0) * mb_metric
                else:
                    loss = eval_loss_fn(pred, y)

            # 反向传播（AMP）+ 梯度累積
            scaler.scale(loss / grad_accum_steps).backward()
            if (batch + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            losses += loss.detach().item()
            size += len(X)

            # 使用 VIT 會出現錯誤所以修改
            # correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # 定期清理記憶體，避免累積
            if batch % 30 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 如果模型有記憶體清理方法，也調用它
            if hasattr(model, '_clear_memory_cache'):
                model._clear_memory_cache()

            train_loss = losses/(batch+1)
            train_acc = correct/size
            progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))

        # 若最後一個累積步未觸發 optimizer.step()，此處補一次
        if grad_accum_steps > 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 每個 epoch 結束後清理記憶體
        if hasattr(model, '_clear_memory_cache'):
            model._clear_memory_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 檢查記憶體洩漏
        if memory_monitor.check_memory_leak(threshold_gb=0.5):
            print("檢測到記憶體洩漏，執行強制清理...")
            memory_monitor.force_cleanup()

        # — 在 train() 內，呼叫 eval() 之前 —
        if hasattr(model, '_clear_memory_cache'):
            model._clear_memory_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


        valid_acc, valid_loss, _ = eval(valid_dataloader, model, eval_loss_fn, False, device = device, use_preprocessed_image=use_preprocessed_image)
        print(f"Test Loss: {valid_loss}, Test Accuracy: {valid_acc}")
        
        # 每個 epoch 結束後清理記憶體
        if hasattr(model, '_clear_memory_cache'):
            model._clear_memory_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 檢查記憶體洩漏
        if memory_monitor.check_memory_leak(threshold_gb=0.5):
            print("檢測到記憶體洩漏，執行強制清理...")
            memory_monitor.force_cleanup()
        
        # 每 5 個 epoch 打印記憶體狀態
        if e % 5 == 0:
            memory_monitor.print_memory_status(f"(Epoch {e} 結束)")



        if scheduler:
            scheduler.step(valid_loss)

        metrics = {
            "train/loss": train_loss,
            "train/epoch": e,
            "train/accuracy": train_acc,
            "train/learnrate": optimizer.param_groups[0]['lr'],
            "valid/loss": valid_loss,
            "valid/accuracy": valid_acc,
        }
        wandb.log(metrics, step=e)

        #early stopping
        if config['early_stop']:
            if valid_acc < best_valid_acc:
                count += 1
                if count >= patience:
                    break

        # update model methods
        if valid_acc > best_valid_acc:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
        # if valid_loss < best_valid_loss:
        #     best_valid_acc = valid_acc
        if train_acc >= best_train_acc:
            best_train_acc = train_acc

            cur_train_loss = train_loss
            cur_train_acc = train_acc

            count = 0

            del checkpoint
            checkpoint = {}
            print(f'best epoch: {e}')
            checkpoint['model_weights'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            checkpoint['train_loss'] = train_loss
            checkpoint['train_acc'] = train_acc
            checkpoint['valid_loss'] = valid_loss
            checkpoint['valid_acc'] = valid_acc

            torch.save(checkpoint, f'{config["save_dir"]}/epochs{e}.pth')
        if e == 200:
            torch.save(checkpoint, f'{config["save_dir"]}/epochs{e}.pth')

                
    # print(model)
    # Monitor
    # Prepare monitor
    # images, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    # for batch in train_dataloader:
    #     imgs, lbls = batch
    #     images = torch.cat((images, imgs.to(device)))
    #     labels = torch.cat((labels, lbls.to(device)))

    # 需要計算 RM 分布指標
    # need_calculate_status = arch["need_calculate_status"]
    # if need_calculate_status:
    #     use_gray = arch['args']['use_gray']  # 使否使用輪廓層
    #     rgb_layers, gray_layers = get_basic_target_layers(model, use_gray=use_gray)
    #     layer_stats, overall_stats = get_all_layers_stats(model, rgb_layers, gray_layers, images)
    #
    #     for key, value in overall_stats.items():
    #         wandb.summary[key] = value
    #
    #     wandb.summary['layers'] = layer_stats
    #     print(layer_stats)

    return cur_train_loss, cur_train_acc, best_valid_loss, best_valid_acc, checkpoint

def eval(dataloader, model, loss_fn, need_table=False, device=None, use_preprocessed_image=False):
    model.eval()
    if hasattr(model, "enable_explanations"): model.enable_explanations(False)
    if hasattr(model, "enable_return_attn"):  model.enable_return_attn(False)

    losses = 0.0; correct = 0; size = 0; table = []
    micro = int(config.get("val_micro_batch", 0))
    progress = tqdm(enumerate(dataloader), desc="Loss: ", total=len(dataloader))

    with torch.inference_mode():   # 比 no_grad 更省顯存
        for batch, (X, y) in progress:
            X = X.to(device); y = y.to(device)
            if use_preprocessed_image:
                X = preprocess_retinal_tensor_batch(X, final_size=config['input_shape'])

            if micro and micro < X.size(0):
                outs = []
                for i in range(0, X.size(0), micro):
                    Xi = X[i:i+micro]
                    with autocast('cuda', enabled=(device.type == 'cuda')):
                        outs.append(model(Xi))
                pred = torch.cat(outs, dim=0)
            else:
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    pred = model(X)

            if y.dim() > 1: y = y.argmax(1)
            y = y.long()
            # 檢查標籤範圍，但不截斷（用於 VIT 30 類別訓練）
            if y.min() < 0 or y.max() >= pred.shape[1]:
                print(f"Warning: 評估時標籤超出範圍 [{y.min()}, {y.max()}], 模型輸出類別數: {pred.shape[1]}")
                print("VIT 訓練不截斷標籤，請檢查模型配置是否正確")
            loss = loss_fn(pred, y)

            losses += float(loss)
            size   += X.size(0)
            correct += (pred.argmax(1) == y).sum().item()

            test_loss = losses/(batch+1)
            test_acc  = correct/size
            progress.set_description(f"Loss: {test_loss:.7f}, Accuracy: {test_acc:.7f}")
            
            # 定期清理記憶體，避免累積
            if batch % 30 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 如果模型有記憶體清理方法，也調用它
            if hasattr(model, '_clear_memory_cache'):
                model._clear_memory_cache()

    return correct/size, losses/len(dataloader), table


config['save_dir'] = increment_path(config['save_dir'], exist_ok = False)
Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=project,

    name = name,

    notes = description,
    
    tags = tags,

    group = group,
    
    # track hyperparameters and run metadata
    config=config
)

train_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], root=config['root'] + '/data/', batch_size=config['batch_size'], input_size=config['input_shape'])

model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
model = model.to(config['device'])
print(model)
# summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']))


eval_loss_fn = get_loss_function(config['loss_fn'])
training_loss_fn = get_loss_function(config['training_loss_fn'])
use_metric_based_loss = config['use_metric_based_loss']

optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), lr=config['lr'], **dict(config['optimizer']['args']))
scheduler = getattr(optim.lr_scheduler, config['lr_scheduler']['name'])(optimizer, **dict(config['lr_scheduler']['args']))

shutil.copyfile(f'./models/{config["model"]["name"]}.py', f'{config["save_dir"]}/{config["model"]["name"]}.py')
shutil.copyfile(f'./config.py', f'{config["save_dir"]}/config.py')

# wandb.watch(model, loss_fn, log="all", log_freq=1)
train_loss, train_acc, valid_loss, valid_acc, checkpoint = train(train_dataloader, test_dataloader, model, eval_loss_fn,
                                                                 optimizer, scheduler, config['epoch'], device = config['device'],
                                                                 training_loss_fn=training_loss_fn, use_metric_based_loss=use_metric_based_loss)
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))
print("Valid: \n\tAccuracy: {}, Avg loss: {} \n".format(valid_acc, valid_loss))

# print(f"check point {checkpoint}")

test_acc, test_loss, test_table = eval(test_dataloader, model, eval_loss_fn, device = config['device'], need_table=False, use_preprocessed_image=config['use_preprocessed_image'])
print("Test 1: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Test model
if 'model_weights' not in checkpoint:
    print("Warning: No model_weights found in checkpoint!")
else:
    model.load_state_dict(checkpoint['model_weights'])
model.to(device)
test_acc, test_loss, test_table = eval(test_dataloader, model, eval_loss_fn, device = config['device'], need_table=False, use_preprocessed_image=config['use_preprocessed_image'])
print("Test 2: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['final_train_accuracy'] = train_acc
wandb.summary['final_train_avg_loss'] = train_loss
wandb.summary['final_test_accuracy'] = test_acc
wandb.summary['final_test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})
print(f'checkpoint keys: {checkpoint.keys()}')

# 儲存模型到 run 中
torch.save(checkpoint, f'{config["save_dir"]}/{config["model"]["name"]}_best.pth')

# 儲存模型到 pth 中
load_model_name= config["load_model_name"]
load_model_path = f'./pth/{config["dataset"]}'
if not os.path.exists(load_model_path):
    os.makedirs(load_model_path)  # 建立資料夾
torch.save(checkpoint, load_model_path + f'/{load_model_name}.pth')

art = wandb.Artifact(f'{config["model"]["name"]}_{config["dataset"]}', type="model")
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}_best.pth')
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}.py')
art.add_file(f'{config["save_dir"]}/config.py')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()