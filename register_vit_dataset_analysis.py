import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

from config import config
from dataloader import get_dataloader
from models.RegisterViT import RegisterViT
from models.VIT_with_Partial_LRP_RegisterAware import VIT_with_Partial_LRP_RegisterAware
import models

"""
分析 RegisterViT 在 Colored MNIST 資料集上的表現
主要觀察 RegisterViT 的每個 Head 的 Relevance 分數
"""

# ----------------------------------------
# 沿用你之前的 build_model 工具函式
# ----------------------------------------
def build_analyzer(device):
    # (這裡請貼上你之前載入模型與權重的程式碼，簡化版)
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_args = dict(model_cfg["args"])
    model = getattr(getattr(models, model_name), model_name)(**model_args)
    
    ckpt_path = "runs/train/exp124/RegisterViT_best.pth" # 請確認你的路徑
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_weights"], strict=False)
    
    analyzer = VIT_with_Partial_LRP_RegisterAware(
        vit_model=model,
        num_patches=196,
        num_registers=model.num_registers
    ).to(device).eval()
    return analyzer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = build_analyzer(device)
    
    # 載入資料集 (建議先跑 Test set)
    _, test_loader = get_dataloader(
        dataset=config["dataset"],
        root=os.path.join(config["root"], "data"),
        batch_size=16, # 可以設大一點加快速度
        input_size=config["input_shape"],
    )
    
    results = []
    
    print("開始分析所有樣本...")
    
    # 使用 enumerate 以便紀錄 sample index (全域)
    global_idx = 0
    
    for xb, yb in tqdm(test_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        B = xb.shape[0]
        
        # 1. 取得基本預測資訊
        with torch.no_grad():
            logits = analyzer.model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            confs = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
            
        if yb.dim() > 1:
            gts = yb.argmax(dim=1).cpu().numpy()
        else:
            gts = yb.cpu().numpy()
            
        # 2. 逐一計算 LRP 分數 (因為 LRP 需要個別 backward，Batch 處理比較麻煩，這裡我們逐張做)
        # 雖然可以在 get_all_heads_importance 支援 Batch，但為了安全起見，這裡用 loop
        for i in range(B):
            img = xb[i:i+1] # [1, C, H, W]
            
            # 使用高效能方法取得 [12, 12] 矩陣
            # 這裡我們解釋 "Predicted Class" (模型為什麼覺得是這個)
            head_scores = analyzer.get_all_heads_importance(img, target_class=int(preds[i]))
            
            # 展平成 144 個數值
            flat_scores = head_scores.flatten()
            
            # 建立資料列
            row = {
                "sample_idx": global_idx,
                "ground_truth": gts[i],
                "predicted": preds[i],
                "is_correct": int(gts[i] == preds[i]),
                "confidence": confs[i],
            }
            
            # 加入 144 個欄位 (L0_H0, L0_H1 ... L11_H11)
            num_layers, num_heads = head_scores.shape
            for l in range(num_layers):
                for h in range(num_heads):
                    col_name = f"L{l}_H{h}"
                    row[col_name] = head_scores[l, h]
            
            results.append(row)
            global_idx += 1
            
    # 存檔
    df = pd.DataFrame(results)
    save_path = "vit_head_analysis_results.csv"
    df.to_csv(save_path, index=False)
    print(f"分析完成！數據已儲存至 {save_path}")
    print(f"總樣本數: {len(df)}")

if __name__ == "__main__":
    main()