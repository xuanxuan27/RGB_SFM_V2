import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

"""
分析 RegisterViT 在不同資料集上的表現
主要觀察 RegisterViT 的每個 Head 的 Relevance 分數
"""

def analyze_class_patterns(csv_path="2025_analyze/caltech101_register_vit_head_analysis_results.csv"):
    df = pd.read_csv(csv_path)
    
    # 找出所有 Head 欄位
    head_cols = [c for c in df.columns if "L" in c and "H" in c]
    
    # 1. 取得所有正確預測的樣本
    df_correct = df[df["is_correct"] == 1]
    
    # 取得類別名稱映射 (假設你有)
    # class_names = {0: "airplane", 1: "car", ...} 
    
    # 針對每個類別，計算所有 Head 的平均 Relevance
    class_means = df_correct.groupby("ground_truth")[head_cols].mean()
    
    # 建立輸出資料夾
    os.makedirs("analysis_plots", exist_ok=True)
    
    # === 分析 A: 全局平均 (Global Average) ===
    # 模型整體最依賴哪些 Head?
    global_mean = df_correct[head_cols].mean().values.reshape(12, 12)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(global_mean, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Global Average Head Importance (All Correct Samples)")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.savefig("2025_analyze/analysis_plots/caltech101/global_head_importance.png")
    plt.close()
    
    # === 分析 B: 類別差異比較 (Class-specific Patterns) ===
    # 找出前 5 個樣本數最多的類別來畫圖
    top_classes = df_correct["ground_truth"].value_counts().head(5).index
    
    for cls_idx in top_classes:
        # 取出該類別的平均向量 (144維)
        cls_vector = class_means.loc[cls_idx].values.reshape(12, 12)
        
        # 為了凸顯差異，我們可以畫 "該類別平均 - 全局平均"
        # 正值代表該類別比平常更依賴這個 Head
        diff_map = cls_vector - global_mean
        
        plt.figure(figsize=(10, 8))
        # 使用 RdBu colormap，紅色正(更依賴)，藍色負(不依賴)
        sns.heatmap(diff_map, center=0, cmap="RdBu_r", annot=False)
        plt.title(f"Class {cls_idx} Specific Head Usage\n(Difference from Global Mean)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(f"2025_analyze/analysis_plots/caltech101/class_{cls_idx}_head_pattern.png")
        plt.close()
        print(f"Saved plot for Class {cls_idx}")

    # === 分析 C: 層級貢獻度 (Layer Importance) ===
    # 每一層的總重要性
    layer_cols = []
    for l in range(12):
        cols = [f"L{l}_H{h}" for h in range(12)]
        layer_cols.append(cols)
        
    layer_importance = []
    for l in range(12):
        # 這一層所有 Head 的總和
        val = df_correct[layer_cols[l]].sum(axis=1).mean()
        layer_importance.append(val)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(12), layer_importance, marker='o')
    plt.title("Average Layer Contribution to Relevance Flow")
    plt.xlabel("Layer")
    plt.ylabel("Total Relevance Score")
    plt.grid(True)
    plt.savefig("2025_analyze/analysis_plots/caltech101/layer_importance_curve.png")
    
if __name__ == "__main__":
    analyze_class_patterns()