import torch
from RegisterViT import RegisterViT

model = RegisterViT(num_registers=1)
model.eval()

# 準備輸入圖像
x_img = torch.randn(1, 3, 224, 224)

# 按照 RegisterViT.forward 的流程準備 encoder 的輸入
B = x_img.shape[0]
backbone = model.backbone

# 1. Patch embedding (只包含 patches，不包含 CLS)
x = backbone._process_input(x_img)  # [B, 196, C]

# 2. 加上 CLS token
cls_tok = backbone.class_token.expand(B, -1, -1)  # [B, 1, C]
x = torch.cat([cls_tok, x], dim=1)  # [B, 197, C]

# 3. 加上 Register tokens
reg = model.register_tokens.expand(B, -1, -1)  # [B, R, C]
x = torch.cat([x, reg], dim=1)  # [B, 197+R, C]

# 4. 加上 Positional embedding
x = x + backbone.encoder.pos_embedding

print(f"Input to encoder shape: {x.shape}")

with torch.no_grad():
    # 測試 1: 調用 encoder(x) - 這可能包含或不包含 ln
    out_encoder = backbone.encoder(x)
    print(f"encoder(x) output shape: {out_encoder.shape}")
    
    # 測試 2: 手動經過所有 layers（不包含 ln）
    # 這模擬 encoder 內部的 forward，但不包含最後的 encoder.ln
    x_manual = x.clone()
    for layer in backbone.encoder.layers:
        # Transformer block: Attention
        res = x_manual
        x_manual = layer.ln_1(x_manual)
        # MultiheadAttention 需要 query, key, value 三個參數
        x_manual = layer.self_attention(x_manual, x_manual, x_manual)[0]
        x_manual = x_manual + res
        
        # Transformer block: MLP
        res = x_manual
        x_manual = layer.ln_2(x_manual)
        x_manual = layer.mlp(x_manual)
        x_manual = x_manual + res
    
    out_manual_layers = x_manual
    print(f"Manual layers (no ln) output shape: {out_manual_layers.shape}")
    
    # 測試 3: 手動經過所有 layers + ln
    out_manual_with_ln = backbone.encoder.ln(out_manual_layers)
    print(f"Manual layers + ln output shape: {out_manual_with_ln.shape}")

# 比較 1: encoder(x) vs 手動 layers（不含 ln）
diff1 = torch.abs(out_encoder - out_manual_layers)
max_diff1 = diff1.max().item()
mean_diff1 = diff1.mean().item()

# 比較 2: encoder(x) vs 手動 layers + ln
diff2 = torch.abs(out_encoder - out_manual_with_ln)
max_diff2 = diff2.max().item()
mean_diff2 = diff2.mean().item()

print(f"\n比較結果:")
print(f"  encoder(x) vs 手動 layers (不含 ln):")
print(f"    Max difference: {max_diff1:.10f}")
print(f"    Mean difference: {mean_diff1:.10f}")
print(f"  encoder(x) vs 手動 layers + ln:")
print(f"    Max difference: {max_diff2:.10f}")
print(f"    Mean difference: {mean_diff2:.10f}")

if max_diff2 < 1e-5:
    print("\n  ✅ encoder(x) 不包含 ln")
    print("     (因為 encoder(x) == 手動 layers + ln)")
elif max_diff1 < 1e-5:
    print("\n  ✅ encoder(x) 包含 ln")
    print("     (因為 encoder(x) == 手動 layers，已經有 ln)")
else:
    print("\n  ⚠️  無法確定（兩者都不同，可能有其他差異）")

# 測試 4: 驗證 RegisterViT.forward 的完整流程
print("\n驗證 RegisterViT.forward 的完整流程:")
with torch.no_grad():
    # 完整流程
    x_full = backbone._process_input(x_img)
    cls_tok_full = backbone.class_token.expand(B, -1, -1)
    x_full = torch.cat([cls_tok_full, x_full], dim=1)
    reg_full = model.register_tokens.expand(B, -1, -1)
    x_full = torch.cat([x_full, reg_full], dim=1)
    x_full = x_full + backbone.encoder.pos_embedding
    
    # encoder
    x_after_encoder = backbone.encoder(x_full)
    # ln
    x_after_ln = backbone.encoder.ln(x_after_encoder)
    cls = x_after_ln[:, 0]
    logits = backbone.heads(cls)
    
    print(f"  After encoder shape: {x_after_encoder.shape}")
    print(f"  After ln shape: {x_after_ln.shape}")
    print(f"  CLS token shape: {cls.shape}")
    print(f"  Final logits shape: {logits.shape}")