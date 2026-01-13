# VIT_with_Partial_LRP vs VIT_with_Partial_LRP_RegisterAware 方法差異

## VIT_with_Partial_LRP 有但 VIT_with_Partial_LRP_RegisterAware 沒有的方法：

### 公共 API 方法：
1. `clear_head_mask()` - 清除所有注意力層的 head mask
2. `set_head_mask()` - 設定每層要保留的 head 清單
3. `_clear_memory_cache()` - 清理所有累積的張量，避免記憶體洩漏
4. `compute_lrp_relevance()` - convenience wrapper
5. `explain_per_layer()` - 計算每一層的 relevance map
6. `get_per_layer_activations()` - 獲取每層的 activation value
7. `get_cls_attention_map()` - 獲取最後一層的 CLS→patch attention map
8. `analyze_head_importance()` - 分析並視覺化每層 head 的重要性

### 靜態方法：
9. `relu_like_safe_div()` - 工具函數
10. `lrp_gelu_deeplift()` - GELU LRP 方法
11. `lrp_gelu_stable()` - Stable GELU LRP 方法

### 內部方法：
12. `_get_parent_by_name()` - 導航模組樹
13. `_wrap_attentions()` - 包裝 attention 模組
14. `_attn_forward_hook()` - attention forward hook
15. `_hook_tokens()` - Hook tokens
16. `_patch_size()` - 獲取 patch size
17. `_fallback_grad_relmap()` - 回退梯度方法
18. `_hook_mlp_and_ln()` - Hook MLP 和 LayerNorm
19. `_hook_layer_activations()` - Hook 每層的 activation
20. `_find_latest_cache()` - 找到最新的 cache
21. `_ln1_hook()` - LayerNorm 1 hook
22. `_ln2_hook()` - LayerNorm 2 hook
23. `_fc1_hook()` - FC1 hook
24. `_act_hook()` - Activation hook
25. `_fc2_hook()` - FC2 hook

## 在視覺化代碼中被使用的方法：
- `clear_head_mask()` - 第 470 行
- `_clear_memory_cache()` - 第 659 行

## 建議：
為 `VIT_with_Partial_LRP_RegisterAware` 添加以下方法以保持兼容性：
1. `clear_head_mask()` - 空實現（因為 RegisterAware 不使用 head mask）
2. `_clear_memory_cache()` - 清理 attn_cache 和 v_cache

