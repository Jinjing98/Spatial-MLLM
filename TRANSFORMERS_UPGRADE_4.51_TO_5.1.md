# Transformers 升级修改总结 (4.51.3 → 5.1.0)

## 1. API 结构变化

### Config 层级变化
```python
# Old (4.51.3)
config.hidden_size
config.pad_token_id
config.rope_theta

# New (5.1.0)
config.text_config.hidden_size
config.text_config.pad_token_id
config.text_config.rope_parameters['rope_theta']  # 字典访问
```

### Model 层级变化
```python
# Old (4.51.3)
Qwen2_5_VLModel
  ├── visual
  ├── embed_tokens
  ├── layers
  └── norm

# New (5.1.0)
Qwen2_5_VLModel
  ├── visual
  └── language_model (Qwen2_5_VLTextModel)
        ├── embed_tokens
        ├── layers
        └── norm
```

### State Dict 键变化
```python
# Old keys
"embed_tokens.weight"
"layers.0.self_attn.q_proj.weight"

# New keys  
"language_model.embed_tokens.weight"
"language_model.layers.0.self_attn.q_proj.weight"
```

## 2. 修改的文件

### `src/qwenvl/external/vggt/layers/vision_transformer.py`
**问题**: Meta tensor 不支持 `.item()` 调用

**修复**:
```python
# Old
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

# New
linspace_tensor = torch.linspace(0, drop_path_rate, depth)
if linspace_tensor.is_meta:
    linspace_tensor = torch.linspace(0, drop_path_rate, depth, device='cpu')
dpr = linspace_tensor.tolist()
```

### `src/qwenvl/model/connector/__init__.py`
**问题**: Config 没有 `hidden_size` 属性

**修复**:
```python
# Old
language_dim=config.hidden_size

# New
language_dim=config.text_config.hidden_size
```

### `src/qwenvl/model/spatial_mllm.py`
**问题**: Model 没有直接的 `embed_tokens` 属性

**修复**:
```python
# Old
inputs_embeds = self.model.embed_tokens(input_ids)

# New
inputs_embeds = self.model.language_model.embed_tokens(input_ids)
```

### `src/qwenvl/model/custom_spatial_mllm.py`
**问题**: 
1. 同上 `embed_tokens` 访问
2. State dict 键不匹配

**修复**:
```python
# 1. Embed tokens 访问
inputs_embeds = self.model.language_model.embed_tokens(input_ids)

# 2. State dict 键映射
original_state = original_vl_model.state_dict()
custom_state = {}
for key, value in original_state.items():
    if key.startswith('language_model.'):
        new_key = key.replace('language_model.', '')
        custom_state[new_key] = value
    else:
        custom_state[key] = value
custom_vl_model.load_state_dict(custom_state, strict=True)
```

### `src/qwenvl/model/custom_qwen2_5_VLM.py`
**问题**:
1. `SlidingWindowCache` 不存在
2. Config 属性访问

**修复**:
```python
# 1. SlidingWindowCache 安全导入
try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    SlidingWindowCache = None

# 使用时检查
using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache) if SlidingWindowCache is not None else False

# 2. Config 属性访问
text_config = config.text_config if hasattr(config, 'text_config') else config
self.padding_idx = text_config.pad_token_id
self.vocab_size = text_config.vocab_size
```

### `src/qwenvl/model/custom_qwen2_5_VLRoPE.py`
**问题**:
1. `SlidingWindowCache` 不存在
2. `ROPE_INIT_FUNCTIONS['default']` 不存在
3. `rope_theta` 访问方式变化

**修复**:
```python
# 1. SlidingWindowCache 安全导入（同上）

# 2. 处理 'default' rope type
if self.rope_type == "default":
    self.rope_init_fn = CustomQwen2_5_VLRotaryEmbedding.compute_default_rope_parameters
else:
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

# 3. 添加 compute_default_rope_parameters 静态方法
@staticmethod
def compute_default_rope_parameters(config, device=None, seq_len=None):
    text_config = config.text_config if hasattr(config, 'text_config') else config
    if hasattr(text_config, 'rope_parameters') and isinstance(text_config.rope_parameters, dict):
        base = text_config.rope_parameters.get("rope_theta", 1000000.0)
    else:
        base = getattr(text_config, 'rope_theta', 1000000.0)
    # ... compute inv_freq
```

### `src/qwenvl/train/trainer.py`
**问题**: `print_trainable_parameters` 方法直接访问 `self.embed_tokens` 和 `self.layers`

**修复**:
```python
def print_trainable_parameters(self) -> None:
    # 兼容新旧 API
    if hasattr(self, 'language_model'):
        embed_tokens = self.language_model.embed_tokens
        layers = self.language_model.layers
    else:
        embed_tokens = self.embed_tokens
        layers = self.layers
    # ... rest of the code
```

## 3. 移除/废弃的 API

| API | 状态 | 替代方案 |
|-----|------|---------|
| `SlidingWindowCache` | 移除 | 使用 try-except 安全导入 |
| `ROPE_INIT_FUNCTIONS['default']` | 移除 | 使用 `compute_default_rope_parameters` 方法 |

## 4. 关键修复模式

### Pattern 1: Config 属性访问兼容
```python
text_config = config.text_config if hasattr(config, 'text_config') else config
# 然后使用 text_config 访问属性
```

### Pattern 2: Model 属性访问兼容
```python
if hasattr(model, 'language_model'):
    embed_tokens = model.language_model.embed_tokens
else:
    embed_tokens = model.embed_tokens
```

### Pattern 3: State dict 键映射
```python
new_key = key.replace('language_model.', '') if key.startswith('language_model.') else key
```

### Pattern 4: 安全导入已移除的类
```python
try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    SlidingWindowCache = None

# 使用时检查
if SlidingWindowCache is not None and isinstance(obj, SlidingWindowCache):
    # handle it
```

## 5. 注意事项

1. **所有修改标记**: 所有修改均添加 `# JJ:` 注释标记，便于追踪
2. **向后兼容**: 修改尽可能保持向后兼容，支持新旧两种 API
3. **Meta device**: 新版本在使用 `device_map` 时会启用 meta device 初始化，某些操作需要特殊处理
4. **RoPE 参数**: 从直接属性改为 `rope_parameters` 字典，需要注意访问方式

## 6. 测试建议

- ✅ 推理脚本已测试通过
- ⚠️ 训练脚本需要单独测试
- ⚠️ 其他 custom model 变体需要检查是否需要类似修复
