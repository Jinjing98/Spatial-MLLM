tokens_per_second: set to 1 in sft spatialmllm (25 in qwen2.5vl)
temporal_patch_size: set to 2 both for qwem2.5 and spatialmllm
this explain why: sft_spatiallmllm need get_rope_index_2 rather get_rope_index_25 (default). sft_spatiallmllm is optimized in a way, that candidate frames are having large view differs! (considering temporal_patch_size still 2, 'the merging' is wrong! )


////temporal_patch_size=2,  in 2D vision enc//////
////visual_temporal_merge_size=2, process VGGT 3d token so as to can be merged 
with 2D ones //////
qwen 2.5 issue: the alias of 'temporal' tag for one token (in fact the one token merged temporal_patch_size at two timepoints )


好，我们就用你给的 **total frames = 16**，图像尺寸 **644×476** 来把“temporal patching / spatial patching / merge tubelet”完整算一遍。为了可计算，我需要指定两个关键超参（这类模型里几乎固定）：

* `patch_size = 14`（Qwen2-VL / InternVL2 常见设置，644/14=46 正好整除）
* `temporal_patch_size = 2`（常见 tubelet 时间长度）
* `visual_temporal_merge_size = 2`
* `visual_spatial_merge_size = 2`

如果你实际参数不同，把数替换即可（我最后给公式）。

---

# Part A：像素级 temporal patching（你上一段 np 代码）

## A1. 先算空间 patch 网格 grid_h, grid_w

图像尺寸：`H=476, W=644`

`patch_size = 14`

* `grid_h = 476 // 14 = 34`
* `grid_w = 644 // 14 = 46`

每帧 patch 数：

* `P = grid_h * grid_w = 34 * 46 = 1564`

---

## A2. temporal patching：16 帧按 temporal_patch_size=2 分组

`T = 16`
`temporal_patch_size = 2`

* `grid_t = T / 2 = 8`

所以：时间上被切成 **8 个 tubelet（每个 tubelet 2 帧）**。

---

## A3. 像素 tubelet token 数量

上一段 flatten 输出 token 数是：

```
num_tokens = grid_t * grid_h * grid_w
           = 8 * 34 * 46
           = 12512
```

**结论：12512 个 tubelet tokens。**

---

## A4. 每个 token 的维度（像素 flatten）

上一段的 token dim 是：

```
token_dim = channel * temporal_patch_size * patch_size * patch_size
```

如果 RGB：

* `C=3`
* `temporal_patch_size=2`
* `patch_size=14`

则：

* `token_dim = 3 * 2 * 14 * 14 = 1176`

---

### ✅ 像素 tubelet flatten 最终 shape

```
flatten_patches.shape = (12512, 1176)
```

---

# Part B：embedding 级 tubelet merge（你这段 preprocess_spatial_embeds）

这段不是处理像素了，而是处理视觉 backbone 输出的 patch embedding。

假设 backbone 对每个 patch 输出 embedding dim `DD`（代码里叫 DD，注释写 2D），例如：

* DD = 1024 或 1536（取决于模型）
* 我这里先保留符号 DD，你可以代入真实值

---

## B1. spatial_embeds 的初始 shape

你的注释给了：

* `spatial_embeds: [B, S, P, DD]`
* B=1
* P=1564（上面算出来了）
* S 是时间长度（注意：这里的 S 不一定等于 total frames，通常是 “帧token数” 或 “merge前时间token数”）

为了和你例子对齐，我们用最自然的情况：

* 每帧有一组 patch embedding
* 所以 `S = total_frames = 16`

因此初始为：

```
[1, 16, 1564, DD]
```

---

## B2. temporal merge：visual_temporal_merge_size = 2

代码里要求：

```
npatch_t = S // visual_temporal_merge_size
```

所以：

* `npatch_t = 16 / 2 = 8`

这和上面像素 tubelet 的 `grid_t=8` 是一致的。

---

## B3. spatial merge：visual_spatial_merge_size = 2

空间网格是：

* npatch_h = 34
* npatch_w = 46

merge_size=2 时，空间 token 数会减少 4 倍：

* merged_h = 34 / 2 = 17
* merged_w = 46 / 2 = 23

---

## B4. tubelet token 数量（embedding merge 后）

最终 tubelet token 数是：

```
num_tokens = npatch_t * merged_h * merged_w
           = 8 * 17 * 23
           = 3128
```

注意：这比像素 flatten 的 12512 少 4 倍（因为 spatial merge=2×2）。

---

## B5. 每个 tubelet token 的维度

代码最后：

```
token_dim = DD * visual_temporal_merge_size * visual_spatial_merge_size^2
```

代入：

* temporal_merge=2
* spatial_merge=2 → spatial_merge^2=4

得到：

```
token_dim = DD * 2 * 4 = 8DD
```

如果 DD=1024，则 token_dim = 8192。

---

### ✅ embedding tubelet 最终 shape

```
all_spatial_embeds.shape = (3128, 8DD)
```

例如 DD=1024：

```
(3128, 8192)
```

---

# 关联总结（你需要记住的结论）

### 对于 16 帧 + 644×476 + patch_size=14：

## 1）不 merge（像素 flatten）

* grid_h=34, grid_w=46
* grid_t=8
* token 数：12512
* token dim：1176

## 2）embedding merge（temporal=2, spatial=2）

* token 数：3128
* token dim：8DD

---

# 如果你实际 patch_size 不是 14

我给你通用公式，你可以自己秒算：

* `grid_h = H // patch_size`
* `grid_w = W // patch_size`
* `P = grid_h * grid_w`
* `grid_t = T // temporal_patch_size`

像素 flatten token：

* `N = grid_t * grid_h * grid_w`
* `D = C * temporal_patch_size * patch_size^2`

embedding merge token：

* `N = (T / temporal_merge) * (grid_h / spatial_merge) * (grid_w / spatial_merge)`
* `D = DD * temporal_merge * spatial_merge^2`

---

如果你把你项目里的真实参数（`patch_size / temporal_patch_size / visual_spatial_merge_size / visual_temporal_merge_size / DD`）贴出来，我可以帮你把最终 token 数、显存规模（FP16/FP32）、以及 attention 复杂度也一起算出来。


好，我们就用你给的 **total frames = 16**，图像尺寸 **644×476** 来把“temporal patching / spatial patching / merge tubelet”完整算一遍。为了可计算，我需要指定两个关键超参（这类模型里几乎固定）：

* `patch_size = 14`（Qwen2-VL / InternVL2 常见设置，644/14=46 正好整除）
* `temporal_patch_size = 2`（常见 tubelet 时间长度）
* `visual_temporal_merge_size = 2`
* `visual_spatial_merge_size = 2`

如果你实际参数不同，把数替换即可（我最后给公式）。

---

# Part A：像素级 temporal patching（你上一段 np 代码）

## A1. 先算空间 patch 网格 grid_h, grid_w

图像尺寸：`H=476, W=644`

`patch_size = 14`

* `grid_h = 476 // 14 = 34`
* `grid_w = 644 // 14 = 46`

每帧 patch 数：

* `P = grid_h * grid_w = 34 * 46 = 1564`

---

## A2. temporal patching：16 帧按 temporal_patch_size=2 分组

`T = 16`
`temporal_patch_size = 2`

* `grid_t = T / 2 = 8`

所以：时间上被切成 **8 个 tubelet（每个 tubelet 2 帧）**。

---

## A3. 像素 tubelet token 数量

上一段 flatten 输出 token 数是：

```
num_tokens = grid_t * grid_h * grid_w
           = 8 * 34 * 46
           = 12512
```

**结论：12512 个 tubelet tokens。**

---

## A4. 每个 token 的维度（像素 flatten）

上一段的 token dim 是：

```
token_dim = channel * temporal_patch_size * patch_size * patch_size
```

如果 RGB：

* `C=3`
* `temporal_patch_size=2`
* `patch_size=14`

则：

* `token_dim = 3 * 2 * 14 * 14 = 1176`

---

### ✅ 像素 tubelet flatten 最终 shape

```
flatten_patches.shape = (12512, 1176)
```

---

# Part B：embedding 级 tubelet merge（你这段 preprocess_spatial_embeds）

这段不是处理像素了，而是处理视觉 backbone 输出的 patch embedding。

假设 backbone 对每个 patch 输出 embedding dim `DD`（代码里叫 DD，注释写 2D），例如：

* DD = 1024 或 1536（取决于模型）
* 我这里先保留符号 DD，你可以代入真实值

---

## B1. spatial_embeds 的初始 shape

你的注释给了：

* `spatial_embeds: [B, S, P, DD]`
* B=1
* P=1564（上面算出来了）
* S 是时间长度（注意：这里的 S 不一定等于 total frames，通常是 “帧token数” 或 “merge前时间token数”）

为了和你例子对齐，我们用最自然的情况：

* 每帧有一组 patch embedding
* 所以 `S = total_frames = 16`

因此初始为：

```
[1, 16, 1564, DD]
```

---

## B2. temporal merge：visual_temporal_merge_size = 2

代码里要求：

```
npatch_t = S // visual_temporal_merge_size
```

所以：

* `npatch_t = 16 / 2 = 8`

这和上面像素 tubelet 的 `grid_t=8` 是一致的。

---

## B3. spatial merge：visual_spatial_merge_size = 2

空间网格是：

* npatch_h = 34
* npatch_w = 46

merge_size=2 时，空间 token 数会减少 4 倍：

* merged_h = 34 / 2 = 17
* merged_w = 46 / 2 = 23

---

## B4. tubelet token 数量（embedding merge 后）

最终 tubelet token 数是：

```
num_tokens = npatch_t * merged_h * merged_w
           = 8 * 17 * 23
           = 3128
```

注意：这比像素 flatten 的 12512 少 4 倍（因为 spatial merge=2×2）。

---

## B5. 每个 tubelet token 的维度

代码最后：

```
token_dim = DD * visual_temporal_merge_size * visual_spatial_merge_size^2
```

代入：

* temporal_merge=2
* spatial_merge=2 → spatial_merge^2=4

得到：

```
token_dim = DD * 2 * 4 = 8DD
```

如果 DD=1024，则 token_dim = 8192。

---

### ✅ embedding tubelet 最终 shape

```
all_spatial_embeds.shape = (3128, 8DD)
```

例如 DD=1024：

```
(3128, 8192)
```

---

# 关联总结（你需要记住的结论）

### 对于 16 帧 + 644×476 + patch_size=14：

## 1）不 merge（像素 flatten）

* grid_h=34, grid_w=46
* grid_t=8
* token 数：12512
* token dim：1176

## 2）embedding merge（temporal=2, spatial=2）

* token 数：3128
* token dim：8DD

---

# 如果你实际 patch_size 不是 14

我给你通用公式，你可以自己秒算：

* `grid_h = H // patch_size`
* `grid_w = W // patch_size`
* `P = grid_h * grid_w`
* `grid_t = T // temporal_patch_size`

像素 flatten token：

* `N = grid_t * grid_h * grid_w`
* `D = C * temporal_patch_size * patch_size^2`

embedding merge token：

* `N = (T / temporal_merge) * (grid_h / spatial_merge) * (grid_w / spatial_merge)`
* `D = DD * temporal_merge * spatial_merge^2`

---

如果你把你项目里的真实参数（`patch_size / temporal_patch_size / visual_spatial_merge_size / visual_temporal_merge_size / DD`）贴出来，我可以帮你把最终 token 数、显存规模（FP16/FP32）、以及 attention 复杂度也一起算出来。


好，我们就用你给的 **total frames = 16**，图像尺寸 **644×476** 来把“temporal patching / spatial patching / merge tubelet”完整算一遍。为了可计算，我需要指定两个关键超参（这类模型里几乎固定）：

* `patch_size = 14`（Qwen2-VL / InternVL2 常见设置，644/14=46 正好整除）
* `temporal_patch_size = 2`（常见 tubelet 时间长度）
* `visual_temporal_merge_size = 2`
* `visual_spatial_merge_size = 2`

如果你实际参数不同，把数替换即可（我最后给公式）。

---

# Part A：像素级 temporal patching（你上一段 np 代码）

## A1. 先算空间 patch 网格 grid_h, grid_w

图像尺寸：`H=476, W=644`

`patch_size = 14`

* `grid_h = 476 // 14 = 34`
* `grid_w = 644 // 14 = 46`

每帧 patch 数：

* `P = grid_h * grid_w = 34 * 46 = 1564`

---

## A2. temporal patching：16 帧按 temporal_patch_size=2 分组

`T = 16`
`temporal_patch_size = 2`

* `grid_t = T / 2 = 8`

所以：时间上被切成 **8 个 tubelet（每个 tubelet 2 帧）**。

---

## A3. 像素 tubelet token 数量

上一段 flatten 输出 token 数是：

```
num_tokens = grid_t * grid_h * grid_w
           = 8 * 34 * 46
           = 12512
```

**结论：12512 个 tubelet tokens。**

---

## A4. 每个 token 的维度（像素 flatten）

上一段的 token dim 是：

```
token_dim = channel * temporal_patch_size * patch_size * patch_size
```

如果 RGB：

* `C=3`
* `temporal_patch_size=2`
* `patch_size=14`

则：

* `token_dim = 3 * 2 * 14 * 14 = 1176`

---

### ✅ 像素 tubelet flatten 最终 shape

```
flatten_patches.shape = (12512, 1176)
```

---

# Part B：embedding 级 tubelet merge（你这段 preprocess_spatial_embeds）

这段不是处理像素了，而是处理视觉 backbone 输出的 patch embedding。

假设 backbone 对每个 patch 输出 embedding dim `DD`（代码里叫 DD，注释写 2D），例如：

* DD = 1024 或 1536（取决于模型）
* 我这里先保留符号 DD，你可以代入真实值

---

## B1. spatial_embeds 的初始 shape

你的注释给了：

* `spatial_embeds: [B, S, P, DD]`
* B=1
* P=1564（上面算出来了）
* S 是时间长度（注意：这里的 S 不一定等于 total frames，通常是 “帧token数” 或 “merge前时间token数”）

为了和你例子对齐，我们用最自然的情况：

* 每帧有一组 patch embedding
* 所以 `S = total_frames = 16`

因此初始为：

```
[1, 16, 1564, DD]
```

---

## B2. temporal merge：visual_temporal_merge_size = 2

代码里要求：

```
npatch_t = S // visual_temporal_merge_size
```

所以：

* `npatch_t = 16 / 2 = 8`

这和上面像素 tubelet 的 `grid_t=8` 是一致的。

---

## B3. spatial merge：visual_spatial_merge_size = 2

空间网格是：

* npatch_h = 34
* npatch_w = 46

merge_size=2 时，空间 token 数会减少 4 倍：

* merged_h = 34 / 2 = 17
* merged_w = 46 / 2 = 23

---

## B4. tubelet token 数量（embedding merge 后）

最终 tubelet token 数是：

```
num_tokens = npatch_t * merged_h * merged_w
           = 8 * 17 * 23
           = 3128
```

注意：这比像素 flatten 的 12512 少 4 倍（因为 spatial merge=2×2）。

---

## B5. 每个 tubelet token 的维度

代码最后：

```
token_dim = DD * visual_temporal_merge_size * visual_spatial_merge_size^2
```

代入：

* temporal_merge=2
* spatial_merge=2 → spatial_merge^2=4

得到：

```
token_dim = DD * 2 * 4 = 8DD
```

如果 DD=1024，则 token_dim = 8192。

---

### ✅ embedding tubelet 最终 shape

```
all_spatial_embeds.shape = (3128, 8DD)
```

例如 DD=1024：

```
(3128, 8192)
```

---

# 关联总结（你需要记住的结论）

### 对于 16 帧 + 644×476 + patch_size=14：

## 1）不 merge（像素 flatten）

* grid_h=34, grid_w=46
* grid_t=8
* token 数：12512
* token dim：1176

## 2）embedding merge（temporal=2, spatial=2）

* token 数：3128
* token dim：8DD

---

# 如果你实际 patch_size 不是 14

我给你通用公式，你可以自己秒算：

* `grid_h = H // patch_size`
* `grid_w = W // patch_size`
* `P = grid_h * grid_w`
* `grid_t = T // temporal_patch_size`

像素 flatten token：

* `N = grid_t * grid_h * grid_w`
* `D = C * temporal_patch_size * patch_size^2`

embedding merge token：

* `N = (T / temporal_merge) * (grid_h / spatial_merge) * (grid_w / spatial_merge)`
* `D = DD * temporal_merge * spatial_merge^2`

---

如果你把你项目里的真实参数（`patch_size / temporal_patch_size / visual_spatial_merge_size / visual_temporal_merge_size / DD`）贴出来，我可以帮你把最终 token 数、显存规模（FP16/FP32）、以及 attention 复杂度也一起算出来。



video_tchw: 1 torch.Size([16, 3, 476, 644])
pixel_values_videos: torch.Size([12512, 1176])
n_video_tokens: 3128
spatial_embeds_list: 1 torch.Size([16, 1569, 2048])
patch_start_idx: [5]
camera_encs: 1 4 torch.Size([16, 9])
video_embeds: torch.Size([3128, 2048])
video_grid_thw: tensor([[ 8, 34, 46]], device='cuda:0')
fused_embeds: torch.Size([3128, 2048])
inputs_embeds: torch.Size([1, 3187, 2048])
position_ids: torch.Size([3, 1, 3187])
inputs_embeds: torch.Size([1, 3187, 2048])
use_cache: True
Convert camera_encs to Ks and viewmats:
position_ids: torch.Size([3, 1, 1])
inputs_embeds: torch.Size([1, 1, 2048])
use_cache: True
Convert camera_encs to Ks and viewmats:
position_ids: torch.Size([3, 1, 1])
inputs_embeds: torch.Size([1, 1, 2048])
use_cache: True
Convert camera_encs to Ks and viewmats:
position_ids: torch.Size([3, 1, 1])
inputs_embeds: torch.Size([1, 1, 2048])
use_cache: True
Convert camera_encs to Ks and viewmats:
position_ids: torch.Size([3, 1, 1])
inputs_embeds: torch.Size([1, 1, 2048])
use_cache: True
Convert camera_encs to Ks and viewmats:
video_tchw: 1 torch.Size([16, 3, 476, 644])
pixel_values_videos: torch.Size([12512, 1176])
n_video_tokens: 3128
spatial_embeds_list: 1 torch.Size([16, 1569, 2048])
patch_start_idx: [5]
camera_encs: 1 4 torch.Size([16, 9])
video_embeds: torch.Size([3128, 2048])
video_grid_thw: tensor([[ 8, 34, 46]], device='cuda:0')