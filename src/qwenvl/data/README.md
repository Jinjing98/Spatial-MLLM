# Dataset Configuration Guide

## Quick Start

Set the dataset in your training script:

```bash
export DATASET_ROOT="/path/to/your/datasets"
DATASETS="dataset_name"
```

## Dataset Naming Conventions

### 1. Individual Datasets

```bash
# SQA3D
DATASETS="sqa3d_filtered_40k"
DATASETS="sqa3d_filtered_40k_small"

# Spatial-MLLM Mix
DATASETS="spatial_mllm_mix_133k"
DATASETS="spatial_mllm_mix_203k"

# ViCA-322K - Specific files (use full path for clarity)
DATASETS="vica_322k_arkitscenes/base/obj_appearance_order"
DATASETS="vica_322k_arkitscenes/base/obj_appearance_order_small"  # Small variant for debugging
DATASETS="vica_322k_scannet/base/object_count"
DATASETS="vica_322k_scannet/complex/conversation"
DATASETS="vica_322k_scannetpp/base/room_size"
```

### 2. ViCA-322K Dataset Groups

```bash
# All ViCA-322K data (~322K samples)
DATASETS="vica_322k_all"

# By data source
DATASETS="vica_322k_arkitscenes"  # All ARKitScenes data
DATASETS="vica_322k_scannet"      # All ScanNet data
DATASETS="vica_322k_scannetpp"    # All ScanNet++ data

# By task type
DATASETS="vica_322k_base"         # All base tasks (6-7 tasks)
DATASETS="vica_322k_complex"      # All complex tasks (6 tasks)

# By source + task type
DATASETS="vica_322k_arkitscenes_base"
DATASETS="vica_322k_arkitscenes_complex"
DATASETS="vica_322k_scannet_base"
DATASETS="vica_322k_scannet_complex"
DATASETS="vica_322k_scannetpp_base"
DATASETS="vica_322k_scannetpp_complex"
```

### 3. Sampling Rates

Use `%N` suffix to sample N% of the data:

```bash
# Use 50% of ViCA data
DATASETS="vica_322k_all%50"

# Use 30% of ARKitScenes base tasks
DATASETS="vica_322k_arkitscenes_base%30"

# Use 20% of a specific dataset
DATASETS="sqa3d_filtered_40k%20"
```

### 4. Multiple Datasets (Comma-separated)

```bash
# Combine datasets
DATASETS="vica_322k_all,sqa3d_filtered_40k"

# Mix with sampling rates
DATASETS="vica_322k_all%50,sqa3d_filtered_40k,spatial_mllm_mix_133k%30"

# Multiple ViCA groups
DATASETS="vica_322k_scannet_base,vica_322k_arkitscenes_complex"
```

## ViCA-322K Task Structure

### Base Tasks (Metadata-grounded)
- `obj_appearance_order` - Object appearance order
- `object_abs_distance` - Absolute distance estimation
- `object_count` - Object counting
- `object_relative_distance` - Relative distance
- `object_size_estimation` - Size estimation
- `room_size` - Room size estimation
- `triangular_positional_relationship` - Triangle geometry (ARKitScenes only)

### Complex Tasks (Language-grounded)
- `conversation` - Multi-turn dialogues
- `furniture` - Furniture reasoning
- `important_daily_necessities` - Daily objects reasoning
- `spatial_description` - Spatial descriptions
- `usage` - Scenario-based planning
- `wheelchair_user` - Accessibility reasoning

## Examples

### Debugging
```bash
DATASETS="vica_322k_arkitscenes/base/obj_appearance_order_small"
```

### Single Task Training
```bash
DATASETS="vica_322k_arkitscenes/base/obj_appearance_order"
```

### Multi-Task Training
```bash
DATASETS="vica_322k_arkitscenes_base"
```

### Full Dataset Training
```bash
DATASETS="vica_322k_all"
```

### Mixed Dataset Training
```bash
DATASETS="vica_322k_all,sqa3d_filtered_40k,spatial_mllm_mix_133k"
```

### Balanced Sampling
```bash
# Equal contribution from each dataset (50% each)
DATASETS="vica_322k_all%50,sqa3d_filtered_40k%50"
```

## Adding New Datasets

### For datasets with standard structure
Edit `__init__.py` and add to `data_dict`:

```python
NEW_DATASET = {
    "annotation_path": os.path.join(DATASET_ROOT, "path/to/annotations.jsonl"),
    "data_path": DATASET_ROOT,
}

data_dict = {
    # ...
    "new_dataset": NEW_DATASET,
}
```

### For datasets with different video/image root
Add `video_root` or `image_root`:

```python
NEW_DATASET = {
    "annotation_path": os.path.join(DATASET_ROOT, "path/to/annotations.jsonl"),
    "data_path": DATASET_ROOT,
    "video_root": os.path.join(DATASET_ROOT, "video"),  # Custom video directory
}
```

## Notes

- Dataset names are case-sensitive
- Sampling rates are applied during data loading (Line 154-158 in `data_qwen.py`)
- All datasets are shuffled after loading (Line 168 in `data_qwen.py`)
- The `DATASET_ROOT` environment variable must be set before training
