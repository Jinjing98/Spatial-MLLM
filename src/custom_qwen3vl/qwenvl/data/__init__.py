import os
import re
from pathlib import Path


def find_repo_path():
    """
    Find the path to a repository based on .git directory.
    """
    file_path = Path(__file__).resolve()
    while file_path != file_path.parent:
        if (file_path / ".git").exists():
            return file_path
        file_path = file_path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")


# REPO_PATH = find_repo_path()

# JJ : Load DATASET_ROOT from environment variable
if "DATASET_ROOT" not in os.environ:
    raise ValueError(
        "DATASET_ROOT environment variable is not set. "
        "Please set it in your training script or shell environment."
    )

DATASET_ROOT = os.environ["DATASET_ROOT"]
print(f"[INFO] DATASET_ROOT: {DATASET_ROOT}")

if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(
        f"DATASET_ROOT directory does not exist: {DATASET_ROOT}"
    )


### Spatial-MLLM-Mix Data
SPATIAL_MLLM_MIX_10_DBG = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/spatial-mllm-mix-10-dbg.jsonl"),
    "data_path": DATASET_ROOT,
}

SPATIAL_MLLM_MIX_133K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/spatial-mllm-mix-133k.jsonl"),
    "data_path": DATASET_ROOT,
}

SPATIAL_MLLM_MIX_203K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/spatial-mllm-mix-203k.jsonl"),
    "data_path": DATASET_ROOT,
}

# Route Plan Data From VLM-3R
ROUTE_PLAN_SCANNET_2K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/routeplan-2k.jsonl"),
    "data_path": DATASET_ROOT,
}

ROUTE_PLAN_4K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/routeplan-4k.jsonl"),
    "data_path": DATASET_ROOT,
}

### VSI-590K Data From Cambrian-S
VSI_590K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/vsi-590k-processed.jsonl"),
    "data_path": DATASET_ROOT,
}

### MindCube Data
MINDCUBE_21K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/mindcube-processed.jsonl"),
    "data_path": DATASET_ROOT,
}

### JJ: SQA3D Data
SQA3D_FILTERED_40K = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/sqa3d_filtered_40k.jsonl"),
    "data_path": DATASET_ROOT,
}

SQA3D_FILTERED_40K_SMALL = {
    "annotation_path": os.path.join(DATASET_ROOT, "annotations/sqa3d_filtered_40k_small.jsonl"),
    "data_path": DATASET_ROOT,
}

### JJ: ViCA-322K Dataset Configurations

# Video root for all ViCA datasets
VICA_VIDEO_ROOT = os.path.join(DATASET_ROOT, "video")

# Define base and complex tasks
VICA_BASE_TASKS = [
    "obj_appearance_order",
    "object_abs_distance", 
    "object_count",
    "object_relative_distance",
    "object_size_estimation",
    "room_size"
]

VICA_COMPLEX_TASKS = [
    "conversation",
    "furniture",
    "important_daily_necessities",
    "spatial_description",
    "usage",
    "wheelchair_user"
]

# ARKitScenes has additional task
ARKITSCENES_EXTRA_TASKS = ["triangular_positional_relationship"]

# Data sources
VICA_SOURCES = ["arkitscenes", "scannet", "scannetpp"]


def _generate_vica_configs():
    """Auto-generate all ViCA-322K dataset configurations"""
    configs = {}
    
    for source in VICA_SOURCES:
        # Base tasks
        tasks = VICA_BASE_TASKS.copy()
        if source == "arkitscenes":
            tasks.extend(ARKITSCENES_EXTRA_TASKS)
        
        for task in tasks:
            # Full dataset
            key = f"vica_322k_{source}/base/{task}"
            configs[key] = {
                "annotation_path": os.path.join(DATASET_ROOT, source, "base", f"{task}.json"),
                "data_path": DATASET_ROOT,
                "video_root": VICA_VIDEO_ROOT,
            }
            # Small variant (for debugging)
            small_key = f"vica_322k_{source}/base/{task}_small"
            configs[small_key] = {
                "annotation_path": os.path.join(DATASET_ROOT, source, "base", f"{task}_small.json"),
                "data_path": DATASET_ROOT,
                "video_root": VICA_VIDEO_ROOT,
            }
        
        # Complex tasks
        for task in VICA_COMPLEX_TASKS:
            key = f"vica_322k_{source}/complex/{task}"
            configs[key] = {
                "annotation_path": os.path.join(DATASET_ROOT, source, "complex", f"{task}.json"),
                "data_path": DATASET_ROOT,
                "video_root": VICA_VIDEO_ROOT,
            }
    
    return configs


# Generate all ViCA configs
_vica_configs = _generate_vica_configs()

# Define dataset groups for easy access
# JJ: Exclude _small variants from training groups (only for overfitting tests)
VICA_DATASET_GROUPS = {
    # All ViCA data (exclude _small for official training)
    "vica_322k_all": [k for k in _vica_configs.keys() if "_small" not in k],
    
    # By source (exclude _small)
    "vica_322k_arkitscenes": [k for k in _vica_configs.keys() if k.startswith("vica_322k_arkitscenes/") and "_small" not in k],
    "vica_322k_scannet": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannet/") and "_small" not in k],
    "vica_322k_scannetpp": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannetpp/") and "_small" not in k],
    
    # By task type (exclude _small)
    "vica_322k_base": [k for k in _vica_configs.keys() if "/base/" in k and "_small" not in k],
    "vica_322k_complex": [k for k in _vica_configs.keys() if "/complex/" in k],  # complex tasks don't have _small
    
    # Combinations (exclude _small)
    "vica_322k_arkitscenes_base": [k for k in _vica_configs.keys() if k.startswith("vica_322k_arkitscenes/base/") and "_small" not in k],
    "vica_322k_arkitscenes_complex": [k for k in _vica_configs.keys() if k.startswith("vica_322k_arkitscenes/complex/")],
    "vica_322k_scannet_base": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannet/base/") and "_small" not in k],
    "vica_322k_scannet_complex": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannet/complex/")],
    "vica_322k_scannetpp_base": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannetpp/base/") and "_small" not in k],
    "vica_322k_scannetpp_complex": [k for k in _vica_configs.keys() if k.startswith("vica_322k_scannetpp/complex/")],
}

data_dict = {
    "spatial_mllm_mix_10_dbg": SPATIAL_MLLM_MIX_10_DBG,
    "spatial_mllm_mix_133k": SPATIAL_MLLM_MIX_133K,
    "spatial_mllm_mix_203k": SPATIAL_MLLM_MIX_203K,
    "route_plan_scannet_2k": ROUTE_PLAN_SCANNET_2K,
    "route_plan_4k": ROUTE_PLAN_4K,
    "vsi_590k": VSI_590K,
    "mindcube_21k": MINDCUBE_21K,
    "sqa3d_filtered_40k": SQA3D_FILTERED_40K,
    "sqa3d_filtered_40k_small": SQA3D_FILTERED_40K_SMALL,
    # JJ: Add all ViCA configs
    **_vica_configs,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    """
    Parse dataset names and return config list.
    Supports:
    - Individual datasets: "sqa3d_filtered_40k"
    - Dataset groups: "vica_322k_all"
    - Sampling rates: "vica_322k_all%50"
    - Multiple datasets: ["ds1", "ds2%30", "group1"]
    """
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        
        # JJ: Check if it's a ViCA dataset group
        if dataset_name in VICA_DATASET_GROUPS:
            # Expand group to individual datasets
            expanded_datasets = VICA_DATASET_GROUPS[dataset_name]
            for ds in expanded_datasets:
                if ds in data_dict:
                    config = data_dict[ds].copy()
                    config["sampling_rate"] = sampling_rate
                    config_list.append(config)
        elif dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    # Test group expansion
    configs = data_list(["vica_322k_arkitscenes%50"])
    print(f"Expanded to {len(configs)} datasets")
    for config in configs[:3]:  # Show first 3
        print(config)
