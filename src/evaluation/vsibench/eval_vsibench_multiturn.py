import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.evaluation.utils.common_utils import chunk_dataset, load_model_and_processor, save_json, setup_logging
from src.evaluation.vsibench.dataset_utils import clean_text, vsi_reward
from src.evaluation.vsibench.eval_vsibench import calculate_metrics, inference_batch, print_latex_results
from src.evaluation.vsibench.multiturn_utils import (
    ContextAnswerMode,
    build_scene_question_index,
    prepare_multiturn_qwen25_batch,
)


def postprocess_multiturn_batch(
    batch_data: List[Dict],
    batch_output_text: List[str],
    prompts_text: List[str],
    debug_infos: List[Dict],
) -> List[Dict]:
    batch_results = []
    for sample, model_output, prompt, debug_info in zip(batch_data, batch_output_text, prompts_text, debug_infos):
        clean_ans = clean_text(model_output)
        clean_ans_gt = clean_text(sample.get("ground_truth", ""))
        reward = vsi_reward(clean_ans_gt, clean_ans, sample["question_type"])
        batch_results.append(
            {
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "cleaned_model_output": clean_ans,
                "cleaned_gt_answer": clean_ans_gt,
                "reward": reward,
                "correct": reward == 1.0,
                "multiturn_info": {
                    "target_video_path": debug_info["target_video_path"],
                    "target_question": debug_info["target_question"],
                    "context_examples": debug_info["context_examples"],
                },
            }
        )
    return batch_results


def evaluate_vsibench_multiturn(
    vsi_data,
    model_type,
    model_path,
    batch_size,
    video_dir,
    output_path,
    video_nframes,
    sample_fps=None,
    num_context=2,
    random_seed=0,
    debug_leak_target_context=False,
    context_answer_mode=ContextAnswerMode.LETTER_ONLY,
):
    setup_logging()

    if model_type != "qwen2.5-vl":
        raise NotImplementedError("Multi-turn evaluation is currently implemented only for model_type='qwen2.5-vl'.")

    model, processor = load_model_and_processor(model_type, model_path)
    scene_index = build_scene_question_index(vsi_data)
    rng = random.Random(random_seed)
    final_output = []

    for i in tqdm(range(0, len(vsi_data), batch_size), desc="Evaluating VSIBench Multi-Turn"):
        batch_data = vsi_data[i : i + batch_size]
        batch_llm_inputs, prompts_text, debug_infos = prepare_multiturn_qwen25_batch(
            batch_data=batch_data,
            processor=processor,
            model_type=model_type,
            video_dir=video_dir,
            video_nframes=video_nframes,
            sample_fps=sample_fps,
            scene_index=scene_index,
            num_context=num_context,
            rng=rng,
            leak_target_in_context=debug_leak_target_context,
            context_answer_mode=context_answer_mode,
        )
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_multiturn_batch(batch_data, batch_output_text, prompts_text, debug_infos)
        final_output.extend(batch_results)

        if (i + 1) % 10 == 0 or (i + batch_size) >= len(vsi_data):
            save_json(output_path, final_output)

    return final_output


def run_worker(
    gpu_id,
    vsi_data,
    full_vsi_data,
    model_type,
    model_path,
    batch_size,
    video_dir,
    output_path,
    video_nframes,
    sample_fps=None,
    num_context=2,
    random_seed=0,
    debug_leak_target_context=False,
    context_answer_mode=ContextAnswerMode.LETTER_ONLY,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    evaluate_vsibench_multiturn(
        vsi_data=full_vsi_data if batch_size == 1 and len(vsi_data) == len(full_vsi_data) else vsi_data,
        model_type=model_type,
        model_path=model_path,
        batch_size=batch_size,
        video_dir=video_dir,
        output_path=output_path,
        video_nframes=video_nframes,
        sample_fps=sample_fps,
        num_context=num_context,
        random_seed=random_seed,
        debug_leak_target_context=debug_leak_target_context,
        context_answer_mode=context_answer_mode,
    )


def evaluate_worker_chunk(
    gpu_id,
    data_chunk,
    full_vsi_data,
    model_type,
    model_path,
    batch_size,
    video_dir,
    output_path,
    video_nframes,
    sample_fps,
    num_context,
    random_seed,
    debug_leak_target_context,
    context_answer_mode,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_logging()

    if model_type != "qwen2.5-vl":
        raise NotImplementedError("Multi-turn evaluation is currently implemented only for model_type='qwen2.5-vl'.")

    model, processor = load_model_and_processor(model_type, model_path)
    scene_index = build_scene_question_index(full_vsi_data)
    rng = random.Random(random_seed + int(gpu_id))
    final_output = []

    for i in tqdm(range(0, len(data_chunk), batch_size), desc=f"GPU {gpu_id} Multi-Turn"):
        batch_data = data_chunk[i : i + batch_size]
        batch_llm_inputs, prompts_text, debug_infos = prepare_multiturn_qwen25_batch(
            batch_data=batch_data,
            processor=processor,
            model_type=model_type,
            video_dir=video_dir,
            video_nframes=video_nframes,
            sample_fps=sample_fps,
            scene_index=scene_index,
            num_context=num_context,
            rng=rng,
            leak_target_in_context=debug_leak_target_context,
            context_answer_mode=context_answer_mode,
        )
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_multiturn_batch(batch_data, batch_output_text, prompts_text, debug_infos)
        final_output.extend(batch_results)

        if (i + 1) % 10 == 0 or (i + batch_size) >= len(data_chunk):
            save_json(output_path, final_output)


def load_filtered_vsi_data(args) -> List[Dict]:
    annotation_dir = Path(args.annotation_dir).resolve()
    vsi_data = load_dataset(str(annotation_dir), "full")["test"]

    if args.datasets:
        print(f"Filtering dataset to datasets: {args.datasets}")
        vsi_data = vsi_data.filter(lambda x: x["dataset"] in args.datasets)
        print(f"Filtered dataset size: {len(vsi_data)}")

    if args.question_types:
        print(f"Filtering dataset to question types: {args.question_types}")
        vsi_data = vsi_data.filter(lambda x: x["question_type"] in args.question_types)
        print(f"Filtered dataset size: {len(vsi_data)}")

    if args.scene_names:
        print(f"Filtering dataset to scene names: {args.scene_names}")
        vsi_data = vsi_data.filter(lambda x: x["scene_name"] in args.scene_names)
        print(f"Filtered dataset size: {len(vsi_data)}")

    return [{**dict(sample), "sample_idx": idx} for idx, sample in enumerate(vsi_data)]


def recompute_and_save_metrics(args, output_dir: Path):
    if args.input_dir and args.skip_eval:
        input_dir = Path(args.input_dir).resolve()
        print(f"Reading existing results from: {input_dir}")
    elif args.input_dir and not args.skip_eval:
        print("Warning: --input_dir is ignored when not using --skip_eval. Reading from output_dir.")
        input_dir = output_dir
    else:
        input_dir = output_dir
        print(f"Reading results from output directory: {input_dir}")

    merged_results_path = input_dir / f"results_{args.model_type}.json"
    shard_paths = sorted(input_dir.glob(f"results_{args.model_type}_*.json"))

    final_output = []
    if shard_paths:
        for path in shard_paths:
            with open(path, "r", encoding="utf-8") as f:
                final_output.extend(json.load(f))
    elif merged_results_path.exists():
        with open(merged_results_path, "r", encoding="utf-8") as f:
            final_output = json.load(f)
    else:
        print(f"Warning: No result files found in {input_dir} for model_type={args.model_type}.")

    if not final_output:
        raise FileNotFoundError(
            f"No result files found in {input_dir} for model_type={args.model_type}. "
            f"Cannot compute metrics without evaluation results."
        )

    if args.datasets:
        original_count = len(final_output)
        final_output = [res for res in final_output if res["sample"].get("dataset") in args.datasets]
        print(f"Filtered by datasets {args.datasets}: {original_count} -> {len(final_output)} samples")

    if args.question_types:
        original_count = len(final_output)
        final_output = [res for res in final_output if res["sample"].get("question_type") in args.question_types]
        print(f"Filtered by question_types {args.question_types}: {original_count} -> {len(final_output)} samples")

    if args.scene_names:
        original_count = len(final_output)
        final_output = [res for res in final_output if res["sample"].get("scene_name") in args.scene_names]
        print(f"Filtered by scene_names {args.scene_names}: {original_count} -> {len(final_output)} samples")

    if not final_output:
        print("Warning: No samples remain after filtering. Saving empty metrics.")
        empty_metrics = calculate_metrics([])
        save_json(merged_results_path, [])
        save_json(output_dir / f"metrics_{args.model_type}.json", empty_metrics)
        print("Finished evaluation for VSIBench multi-turn (no samples).")
        return

    for res in final_output:
        clean_pred = clean_text(res.get("model_output", ""))
        clean_gt = clean_text(res.get("sample", {}).get("ground_truth", ""))
        qtype = res.get("sample", {}).get("question_type", "")
        res["cleaned_model_output"] = clean_pred
        res["cleaned_gt_answer"] = clean_gt
        res["reward"] = vsi_reward(clean_gt, clean_pred, qtype)
        res["correct"] = res["reward"] == 1.0

    final_metrics = calculate_metrics(final_output)
    save_json(merged_results_path, final_output)
    save_json(output_dir / f"metrics_{args.model_type}.json", final_metrics)
    print(f"Finished evaluation for VSIBench multi-turn.")
    print(f"Final Metrics (Overall): {final_metrics}")

    dataset_metrics_dict = {}
    unique_datasets = sorted(set(res["sample"].get("dataset") for res in final_output if res["sample"].get("dataset")))
    for dataset in unique_datasets:
        dataset_results = [res for res in final_output if res["sample"].get("dataset") == dataset]
        if not dataset_results:
            continue
        dataset_metrics = calculate_metrics(dataset_results)
        dataset_metrics_dict[dataset] = dataset_metrics
        save_json(output_dir / f"metrics_{args.model_type}_{dataset}.json", dataset_metrics)

    print_latex_results(final_metrics, args.model_type, dataset_metrics_dict if dataset_metrics_dict else None)


def main(args):
    setup_logging()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.model_type != "qwen2.5-vl":
        raise NotImplementedError("Multi-turn evaluation is currently implemented only for model_type='qwen2.5-vl'.")

    output_dir = Path(args.output_dir).resolve() / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = Path(args.annotation_dir).resolve()
    video_dir = Path(args.video_dir).resolve() if args.video_dir else annotation_dir

    vsi_data = load_filtered_vsi_data(args)

    if not args.skip_eval:
        n_gpu = torch.cuda.device_count()
        if n_gpu <= 0:
            raise RuntimeError("VSIBench evaluation requires at least one CUDA device.")

        print(f"Starting multi-turn evaluation on {n_gpu} GPUs...")
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
        else:
            gpu_ids = [str(i) for i in range(n_gpu)]
        print("GPU IDs:", gpu_ids)

        processes = []
        for idx, data_chunk in enumerate(chunk_dataset(vsi_data, n_gpu)):
            output_path_gpu = output_dir / f"results_{args.model_type}_{idx}.json"
            gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else str(idx)
            p = mp.Process(
                target=evaluate_worker_chunk,
                args=(
                    gpu_id,
                    data_chunk,
                    vsi_data,
                    args.model_type,
                    args.model_path,
                    args.batch_size,
                    video_dir,
                    output_path_gpu,
                    args.nframes,
                    args.sample_fps,
                    args.num_context,
                    args.random_seed,
                    args.debug_leak_target_context,
                    args.context_answer_mode,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"A worker exited with code {p.exitcode}.")
    else:
        print("Skipping evaluation phase (--skip_eval).")

    if not args.skip_metric:
        recompute_and_save_metrics(args, output_dir)
    else:
        print("Skipping metrics computation (--skip_metric).")


def int_or_none(x):
    if x.lower() == "none":
        return None
    return int(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate qwen2.5-vl on VSIBench with fake multi-turn context.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model_type", type=str, default="qwen2.5-vl", help="Model type. Only qwen2.5-vl is supported.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing the VSIBench data files.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory containing video files or frame folders.")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument("--output_name", type=str, default="eval_vsibench_multiturn", help="Subdirectory name for evaluation results.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory to read existing results from when using --skip_eval.")
    parser.add_argument("--question_types", type=str, nargs="+", default=None, help="List of question types to evaluate.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="List of datasets to evaluate.")
    parser.add_argument("--scene_names", type=str, nargs="+", default=None, help="List of scene names to evaluate.")
    parser.add_argument("--nframes", type=int_or_none, default=None, help="Number of frames to sample from each video, or 'None'.")
    parser.add_argument("--sample_fps", type=float, default=None, help="Sample FPS for video. Default uses nframes or source behavior.")
    parser.add_argument("--num_context", type=int, default=2, help="Maximum number of context QA pairs to sample from the same video.")
    parser.add_argument("--random_seed", type=int, default=0, help="Seed used for context sampling.")
    parser.add_argument(
        "--debug_leak_target_context",
        action="store_true",
        default=False,
        help="Debug mode: force one context pair to use the target question and ground-truth answer.",
    )
    parser.add_argument(
        "--context_answer_mode",
        type=str,
        default="letter_only",
        choices=[mode.value for mode in ContextAnswerMode],
        help="Format mode for context MCA answers. Options: letter_only (default), letter_with_text, explicit_reasoning, full_reasoning.",
    )
    parser.add_argument("--skip_eval", action="store_true", default=False, help="Skip the evaluation phase and only compute metrics.")
    parser.add_argument("--skip_metric", action="store_true", default=False, help="Skip metrics computation and only run evaluation.")
    
    args = parser.parse_args()
    # Convert string argument to ContextAnswerMode enum
    args.context_answer_mode = ContextAnswerMode(args.context_answer_mode)
    main(args)