import re
from typing import Iterable, Optional

import torch

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

def _extract_answer(text: str) -> str:
    """Return content inside the last <answer>...</answer> block if it exists."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1] if matches else text


def clean_text(text: str, exclude_chars: Iterable[str] = ("\n", "\r")) -> str:
    """Normalize model output to a simple, comparable string."""
    cleaned = _extract_answer(text)

    for char in exclude_chars:
        cleaned = cleaned.replace(char, " ")

    cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace
    return cleaned.strip().rstrip(".").lower()


def normalize_number(num_str: str) -> Optional[float]:
    """Convert string number to float, handling commas."""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Calculate mean relative accuracy for regression tasks."""
    pred_tensor = torch.as_tensor(pred, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred_tensor - target_tensor) / (torch.abs(target_tensor) + epsilon)

    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


# JJ : extract the leading option letter from MCA answers like "b. table" -> "b"
def _extract_mca_choice(text: str) -> str:
    """Extract leading option letter (a-d) from an MCA answer string."""
    text = text.strip()
    m = re.match(r"^([a-d])(?:[\.\)\s:,]|$)", text)
    return m.group(1) if m else text


def vsi_reward(clean_ans_gt: str, clean_ans_pred: str, question_type: str) -> float:
    """Calculate reward based on question type and model output."""
    if question_type in MCA_QUESTION_TYPES:
        # JJ : use extracted choice letter for robust MCA comparison: support qwen3 answer with "B. xxx" format.
        gt_choice = _extract_mca_choice(clean_ans_gt.strip())
        pred_choice = _extract_mca_choice(clean_ans_pred.strip())
        return 1.0 if pred_choice == gt_choice else 0.0

    if question_type in NA_QUESTION_TYPES:
        gt_number = normalize_number(clean_ans_gt)
        pred_number = normalize_number(clean_ans_pred)
        if gt_number is None or pred_number is None:
            return 0.0
        return mean_relative_accuracy(pred_number, gt_number)

    raise ValueError(f"Unsupported question type: {question_type}")