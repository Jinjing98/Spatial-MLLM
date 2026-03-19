import glob
import logging
import re
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


class ContextAnswerMode(Enum):
    """Enum for different context answer formatting modes."""
    LETTER_ONLY = "letter_only"  # e.g., "A"
    LETTER_WITH_TEXT = "letter_with_text"  # e.g., "A. left"
    EXPLICIT_REASONING = "explicit_reasoning"  # e.g., "The answer is A. left."
    FULL_REASONING = "full_reasoning"  # e.g., "If I am standing by the stove and facing the stool, is the tv to the left or the right of the stool? The answer is A. left."
    QA_TO_DESCRIPTION = "qa_to_description"  # Convert question + answer into a declarative description.

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


SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "mca": "Answer with the option's letter from the given choices directly.",
    "na": "Please answer the question using a single word or phrase.",
}


def _normalize_prompt_video_placeholders(prompt: str) -> str:
    # Keep saved prompts compact/readable while preserving model inputs unchanged.
    return re.sub(r"(?:<\|video_pad\|>)+", "<|video_pad|>", prompt)


def _normalize_option_letter(answer: str) -> str:
    match = re.search(r"([A-Za-z])", answer)
    return match.group(1).upper() if match else ""


def _format_comprehensive_mca_answer(item: Dict[str, Any]) -> str:
    answer = str(item.get("ground_truth", "")).strip()
    if not answer:
        raise ValueError("Context samples must include a non-empty ground_truth answer.")

    letter = _normalize_option_letter(answer)
    if not letter:
        return answer

    options = item.get("options") or []
    for option in options:
        option_text = str(option).strip()
        if option_text.upper().startswith(f"{letter}.") or option_text.upper().startswith(f"{letter})"):
            return option_text
    return answer


def _option_letter_to_text(options: Sequence[Any]) -> Dict[str, str]:
    letter_to_text: Dict[str, str] = {}
    for option in options:
        option_text = str(option).strip()
        if not option_text:
            continue
        match = re.match(r"^([A-Za-z])[\.)]\s*(.+)$", option_text)
        if match:
            letter_to_text[match.group(1).upper()] = match.group(2).strip()
    return letter_to_text


def _resolve_answer_text(item: Dict[str, Any], answer: str) -> str:
    options = item.get("options") or []
    letter = _normalize_option_letter(answer)
    if not letter:
        return answer
    letter_to_text = _option_letter_to_text(options)
    if letter not in letter_to_text:
        error_msg = (
            f"Cannot map option letter {letter!r} to text. "
            f"ground_truth={answer!r}, available_options={options!r}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return letter_to_text[letter]


def _with_prefix(pre_text: str, sentence: str) -> str:
    prefix = (pre_text or "").strip()
    sentence = sentence.strip()
    if not prefix:
        return sentence
    if not prefix.endswith(","):
        prefix = f"{prefix},"
    return f"{prefix} {sentence}"


def qa2description(item: Dict[str, Any], pre_text: str = "ok. Based on the visual inputs,") -> str:
    """Convert question + ground truth answer into a declarative description."""
    q_type = str(item.get("question_type", "")).strip()
    question = str(item.get("question", "")).strip()
    answer = str(item.get("ground_truth", "")).strip()
    if not answer:
        raise ValueError("Context samples must include a non-empty ground_truth answer.")

    resolved_answer = _resolve_answer_text(item, answer)

    def _raise_parse_error(expected_pattern: str) -> None:
        error_msg = (
            "Failed to parse question template for qa2description. "
            f"question_type={q_type!r}, expected_pattern={expected_pattern!r}, "
            f"question={question!r}, ground_truth={answer!r}, resolved_answer={resolved_answer!r}, "
            f"options={item.get('options')!r}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if q_type == "object_counting":
        match = re.search(r"How many\s+(.+?)\(s\)\s+are\s+in\s+this\s+room\?", question, flags=re.IGNORECASE)
        if match:
            category = match.group(1).strip()
            return _with_prefix(pre_text, f"there are {resolved_answer} {category}(s) in this room.")
        _raise_parse_error("How many <category>(s) are in this room?")

    if q_type == "object_size_estimation":
        match = re.search(
            r"longest\s+dimension(?:\s*\([^)]*\))?\s+of\s+the\s+(.+?),\s+measured\s+in\s+centimeters\?",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"How\s+long\s+is\s+the\s+longest\s+side\s+of\s+the\s+(.+?),\s+measured\s+in\s+centimeters\?",
                question,
                flags=re.IGNORECASE,
            )
        if match:
            category = match.group(1).strip()
            return _with_prefix(pre_text, f"the longest dimension of the {category} is {resolved_answer} centimeters.")
        _raise_parse_error("How long is the longest side of the <object>, measured in centimeters?")

    if q_type == "room_size_estimation":
        return _with_prefix(pre_text, f"the room size is {resolved_answer} square meters.")

    if q_type == "object_abs_distance":
        match = re.search(
            r"distance\s+between\s+the\s+(.+?)\s+and\s+the\s+(.+?)\s*\(in\s+meters\)\?",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"How\s+far\s+is\s+the\s+(.+?)\s+from\s+the\s+(.+?)\?",
                question,
                flags=re.IGNORECASE,
            )
        if match:
            obj_a = match.group(1).strip()
            obj_b = match.group(2).strip()
            return _with_prefix(pre_text, f"the distance between the {obj_a} and the {obj_b} is {resolved_answer} meters.")
        _raise_parse_error("What is the distance between the <obj_a> and the <obj_b> (in meters)?")

    if q_type == "obj_appearance_order":
        return _with_prefix(pre_text, f"the first-time appearance order is {resolved_answer}.")

    if q_type == "object_rel_distance":
        match = re.search(r"closest\s+to\s+the\s+(.+?)\?", question, flags=re.IGNORECASE)
        if match:
            anchor = match.group(1).strip()
            return _with_prefix(pre_text, f"among the listed objects, {resolved_answer} is the closest to the {anchor}.")
        _raise_parse_error("Which object is closest to the <anchor>?")

    if q_type in {
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
    }:
        match = re.search(
            r"If\s+I\s+am\s+standing\s+by\s+the\s+(.+?)\s+and\s+facing\s+the\s+(.+?),\s+is\s+the\s+(.+?)\s+",
            question,
            flags=re.IGNORECASE,
        )
        if match:
            position = match.group(1).strip()
            facing = match.group(2).strip()
            query = match.group(3).strip()
            if q_type in {"object_rel_direction_medium", "object_rel_direction_hard"}:
                return _with_prefix(
                    pre_text,
                    f"if standing by the {position} and facing the {facing}, the {query} is on my {resolved_answer}.",
                )
            return _with_prefix(
                pre_text,
                f"if standing by the {position} and facing the {facing}, the {query} is to the {resolved_answer} of the {facing}.",
            )
        match = re.search(
            r"Where\s+is\s+the\s+(.+?)\s+relative\s+to\s+the\s+(.+?)\?",
            question,
            flags=re.IGNORECASE,
        )
        if match:
            query = match.group(1).strip()
            anchor = match.group(2).strip()
            return _with_prefix(pre_text, f"the {query} is {resolved_answer} relative to the {anchor}.")
        _raise_parse_error("If I am standing by the <position> and facing the <facing>, is the <query> ...?")

    if q_type == "route_planning":
        match = re.search(
            r"beginning\s+(?:at|by)\s+the\s+(.+?)\s+(?:and\s+)?facing(?:\s+to)?\s+(?:the\s+)?(.+?)\.\s+You\s+want\s+to\s+navigate\s+to\s+the\s+(.+?)\.",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"beginning\s+at\s+the\s+(.+?),\s+with\s+your\s+back\s+to\s+the\s+(.+?)\.\s+You\s+want\s+to\s+navigate\s+to\s+the\s+(.+?)\.",
                question,
                flags=re.IGNORECASE,
            )
        if match:
            start = match.group(1).strip()
            facing = match.group(2).strip()
            destination = match.group(3).strip()
            return _with_prefix(
                pre_text,
                (
                    f"assume i am a robot beginning at the {start} and facing the {facing}. "
                    f"i want to navigate to the {destination}. "
                    f"to reach the destination, i can perform following actions in order: "
                    f"1. Go forward until the {facing} 2. {resolved_answer} 3. Go forward until the {destination}."
                ),
            )
        _raise_parse_error(
            "You are a robot beginning at the <start> facing the <facing>. You want to navigate to the <destination>."
        )

    error_msg = (
        f"Unknown question_type {q_type!r} — no parser registered for this type. "
        f"question={question!r}, ground_truth={answer!r}, options={item.get('options')!r}"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def format_context_answer(item: Dict[str, Any], mode: ContextAnswerMode = ContextAnswerMode.LETTER_ONLY) -> str:
    """
    Format context answer according to specified mode.
    
    Args:
        item: Sample item with ground_truth and question fields
        mode: ContextAnswerMode enum specifying the answer format
    
    Returns:
        Formatted answer string
    """
    answer = str(item.get("ground_truth", "")).strip()
    if not answer:
        raise ValueError("Context samples must include a non-empty ground_truth answer.")

    if mode == ContextAnswerMode.LETTER_ONLY:
        # Extract just the letter: "A"
        letter = _normalize_option_letter(answer)
        return letter if letter else answer

    if mode == ContextAnswerMode.LETTER_WITH_TEXT:
        # Extract full option text: "A. left"
        letter = _normalize_option_letter(answer)
        if not letter:
            return answer
        options = item.get("options") or []
        for option in options:
            option_text = str(option).strip()
            if option_text.upper().startswith(f"{letter}.") or option_text.upper().startswith(f"{letter})"):
                return option_text
        return answer

    if mode == ContextAnswerMode.EXPLICIT_REASONING:
        # Explicit reasoning: "The answer is A. left."
        letter = _normalize_option_letter(answer)
        if not letter:
            return f"The answer is {answer}."
        options = item.get("options") or []
        for option in options:
            option_text = str(option).strip()
            if option_text.upper().startswith(f"{letter}.") or option_text.upper().startswith(f"{letter})"):
                return f"The answer is {option_text}."
        return f"The answer is {answer}."

    if mode == ContextAnswerMode.FULL_REASONING:
        # Full reasoning: include question + explicit answer
        question = item.get("question", "")
        letter = _normalize_option_letter(answer)
        if not letter:
            return f"{question}. The answer is {answer}."
        options = item.get("options") or []
        for option in options:
            option_text = str(option).strip()
            if option_text.upper().startswith(f"{letter}.") or option_text.upper().startswith(f"{letter})"):
                return f"{question}. The answer is {option_text}."
        return f"{question}. The answer is {answer}."

    if mode == ContextAnswerMode.QA_TO_DESCRIPTION:
        return qa2description(item)

    # Default fallback
    return answer


def format_vsibench_question(item: Dict[str, Any]) -> str:
    raw_question = SFT_QUESTION_TEMPLATE.format(Question=item["question"])
    q_type = item["question_type"]

    if q_type in MCA_QUESTION_TYPES:
        options = item.get("options") or []
        if not options:
            raise ValueError("Multiple-choice samples must include 'options'.")
        options_text = "Options:\n" + "\n".join(options)
        return f"{raw_question}\n{options_text}\n{SFT_TYPE_TEMPLATE['mca']}"

    if q_type in NA_QUESTION_TYPES:
        return f"{raw_question}\n{SFT_TYPE_TEMPLATE['na']}"

    raise ValueError(f"Unknown question type: {q_type}")


def resolve_video_content(
    item: Dict[str, Any],
    video_dir: Path,
    video_nframes: Optional[int],
    sample_fps: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    video_content: Dict[str, Any] = {"type": "video"}
    mp4_path = (video_dir / item["dataset"] / f"{item['scene_name']}.mp4").resolve()
    frame_dir = (video_dir / item["dataset"] / item["scene_name"]).resolve()

    if mp4_path.exists():
        if video_nframes is not None and sample_fps is not None:
            raise ValueError(
                f"Cannot specify both nframes ({video_nframes}) and sample_fps ({sample_fps}). Use one or the other."
            )
        video_content["video"] = str(mp4_path)
        video_content["do_sample_frames"] = (video_nframes is None) and (sample_fps is None)
        if video_nframes is not None:
            video_content["nframes"] = video_nframes
        if sample_fps is not None:
            video_content["fps"] = sample_fps
        return video_content, str(mp4_path)

    if frame_dir.exists():
        frame_paths = sorted(glob.glob(str(frame_dir / "*.png")))
        if not frame_paths:
            raise FileNotFoundError(f"No PNG files found in {frame_dir}")
        if video_nframes is not None and len(frame_paths) != video_nframes:
            raise ValueError(
                f"Number of frames in {frame_dir} ({len(frame_paths)}) does not match expected {video_nframes}."
            )
        video_content["video"] = frame_paths
        return video_content, str(frame_dir)

    raise FileNotFoundError(
        f"Data file not found for dataset={item['dataset']} scene={item['scene_name']} under {video_dir}"
    )


def build_qwen25_user_turn(
    item: Dict[str, Any],
    video_dir: Path,
    video_nframes: Optional[int],
    sample_fps: Optional[float] = None,
) -> Tuple[Dict[str, Any], str, str]:
    question = format_vsibench_question(item)
    video_content, debug_video_path = resolve_video_content(item, video_dir, video_nframes, sample_fps)
    message = {
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": question},
        ],
    }
    return message, question, debug_video_path


def build_context_assistant_turn(
    item: Dict[str, Any],
    answer_mode: ContextAnswerMode = ContextAnswerMode.LETTER_ONLY,
) -> Dict[str, str]:
    if answer_mode == ContextAnswerMode.QA_TO_DESCRIPTION:
        answer = format_context_answer(item, mode=answer_mode)
    elif item.get("question_type") in MCA_QUESTION_TYPES:
        answer = format_context_answer(item, mode=answer_mode)
    else:
        answer = str(item.get("ground_truth", "")).strip()
        if not answer:
            raise ValueError("Context samples must include a non-empty ground_truth answer.")
    return {"role": "assistant", "content": answer}


def build_scene_question_index(samples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    scene_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (sample["dataset"], sample["scene_name"])
        scene_index[key].append(sample)
    return dict(scene_index)


def sample_context_items(
    target_item: Dict[str, Any],
    scene_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    num_context: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_context <= 0:
        return []

    key = (target_item["dataset"], target_item["scene_name"])
    target_id = target_item.get("sample_idx")
    candidates = []

    for candidate in scene_index.get(key, []):
        candidate_id = candidate.get("sample_idx")
        if target_id is not None and candidate_id == target_id:
            continue
        if not str(candidate.get("ground_truth", "")).strip():
            continue
        candidates.append(candidate)

    if len(candidates) <= num_context:
        return list(candidates)
    return rng.sample(candidates, k=num_context)


def build_multiturn_messages(
    target_item: Dict[str, Any],
    context_items: Sequence[Dict[str, Any]],
    video_dir: Path,
    video_nframes: Optional[int],
    sample_fps: Optional[float] = None,
    leak_target_in_context: bool = False,
    context_answer_mode: ContextAnswerMode = ContextAnswerMode.LETTER_ONLY,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    context_examples: List[Dict[str, Any]] = []

    context_items = list(context_items)
    if leak_target_in_context:
        # Debug mode: force one context pair to be exactly the target QA.
        if context_items:
            context_items[0] = target_item
        else:
            context_items = [target_item]

    for context_item in context_items:
        user_turn, question, video_path = build_qwen25_user_turn(
            context_item,
            video_dir,
            video_nframes,
            sample_fps,
        )
        assistant_turn = build_context_assistant_turn(
            context_item,
            answer_mode=context_answer_mode,
        )
        messages.extend([user_turn, assistant_turn])
        context_examples.append(
            {
                "dataset": context_item["dataset"],
                "scene_name": context_item["scene_name"],
                "question_type": context_item["question_type"],
                "video_path": video_path,
                "question": question,
                "answer": assistant_turn["content"],
            }
        )

    target_turn, target_question, target_video_path = build_qwen25_user_turn(
        target_item,
        video_dir,
        video_nframes,
        sample_fps,
    )
    messages.append(target_turn)

    debug_info = {
        "target_video_path": target_video_path,
        "target_question": target_question,
        "context_examples": context_examples,
    }
    return messages, debug_info


def prepare_multiturn_qwen25_batch(
    batch_data: List[Dict[str, Any]],
    processor: Any,
    model_type: str,
    video_dir: Path,
    video_nframes: Optional[int],
    sample_fps: Optional[float],
    scene_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    num_context: int,
    rng: random.Random,
    leak_target_in_context: bool = False,
    context_answer_mode: ContextAnswerMode = ContextAnswerMode.LETTER_ONLY,
) -> Tuple[Any, List[str], List[Dict[str, Any]]]:
    if model_type != "qwen2.5-vl":
        raise NotImplementedError("Multi-turn evaluation is currently implemented only for model_type='qwen2.5-vl'.")

    from qwen_vl_utils import process_vision_info

    batch_messages: List[List[Dict[str, Any]]] = []
    debug_infos: List[Dict[str, Any]] = []
    for item in batch_data:
        context_items = sample_context_items(item, scene_index, num_context, rng)
        messages, debug_info = build_multiturn_messages(
            item,
            context_items,
            video_dir,
            video_nframes,
            sample_fps,
            leak_target_in_context=leak_target_in_context,
            context_answer_mode=context_answer_mode,
        )
        batch_messages.append(messages)
        debug_infos.append(debug_info)

    prompts_text = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True)
        for example in batch_messages
    ]
    prompts_text_for_save = [_normalize_prompt_video_placeholders(p) for p in prompts_text]

    if debug_infos:
        first_debug = debug_infos[0]
        print("[MultiTurn] Target video:", first_debug["target_video_path"])
        print("[MultiTurn] Target question:", first_debug["target_question"])
        if not first_debug["context_examples"]:
            print("[MultiTurn] Context examples: none available")
        for idx, context in enumerate(first_debug["context_examples"], start=1):
            print(
                f"[MultiTurn] Context {idx}: video={context['video_path']} | question={context['question']} | answer={context['answer']}"
            )
        print(f"[MultiTurn] Context answer mode: {context_answer_mode.value}")
        print("[MultiTurn] Sample prompt text:")
        print(prompts_text_for_save[0])

    video_inputs: List[Any] = []
    image_inputs: List[Any] = []
    for example in batch_messages:
        images, videos = process_vision_info(example)
        if images:
            image_inputs.extend(images)
        if videos:
            video_inputs.extend(videos)
        if not images and not videos:
            raise ValueError("Each example must contain at least one image or video.")

    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    return batch, prompts_text_for_save.copy(), debug_infos