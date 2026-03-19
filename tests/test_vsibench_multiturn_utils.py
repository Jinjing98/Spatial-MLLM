import random
import tempfile
import unittest
import json
from pathlib import Path

from src.evaluation.vsibench.multiturn_utils import (
    ContextAnswerMode,
    build_multiturn_messages,
    build_scene_question_index,
    format_vsibench_question,
    qa2description,
    resolve_video_content,
    sample_context_items,
)


class MultiTurnUtilsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.video_root = Path(self.tmpdir.name)
        dataset_dir = self.video_root / "arkitscenes"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "sceneA.mp4").write_bytes(b"fake")

        self.target_item = {
            "sample_idx": 0,
            "dataset": "arkitscenes",
            "scene_name": "sceneA",
            "question_type": "object_counting",
            "question": "How many chairs are visible?",
            "ground_truth": "2",
        }
        self.context_item_1 = {
            "sample_idx": 1,
            "dataset": "arkitscenes",
            "scene_name": "sceneA",
            "question_type": "object_abs_distance",
            "question": "How far is the sofa from the stove?",
            "ground_truth": "3.5",
        }
        self.context_item_2 = {
            "sample_idx": 2,
            "dataset": "arkitscenes",
            "scene_name": "sceneA",
            "question_type": "object_rel_direction_easy",
            "question": "Where is the chair relative to the table?",
            "options": ["A. left", "B. right", "C. front", "D. back"],
            "ground_truth": "B",
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_format_vsibench_question_handles_na_and_mca(self):
        na_question = format_vsibench_question(self.target_item)
        self.assertIn("How many chairs are visible?", na_question)
        self.assertIn("Please answer the question using a single word or phrase.", na_question)

        mca_question = format_vsibench_question(self.context_item_2)
        self.assertIn("Options:\nA. left", mca_question)
        self.assertIn("Answer with the option's letter from the given choices directly.", mca_question)

    def test_resolve_video_content_prefers_mp4_and_sets_nframes(self):
        video_content, debug_path = resolve_video_content(
            self.target_item,
            self.video_root,
            video_nframes=16,
            sample_fps=None,
        )
        self.assertEqual(video_content["type"], "video")
        self.assertEqual(video_content["nframes"], 16)
        self.assertTrue(debug_path.endswith("sceneA.mp4"))

    def test_sample_context_items_excludes_target_and_caps_count(self):
        scene_index = build_scene_question_index([self.target_item, self.context_item_1, self.context_item_2])
        selected = sample_context_items(self.target_item, scene_index, num_context=2, rng=random.Random(0))
        self.assertEqual(len(selected), 2)
        self.assertEqual({item["sample_idx"] for item in selected}, {1, 2})

    def test_build_multiturn_messages_adds_assistant_history_and_target_last(self):
        messages, debug_info = build_multiturn_messages(
            target_item=self.target_item,
            context_items=[self.context_item_1, self.context_item_2],
            video_dir=self.video_root,
            video_nframes=8,
            sample_fps=None,
        )
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1], {"role": "assistant", "content": "3.5"})
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[3], {"role": "assistant", "content": "B"})
        self.assertEqual(messages[4]["role"], "user")
        self.assertEqual(len(debug_info["context_examples"]), 2)
        self.assertTrue(debug_info["target_video_path"].endswith("sceneA.mp4"))

    def test_qa2description_route_planning(self):
        item = {
            "question_type": "route_planning",
            "question": (
                "You are a robot beginning at the bed facing the tv. "
                "You want to navigate to the toilet."
            ),
            "ground_truth": "C",
            "options": [
                "A. Turn Back, Turn Left",
                "B. Turn Left, Turn Left",
                "C. Turn Left, Turn Right",
                "D. Turn Right, Turn Right",
            ],
        }
        converted = qa2description(item)
        self.assertIn("based on the visual inputs,", converted.lower())
        self.assertIn("assume i am a robot beginning at the bed and facing the tv.", converted.lower())
        self.assertIn("1. go forward until the tv", converted.lower())
        self.assertIn("3. go forward until the toilet.", converted.lower())
        self.assertIn("turn left, turn right", converted.lower())

    def test_qa2description_route_planning_fill_in_template(self):
        item = {
            "question_type": "route_planning",
            "question": (
                "You are a robot beginning at the toilet and facing the bathtub. "
                "You want to navigate to the towel rack. "
                "You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): "
                "1. Go forward until the bathtub "
                "2. [please fill in]. "
                "3. Go forward until the towel rack. "
                "You have reached the final destination."
            ),
            "ground_truth": "C",
            "options": ["A. Turn Left", "B. Turn Back", "C. Turn Right"],
        }
        converted = qa2description(item)
        self.assertIn("assume i am a robot beginning at the toilet and facing the bathtub.", converted.lower())
        self.assertIn("i want to navigate to the towel rack.", converted.lower())
        self.assertIn("1. go forward until the bathtub", converted.lower())
        self.assertIn("2. turn right", converted.lower())
        self.assertIn("3. go forward until the towel rack.", converted.lower())

    def test_qa2description_rel_direction_includes_anchor_object(self):
        item = {
            "question_type": "object_rel_direction_easy",
            "question": "If I am standing by the dishwasher and facing the refrigerator, is the washer to the left or the right of the refrigerator?",
            "ground_truth": "B",
            "options": ["A. right", "B. left"],
        }
        converted = qa2description(item)
        self.assertIn("the washer is to the left of the refrigerator.", converted.lower())

    def test_qa2description_rel_direction_medium_uses_egocentric_phrase(self):
        item = {
            "question_type": "object_rel_direction_medium",
            "question": (
                "If I am standing by the sofa and facing the tv, is the fireplace to my left, right, or back? "
                "An object is to my back if I would have to turn at least 135 degrees in order to face it."
            ),
            "ground_truth": "B",
            "options": ["A. left", "B. right", "C. back"],
        }
        converted = qa2description(item)
        self.assertIn("the fireplace is on my right.", converted.lower())

    def test_qa2description_rel_direction_hard_uses_egocentric_phrase(self):
        item = {
            "question_type": "object_rel_direction_hard",
            "question": (
                "If I am standing by the window and facing the sofa, is the lamp to my front-left, front-right, back-left, or back-right? "
                "The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis)."
            ),
            "ground_truth": "B",
            "options": ["A. front-right", "B. front-left", "C. back-right", "D. back-left"],
        }
        converted = qa2description(item)
        self.assertIn("the lamp is on my front-left.", converted.lower())

    def test_qa2description_object_size_extracts_object_name_only(self):
        item = {
            "question_type": "object_size_estimation",
            "question": "What is the length of the longest dimension (length, width, or height) of the whiteboard, measured in centimeters?",
            "ground_truth": "202",
        }
        converted = qa2description(item)
        self.assertIn("the longest dimension of the whiteboard is 202 centimeters.", converted.lower())

    def test_qa2description_appearance_order_clean_wording(self):
        item = {
            "question_type": "obj_appearance_order",
            "question": "What will be the first-time appearance order of the following categories in the video: laptop, bed, basket, shoes?",
            "ground_truth": "C",
            "options": [
                "A. basket, laptop, bed, shoes",
                "B. basket, bed, laptop, shoes",
                "C. bed, shoes, laptop, basket",
                "D. laptop, bed, basket, shoes",
            ],
        }
        converted = qa2description(item)
        self.assertIn("the first-time appearance order is bed, shoes, laptop, basket.", converted.lower())
        self.assertNotIn("mentioned options", converted.lower())

    def test_build_multiturn_messages_qa_to_description_mode(self):
        messages, _ = build_multiturn_messages(
            target_item=self.target_item,
            context_items=[self.context_item_1, self.context_item_2],
            video_dir=self.video_root,
            video_nframes=8,
            sample_fps=None,
            context_answer_mode=ContextAnswerMode.QA_TO_DESCRIPTION,
        )
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("based on the visual inputs,", messages[1]["content"].lower())
        self.assertIn("meters", messages[1]["content"])
        self.assertIn("right", messages[3]["content"].lower())

    def test_qa2description_unknown_type_raises(self):
        item = {
            "question_type": "unknown_type_xyz",
            "question": "Some unknown template",
            "ground_truth": "B",
            "options": ["A. alpha", "B. beta"],
        }
        with self.assertRaises(ValueError) as ctx:
            qa2description(item)
        self.assertIn("unknown_type_xyz", str(ctx.exception))

    def test_qa2description_bad_options_raises(self):
        item = {
            "question_type": "route_planning",
            "question": (
                "You are a robot beginning at the desk facing the lamp. "
                "You want to navigate to the door."
            ),
            "ground_truth": "C",
            "options": ["turn left", "turn right"],
        }
        with self.assertRaises(ValueError) as ctx:
            qa2description(item)
        self.assertIn("Cannot map option letter", str(ctx.exception))

    def test_qa2description_all_vsibench_jsonl_questions_parse(self):
        dataset_path = Path("datasets/vsibench/test.jsonl")
        if not dataset_path.exists():
            self.skipTest(f"VSIBench jsonl not found: {dataset_path}")

        question_types_seen = set()
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                question_types_seen.add(sample.get("question_type", ""))

                ground_truth = str(sample.get("ground_truth", "")).strip()
                if not ground_truth:
                    self.fail(f"Missing ground_truth at line {line_no}")

                try:
                    converted = qa2description(sample)
                except Exception as exc:
                    self.fail(
                        "qa2description raised on VSIBench sample: "
                        f"line={line_no}, question_type={sample.get('question_type')}, "
                        f"question={sample.get('question')!r}, error={exc}"
                    )

                self.assertIn("based on the visual inputs,", converted.lower())

        self.assertGreater(len(question_types_seen), 0)


if __name__ == "__main__":
    # unittest.main()
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(".").resolve()))
    from src.evaluation.vsibench.multiturn_utils import qa2description
    p = Path("datasets/vsibench/test.jsonl")
    ok = fail = 0
    for i, line in enumerate(p.open("r", encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        try:
            des = qa2description(item)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERR] line={i} id={item.get('id')} type={item.get('question_type')} err={e}")
        print(f"  Q: {item.get('question')}")
        print(f"  GT: {item.get('ground_truth')}")
        print(f"  OPTIONS: {item.get('options')}")
        print(f"  DESC: {des}")
    print(f"SUMMARY ok={ok} fail={fail}")