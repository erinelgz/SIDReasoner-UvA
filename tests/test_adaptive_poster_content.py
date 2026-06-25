import tempfile
import unittest
from pathlib import Path

from generate_adaptive_poster_content import (
    ADAPTIVE_PROCESS_BULLET,
    PosterCaseDefinition,
    VALIDATION_BENCHMARK_NOTE,
    build_case,
    change_kind,
    load_sid_titles,
    ndcg_at_10,
    render_markdown,
    target_rank,
    validate_case,
)


def result(row_index, target, predictions, trace="saved trace"):
    return {
        "row_index": row_index,
        "target_sid": target,
        "predict": predictions,
        "reasonings": [{"text": trace}],
        "reasoning_token_count": 12,
    }


class AdaptivePosterContentTests(unittest.TestCase):
    def setUp(self):
        self.definition = PosterCaseDefinition(
            case_id="helpful-case",
            model_key="paper_stage3",
            outcome="Reasoning helps",
            row_index=0,
            history_indices=(0,),
            expected_direct_rank=None,
            expected_generated_rank=1,
            trace_summary="The trace selects the matching variant.",
        )
        self.test_row = {
            "history_item_title": "['History item']",
            "history_item_sid": "['history-sid']",
            "item_title": "Target item",
            "item_sid": "target-sid",
        }
        self.titles = {
            "history-sid": "History item",
            "target-sid": "Target item",
            "wrong-sid": "Wrong item",
        }

    def test_rank_metrics_and_change_classification(self):
        direct = result(0, "target-sid", ["wrong-sid"])
        generated = result(0, "target-sid", ["target-sid"])
        self.assertIsNone(target_rank(direct))
        self.assertEqual(target_rank(generated), 1)
        self.assertEqual(ndcg_at_10(direct), 0.0)
        self.assertEqual(ndcg_at_10(generated), 1.0)
        self.assertEqual(change_kind(direct, generated), "helped")

    def test_build_case_preserves_trace_and_maps_titles(self):
        direct = result(0, "target-sid", ["wrong-sid"])
        generated = result(0, "target-sid", ["target-sid"], trace="complete model trace")
        case = build_case(self.definition, direct, generated, self.test_row, self.titles)
        self.assertEqual(case["target"]["title"], "Target item")
        self.assertEqual(case["direct"]["top_recommendation"]["title"], "Wrong item")
        self.assertEqual(case["generated_reasoning"]["full_generated_trace"], "complete model trace")
        self.assertEqual(case["generated_reasoning"]["ndcg_change"], 1.0)

    def test_case_validation_rejects_changed_rank_transition(self):
        direct = result(0, "target-sid", ["target-sid"])
        generated = result(0, "target-sid", ["target-sid"])
        with self.assertRaisesRegex(ValueError, "expected rank transition"):
            validate_case(self.definition, direct, generated)

    def test_catalog_loader_uses_test_titles_to_override_catalogue(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            info_file = Path(temporary_directory) / "info.txt"
            info_file.write_text("target-sid\tOld catalogue title\t0\n")
            titles = load_sid_titles(info_file, [self.test_row])
        self.assertEqual(titles["target-sid"], "Target item")

    def test_markdown_includes_provenance_and_required_sections(self):
        evidence = {
            "model_summaries": {
                "paper_stage3": {
                    "reasoning_change_counts": {"helped": 105, "harmed": 161, "unchanged": 4600},
                    "generated_minus_direct": {"delta": -0.0057, "ci95": [-0.0083, -0.0032]},
                    "adaptive_gate": {
                        "thinking_rate": 0.275,
                        "reasoning_token_reduction_vs_always_think": 0.725,
                    },
                },
                "lora_no_constrained": {
                    "reasoning_change_counts": {"helped": 142, "harmed": 190, "unchanged": 4534},
                    "generated_minus_direct": {"delta": -0.0047, "ci95": [-0.0076, -0.0018]},
                    "adaptive_gate": {
                        "thinking_rate": 0.302,
                        "reasoning_token_reduction_vs_always_think": 0.697,
                    },
                },
            },
            "cases": [
                {
                    "model_label": "Reproduced Stage-3",
                    "outcome": "Reasoning helps",
                    "history": {"selected_titles": ["History item"]},
                    "target": {"title": "Target item"},
                    "direct": {"top_recommendation": {"title": "Wrong item"}, "target_rank": None},
                    "generated_reasoning": {
                        "top_recommendation": {"title": "Target item"},
                        "target_rank": 1,
                        "ndcg_change": 1.0,
                        "poster_trace_summary": "The trace finds the target.",
                    },
                }
            ],
        }
        markdown = render_markdown(evidence)
        self.assertIn("## How It Works", markdown)
        self.assertIn("## Held-Out Office Products Test", markdown)
        self.assertIn("## Illustrative Held-Out Test Cases", markdown)
        self.assertIn(ADAPTIVE_PROCESS_BULLET, markdown)
        self.assertIn(VALIDATION_BENCHMARK_NOTE, markdown)
        self.assertIn("illustrative, not representative", markdown)


if __name__ == "__main__":
    unittest.main()
