import json
import tempfile
import unittest
from pathlib import Path
import zipfile

from analyze_author_final_baseline_test import (
    MODEL_ORDER,
    OFFICE_PUBLIC_MEMBERS,
    align_results,
    codebook_compatibility,
    comparison,
    public_data_audit,
    verdict,
)


def row(row_index: int, target: str, predictions: list[str]) -> dict:
    return {"row_index": row_index, "target_sid": target, "predict": predictions, "history_sids": []}


class AuthorFinalBaselineTests(unittest.TestCase):
    def test_alignment_reorders_and_preserves_targets(self):
        aligned = align_results(
            {
                "author": [row(2, "b", ["b"]), row(1, "a", ["x"])],
                "reproduction": [row(1, "a", ["a"]), row(2, "b", ["x"])],
            }
        )
        self.assertEqual([item["row_index"] for item in aligned["author"]], [1, 2])
        self.assertEqual([item["target_sid"] for item in aligned["author"]], ["a", "b"])

    def test_comparison_and_verdict(self):
        baseline = [row(1, "a", ["x"]), row(2, "b", ["x"])]
        candidate = [row(1, "a", ["a"]), row(2, "b", ["b"])]
        result = comparison(baseline, candidate, samples=500, seed=42)
        self.assertGreater(result["NDCG@10"]["delta"], 0.0)
        self.assertEqual(verdict(result["NDCG@10"]), "improved")

    def test_model_order_is_three_stage3_models(self):
        self.assertEqual(MODEL_ORDER, ("author_stage3", "paper_stage3", "lora_no_constrained"))

    def test_codebook_compatibility_detects_missing_sid_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            author = root / "author"
            reference = root / "reference"
            author.mkdir()
            reference.mkdir()
            (author / "added_tokens.json").write_text(json.dumps({"<a_1>": 1, "<b_0>": 2, "<c_0>": 3}))
            (reference / "added_tokens.json").write_text(
                json.dumps({"<a_0>": 1, "<a_1>": 2, "<b_0>": 3, "<c_0>": 4})
            )
            result = codebook_compatibility(str(author), str(reference))
            self.assertFalse(result["compatible"])
            self.assertEqual(result["missing_from_author_a_tokens"], ["<a_0>"])

    def test_public_data_audit_compares_bytes_and_index_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            author = root / "author"
            local_data = root / "Amazon"
            author.mkdir()
            (author / "added_tokens.json").write_text(
                json.dumps({"<a_1>": 1, "<b_0>": 2, "<c_0>": 3})
            )
            locations = {
                "test": local_data / "test" / "Office_Products_5_2016-10-2018-11.csv",
                "info": local_data / "info" / "Office_Products_5_2016-10-2018-11.txt",
                "index": local_data / "index" / "Office_Products.index.json",
                "item": local_data / "index" / "Office_Products.item.json",
            }
            values = {
                "test": b"test rows\n",
                "info": b"<a_1><b_0><c_0>\titem\t1\n",
                "index": json.dumps({"1": ["<a_0>", "<b_0>", "<c_0>"]}).encode(),
                "item": b'{"1": {}}\n',
            }
            archive = root / "Amazon.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                for key, member in OFFICE_PUBLIC_MEMBERS.items():
                    locations[key].parent.mkdir(parents=True, exist_ok=True)
                    locations[key].write_bytes(values[key])
                    zip_file.writestr(member, values[key])
            audit = public_data_audit(archive, local_data, str(author))
            self.assertTrue(audit["all_office_assets_byte_identical_to_local"])
            self.assertEqual(audit["missing_index_a_tokens_from_author_tokenizer"], ["<a_0>"])
            self.assertFalse(audit["author_can_represent_all_public_index_tokens"])


if __name__ == "__main__":
    unittest.main()
