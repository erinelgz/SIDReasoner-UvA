import importlib.util
import tempfile
import unittest
from pathlib import Path

from verl.model_merger.base_model_merger import BaseModelMerger


class MergerTests(unittest.TestCase):
    def test_merger_module_imports_without_vision_auto_model(self):
        self.assertIsNotNone(BaseModelMerger)
        path = Path(__file__).resolve().parents[1] / "scripts" / "merge_fsdp_checkpoint.py"
        spec = importlib.util.spec_from_file_location("merge_fsdp_checkpoint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module._merge_lora))
        self.assertTrue(callable(module._merge_fsdp))

    def test_archived_raw_actor_directory_is_accepted(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "merge_fsdp_checkpoint.py"
        spec = importlib.util.spec_from_file_location("merge_fsdp_checkpoint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary_directory:
            actor_dir = Path(temporary_directory) / "raw_actor"
            actor_dir.mkdir()
            (actor_dir / "fsdp_config.json").write_text("{}")
            self.assertEqual(module._resolve_actor_dir(actor_dir), actor_dir)


if __name__ == "__main__":
    unittest.main()
