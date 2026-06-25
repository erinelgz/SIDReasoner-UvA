"""Merge VERL FSDP or LoRA actor checkpoints into a standalone Hugging Face model.

Example
-------
python scripts/merge_fsdp_checkpoint.py \
    --checkpoint checkpoints/gsm8k_async_rl/qwen3-1.7b_Agentic-CRS_async-sgl-multi-w-tool-verify-n16-2cards/global_step_20

Outputs the merged model under ``global_step_20/merged`` so it can be used for
single-GPU evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a VERL actor checkpoint into standalone HF format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to global_step_xx directory or its actor subfolder produced by VERL training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Durable directory in which to write the standalone model.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base Hugging Face model used by a LoRA checkpoint. Required when it cannot be read from the adapter.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "fsdp", "lora"),
        default="auto",
        help="Checkpoint type. Auto selects LoRA when actor/lora_adapter exists.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to transformers AutoConfig when loading tokenizer/config files.",
    )
    parser.add_argument(
        "--use-cpu-init",
        action="store_true",
        help="Initialise the transformers model on CPU before loading weights (helps for large models).",
    )
    return parser.parse_args()


def _resolve_actor_dir(checkpoint: Path) -> Path:
    """Return the actor directory whether the input is global_step or actor itself."""
    if (checkpoint / "fsdp_config.json").is_file():
        return checkpoint
    actor_dir = checkpoint / "actor"
    if actor_dir.is_dir():
        return actor_dir
    raise FileNotFoundError(f"Could not locate actor directory under {checkpoint}")


def _merge_lora(actor_dir: Path, output_dir: Path, base_model: str | None) -> None:
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = actor_dir / "lora_adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Expected LoRA adapter config at {adapter_config_path}")

    with adapter_config_path.open() as f:
        adapter_config = json.load(f)
    base_model = base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("--base-model is required because the adapter does not record its base model")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)


def _merge_fsdp(actor_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"Expected Hugging Face config files under {hf_dir}")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(output_dir),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=args.trust_remote_code,
        is_value_model=False,
        local_dir=str(actor_dir),
        hf_model_config_path=str(hf_dir),
        use_cpu_initialization=args.use_cpu_init,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()


def main() -> None:
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    actor_dir = _resolve_actor_dir(checkpoint_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "lora" if (actor_dir / "lora_adapter" / "adapter_model.safetensors").is_file() else "fsdp"

    if mode == "lora":
        _merge_lora(actor_dir, output_dir, args.base_model)
    else:
        _merge_fsdp(actor_dir, output_dir, args)

    print(f"Merged checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
