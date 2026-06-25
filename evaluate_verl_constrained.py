#!/usr/bin/env python
"""
Evaluate SID generation on the same VERL parquet split used by PPO validation.
"""

import json
import os

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sid_eval_utils import (
    ANSWER_SEPARATOR,
    SID_RE,
    compute_metrics,
    derive_metrics_path,
    extract_generated_sid,
    load_valid_sids,
    load_verl_examples,
    set_seed,
)
from verl.utils.reward_score.direct_recommendation_StepRule_Office import extract_solution
from verl.utils.sid_constraints import make_sid_prefix_allowed_tokens_fn


def main(
    base_model: str = "/home/scur1249/Office_Products_checkpoint/merged",
    parquet_path: str = "./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet",
    info_file: str = "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt",
    index_file: str = "./data/Amazon/index/Office_Products.index.json",
    result_json_data: str = "./temp/test_results_verl_constrained_Qwen3.json",
    metrics_json_data: str | None = None,
    num_samples: int = -1,
    max_new_tokens: int = 32,
    batch_size: int = 16,
    num_beams: int = 10,
    length_penalty: float = 0.0,
    constrained_decoding: bool = True,
    direct_sid_decoding: bool = True,
    save_full_text: bool = False,
    seed: int = 42,
):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 60)
    print("VERL-Parquet SID Evaluation")
    print("=" * 60)
    print(f"Model:         {base_model}")
    print(f"Parquet:       {parquet_path}")
    print(f"SID Catalog:   {index_file}")
    print(f"Samples:       {'all' if num_samples < 0 else num_samples}")
    print(f"Batch size:    {batch_size}")
    print(f"Num beams:     {num_beams}")
    print(f"Max tokens:    {max_new_tokens}")
    print(f"Constrained:   {constrained_decoding}")
    print(f"Direct SID:    {direct_sid_decoding}")
    print("=" * 60 + "\n")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading VERL examples from {parquet_path}...")
    examples = load_verl_examples(
        parquet_path,
        tokenizer=tokenizer,
        num_samples=num_samples,
        seed=seed,
        direct_sid_decoding=direct_sid_decoding,
    )

    valid_sids = load_valid_sids(index_file=index_file, info_file=info_file)
    prefix_allowed_tokens_fn = None
    if constrained_decoding:
        prefix_allowed_tokens_fn = make_sid_prefix_allowed_tokens_fn(
            tokenizer,
            index_file=index_file,
            info_file=info_file,
            eos_token_id=tokenizer.eos_token_id,
            answer_separator=ANSWER_SEPARATOR,
            fail_on_missing_separator=direct_sid_decoding,
        )

    results = []
    print(f"Running inference on {len(examples)} examples...\n")
    for start in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[start : start + batch_size]
        input_texts = [example["input_text"] for example in batch]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[-1]

        try:
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "num_return_sequences": num_beams,
                "early_stopping": True,
                "length_penalty": length_penalty,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if prefix_allowed_tokens_fn is not None:
                generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

            with torch.no_grad():
                output_ids = model.generate(**generate_kwargs)

            for batch_idx, example in enumerate(batch):
                seq_start = batch_idx * num_beams
                seq_end = seq_start + num_beams
                sequences = output_ids[seq_start:seq_end]
                completion_ids = sequences[:, prompt_len:]
                completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                generated_sids = [extract_generated_sid(text) for text in completion_texts]
                generated_sids = list(dict.fromkeys(generated_sids))
                generated_sid = generated_sids[0] if generated_sids else ""
                first_completion = completion_texts[0].strip() if completion_texts else ""
                full_generated_text = tokenizer.decode(sequences[0], skip_special_tokens=False)
                reasoning_open = full_generated_text.find("<think>")
                reasoning_close = full_generated_text.find("</think>", reasoning_open + 1)

                result = {
                    "row_index": example["row_index"],
                    "input_history": example["input_history"],
                    "history_sids": example["history_sids"],
                    "target_sid": example["target_sid"],
                    "output": example["target_sid"],
                    "generated_output": generated_sid,
                    "predict": generated_sids,
                    "completion_text": first_completion,
                    "has_sid_pattern": bool(SID_RE.fullmatch(generated_sid)),
                    "is_valid_catalog_sid": generated_sid in valid_sids,
                    "is_exact_match": generated_sid == example["target_sid"],
                    "has_reasoning_open": reasoning_open >= 0,
                    "has_reasoning_close": reasoning_close >= 0,
                    "reasoning_enclosed": reasoning_open >= 0 and reasoning_close > reasoning_open,
                    "reward_parseable": extract_solution(first_completion) is not None,
                }
                if save_full_text:
                    result["full_generated_text"] = full_generated_text
                results.append(result)
        except Exception as exc:
            print(f"[batch {start // batch_size + 1}] Error: {str(exc)[:160]}")
            for example in batch:
                results.append(
                    {
                        "row_index": example["row_index"],
                        "input_history": example["input_history"],
                        "history_sids": example["history_sids"],
                        "target_sid": example["target_sid"],
                        "output": example["target_sid"],
                        "predict": [],
                        "error": str(exc),
                    }
                )

    metrics = compute_metrics(results, valid_sids=valid_sids)

    os.makedirs(os.path.dirname(result_json_data) or ".", exist_ok=True)
    with open(result_json_data, "w") as f:
        json.dump(results, f, indent=2)

    metrics_json_data = metrics_json_data or derive_metrics_path(result_json_data)
    os.makedirs(os.path.dirname(metrics_json_data) or ".", exist_ok=True)
    with open(metrics_json_data, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to: {result_json_data}")
    print(f"Metrics saved to: {metrics_json_data}")
    for key in ("sid_pattern@1", "catalog_valid@1", "exact_match@1", "HR@10", "NDCG@10"):
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    fire.Fire(main)
