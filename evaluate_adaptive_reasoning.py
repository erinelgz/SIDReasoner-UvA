#!/usr/bin/env python
"""Unified direct, generated-reasoning, and external-reasoning SID evaluation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_reasoning_utils import (
    confidence_from_scores,
    deranged_source_rows,
    extract_reasoning_text,
    reciprocal_rank_fusion,
    truncate_reasoning,
)
from sid_eval_utils import (
    ANSWER_SEPARATOR,
    SID_RE,
    compute_metrics,
    derive_metrics_path,
    extract_generated_sid,
    extract_history_sids,
    load_valid_sids,
    load_verl_examples,
    set_seed,
)
from verl.utils.sid_constraints import make_sid_prefix_allowed_tokens_fn


DEFAULT_VALIDATION = Path("data/Amazon/rec_reasoning_verl/Office_Products/test.parquet")
DEFAULT_TEST = Path("data/Amazon/test/Office_Products_5_2016-10-2018-11.csv")
DEFAULT_INFO = Path("data/Amazon/info/Office_Products_5_2016-10-2018-11.txt")
DEFAULT_INDEX = Path("data/Amazon/index/Office_Products.index.json")
SYSTEM_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n"
    "Can you recommend the next item for the user based on their interaction history?\n"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_csv_examples(path: Path, tokenizer, max_examples: int, seed: int) -> list[dict]:
    frame = pd.read_csv(path)
    if max_examples > 0:
        frame = frame.sample(n=min(max_examples, len(frame)), random_state=seed)

    examples = []
    for row_index, row in frame.iterrows():
        history_sids = list(ast.literal_eval(row["history_item_sid"]))
        user_prompt = (
            f"The user has sequentially interacted with items {', '.join(history_sids)}. "
            "Can you recommend the next item for him? Let's think step by step before making recommendation. "
            "Directly output the item SID after thinking."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        examples.append(
            {
                "row_index": int(row_index),
                "prompt": messages,
                "input_text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "input_history": ", ".join(history_sids),
                "history_sids": history_sids,
                "target_sid": str(row["item_sid"]),
            }
        )
    return examples


def load_examples(path: Path, tokenizer, max_examples: int, seed: int) -> list[dict]:
    if path.suffix == ".parquet":
        return load_verl_examples(
            str(path),
            tokenizer=tokenizer,
            num_samples=max_examples,
            seed=seed,
            direct_sid_decoding=False,
        )
    if path.suffix == ".csv":
        return _load_csv_examples(path, tokenizer, max_examples, seed)
    raise ValueError(f"Unsupported evaluation data format: {path}")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reasoning_record(text: str, token_count: int, has_open: bool, has_close: bool, source_row: int) -> dict:
    return {
        "text": text,
        "token_count": int(token_count),
        "generated_open_tag": bool(has_open),
        "generated_close_tag": bool(has_close),
        "source_row_index": int(source_row),
    }


def generate_reasonings(model, tokenizer, examples: list[dict], args) -> tuple[list[list[dict]], list[float]]:
    all_reasonings: list[list[dict]] = [[] for _ in examples]
    per_example_latency = [0.0 for _ in examples]
    close_token_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_ids = [tokenizer.eos_token_id]
    if close_token_id is not None and close_token_id >= 0 and close_token_id != tokenizer.unk_token_id:
        eos_ids.append(close_token_id)

    do_sample = args.reasoning_samples > 1 or args.temperature > 0
    for start in tqdm(
        range(0, len(examples), args.reasoning_batch_size),
        desc="Generating reasoning",
    ):
        batch = examples[start : start + args.reasoning_batch_size]
        prompts = [example["input_text"] for example in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        prompt_width = input_ids.shape[-1]

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_reasoning_tokens,
            "num_return_sequences": args.reasoning_samples,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": eos_ids,
        }
        if do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        synchronize()
        elapsed = time.perf_counter() - started

        completions = generated[:, prompt_width:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=False)
        for batch_index, example in enumerate(batch):
            per_example_latency[start + batch_index] = elapsed / len(batch)
            sample_start = batch_index * args.reasoning_samples
            for sample_offset in range(args.reasoning_samples):
                completion_ids = completions[sample_start + sample_offset].tolist()
                while completion_ids and completion_ids[-1] in {tokenizer.pad_token_id, tokenizer.eos_token_id}:
                    completion_ids.pop()
                completion = texts[sample_start + sample_offset]
                reasoning, has_open, has_close = extract_reasoning_text(completion)
                reasoning_token_count = len(tokenizer.encode(reasoning, add_special_tokens=False))
                all_reasonings[start + batch_index].append(
                    _reasoning_record(
                        reasoning,
                        reasoning_token_count,
                        has_open,
                        has_close,
                        example["row_index"],
                    )
                )
    return all_reasonings, per_example_latency


def _external_reasoning_map(path: Path) -> dict[int, list[dict]]:
    records = json.loads(path.read_text())
    mapping: dict[int, list[dict]] = {}
    for record in records:
        reasonings = record.get("reasonings")
        if reasonings is None and "reasoning" in record:
            reasonings = [{"text": record["reasoning"], "token_count": record.get("reasoning_token_count", 0)}]
        if not reasonings:
            raise ValueError(f"External result row {record.get('row_index')} contains no reasoning")
        mapping[int(record["row_index"])] = reasonings
    return mapping


def load_external_reasonings(tokenizer, examples: list[dict], args) -> tuple[list[list[dict]], list[float]]:
    if not args.external_reasoning_file:
        raise ValueError("--external-reasoning-file is required for external mode")
    source_map = _external_reasoning_map(Path(args.external_reasoning_file))
    row_ids = [example["row_index"] for example in examples]
    source_rows = {row_id: row_id for row_id in row_ids}
    if args.external_transform == "shuffle":
        source_rows = deranged_source_rows(row_ids, args.seed)

    all_reasonings = []
    for example in examples:
        source_row = source_rows[example["row_index"]]
        if source_row not in source_map:
            raise ValueError(f"External reasoning file has no row {source_row}")
        source_reasonings = source_map[source_row]
        if len(source_reasonings) < args.reasoning_samples:
            raise ValueError(
                f"Row {source_row} has {len(source_reasonings)} reasoning samples, "
                f"but {args.reasoning_samples} were requested"
            )
        selected = []
        for source in source_reasonings[: args.reasoning_samples]:
            text = str(source.get("text", ""))
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if args.external_transform == "truncate":
                text, token_count = truncate_reasoning(text, tokenizer, args.truncate_fraction)
            selected.append(
                _reasoning_record(
                    text,
                    token_count,
                    bool(source.get("generated_open_tag", True)),
                    bool(source.get("generated_close_tag", True)),
                    source_row,
                )
            )
        all_reasonings.append(selected)
    return all_reasonings, [0.0 for _ in examples]


def direct_reasonings(examples: list[dict]) -> tuple[list[list[dict]], list[float]]:
    return [
        [_reasoning_record("", 0, False, False, example["row_index"])] for example in examples
    ], [0.0 for _ in examples]


def build_reasoning_prompt(base_prompt: str, reasoning: str) -> str:
    content = reasoning.strip()
    if content:
        return f"{base_prompt}<think>\n{content}\n{ANSWER_SEPARATOR}"
    return f"{base_prompt}<think>\n{ANSWER_SEPARATOR}"


def rank_sids(model, tokenizer, examples: list[dict], all_reasonings: list[list[dict]], valid_sids: set[str], args):
    prefix_allowed_tokens_fn = make_sid_prefix_allowed_tokens_fn(
        tokenizer,
        index_file=args.index_file,
        info_file=args.info_file,
        eos_token_id=tokenizer.eos_token_id,
        answer_separator=ANSWER_SEPARATOR,
        fail_on_missing_separator=True,
    )
    flat_inputs = []
    for example_index, (example, reasonings) in enumerate(zip(examples, all_reasonings)):
        for reasoning_index, reasoning in enumerate(reasonings):
            flat_inputs.append(
                {
                    "example_index": example_index,
                    "reasoning_index": reasoning_index,
                    "text": build_reasoning_prompt(example["input_text"], reasoning["text"]),
                }
            )

    sample_rankings: list[list[dict]] = [[] for _ in examples]
    per_example_sid_latency = [0.0 for _ in examples]
    for start in tqdm(range(0, len(flat_inputs), args.batch_size), desc="Ranking SIDs"):
        batch = flat_inputs[start : start + args.batch_size]
        encoded = tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        prompt_width = input_ids.shape[-1]

        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_sid_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                early_stopping=True,
                length_penalty=args.length_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=True,
            )
        synchronize()
        elapsed = time.perf_counter() - started

        completion_ids = generated.sequences[:, prompt_width:]
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        sequence_scores = generated.sequences_scores.detach().float().cpu().tolist()
        for input_index, item in enumerate(batch):
            seq_start = input_index * args.num_beams
            seq_end = seq_start + args.num_beams
            texts = completion_texts[seq_start:seq_end]
            scores = sequence_scores[seq_start:seq_end]
            predictions = []
            prediction_scores = []
            for text, score in zip(texts, scores):
                sid = extract_generated_sid(text)
                if sid not in predictions:
                    predictions.append(sid)
                    prediction_scores.append(float(score))
            confidence = confidence_from_scores(prediction_scores)
            ranking = {
                "reasoning_index": item["reasoning_index"],
                "predict": predictions,
                "completion_texts": [text.strip() for text in texts],
                "confidence": confidence,
            }
            if args.save_sequence_scores:
                ranking["sequence_scores"] = prediction_scores
            sample_rankings[item["example_index"]].append(ranking)
            per_example_sid_latency[item["example_index"]] += elapsed / len(batch)

    results = []
    for example, reasonings, rankings, sid_latency in zip(
        examples,
        all_reasonings,
        sample_rankings,
        per_example_sid_latency,
    ):
        if len(rankings) == 1:
            predictions = rankings[0]["predict"][: args.num_beams]
            fusion_scores = rankings[0].get("sequence_scores", [])
            confidence = rankings[0]["confidence"]
        else:
            predictions, fusion_scores = reciprocal_rank_fusion(
                [ranking["predict"] for ranking in rankings],
                limit=args.num_beams,
                rank_constant=args.rrf_k,
            )
            confidence = None
        generated_sid = predictions[0] if predictions else ""
        result = {
            "row_index": example["row_index"],
            "split": args.split,
            "reasoning_mode": args.reasoning_mode,
            "input_history": example["input_history"],
            "history_sids": example["history_sids"],
            "target_sid": example["target_sid"],
            "output": example["target_sid"],
            "generated_output": generated_sid,
            "predict": predictions,
            "reasonings": reasonings,
            "sample_rankings": rankings,
            "fusion_scores": fusion_scores,
            "confidence": confidence,
            "reasoning_token_count": sum(reasoning["token_count"] for reasoning in reasonings),
            "sid_latency_seconds": sid_latency,
            "has_sid_pattern": bool(SID_RE.fullmatch(generated_sid)),
            "is_valid_catalog_sid": generated_sid in valid_sids,
            "is_exact_match": generated_sid == example["target_sid"],
            "generated_reasoning_open": all(reasoning["generated_open_tag"] for reasoning in reasonings),
            "generated_reasoning_close": all(reasoning["generated_close_tag"] for reasoning in reasonings),
            "repeat_target": example["target_sid"] in set(example["history_sids"]),
        }
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--reasoning-mode", choices=("direct", "generated", "external"), required=True)
    parser.add_argument("--reasoning-samples", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--save-sequence-scores", action="store_true")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--external-reasoning-file")
    parser.add_argument("--external-transform", choices=("none", "shuffle", "truncate"), default="none")
    parser.add_argument("--truncate-fraction", type=float, default=1.0)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--index-file", default=str(DEFAULT_INDEX))
    parser.add_argument("--info-file", default=str(DEFAULT_INFO))
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reasoning-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-reasoning-tokens", type=int, default=1024)
    parser.add_argument("--max-sid-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to evaluate the official test split without --allow-test")
    if args.reasoning_mode != "external" and args.external_transform != "none":
        raise SystemExit("--external-transform is only valid with --reasoning-mode external")
    if args.reasoning_mode == "direct" and args.reasoning_samples != 1:
        raise SystemExit("Direct mode supports exactly one empty reasoning trajectory")
    if args.reasoning_samples > 1 and args.reasoning_mode == "generated" and args.temperature <= 0:
        raise SystemExit("Multiple generated reasoning samples require --temperature > 0")

    data_path = args.data_path or (DEFAULT_VALIDATION if args.split == "validation" else DEFAULT_TEST)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wall_started = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, device_map="auto")
    model.eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.config.pad_token_id = tokenizer.eos_token_id

    examples = load_examples(data_path, tokenizer, args.max_examples, args.seed)
    valid_sids = load_valid_sids(index_file=args.index_file, info_file=args.info_file)

    if args.reasoning_mode == "generated":
        all_reasonings, reasoning_latencies = generate_reasonings(model, tokenizer, examples, args)
    elif args.reasoning_mode == "external":
        all_reasonings, reasoning_latencies = load_external_reasonings(tokenizer, examples, args)
    else:
        all_reasonings, reasoning_latencies = direct_reasonings(examples)

    results = rank_sids(model, tokenizer, examples, all_reasonings, valid_sids, args)
    for result, reasoning_latency in zip(results, reasoning_latencies):
        result["reasoning_latency_seconds"] = reasoning_latency
        result["total_latency_seconds"] = reasoning_latency + result["sid_latency_seconds"]

    metrics = compute_metrics(results, valid_sids=valid_sids)
    evaluated = [result for result in results if not result.get("error")]
    wall_seconds = time.perf_counter() - wall_started
    metrics.update(
        {
            "split": args.split,
            "data_path": str(data_path),
            "data_sha256": file_sha256(data_path),
            "base_model": args.base_model,
            "reasoning_mode": args.reasoning_mode,
            "reasoning_samples": args.reasoning_samples,
            "external_transform": args.external_transform,
            "reasoning_tokens_total": sum(result["reasoning_token_count"] for result in evaluated),
            "reasoning_tokens_mean": (
                sum(result["reasoning_token_count"] for result in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "generated_reasoning_open_rate": (
                sum(result["generated_reasoning_open"] for result in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "generated_reasoning_close_rate": (
                sum(result["generated_reasoning_close"] for result in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "reasoning_latency_seconds": sum(result["reasoning_latency_seconds"] for result in evaluated),
            "sid_latency_seconds": sum(result["sid_latency_seconds"] for result in evaluated),
            "wall_seconds": wall_seconds,
        }
    )

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(results, indent=2) + "\n")
    metrics_path = args.metrics_json or Path(derive_metrics_path(str(args.result_json)))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Results: {args.result_json}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
