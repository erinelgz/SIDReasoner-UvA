import pandas as pd
import fire
import torch
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from data_Qwen3 import Reasoning_Eval_Dataset
import random
from vllm import LLM, SamplingParams
import logging
from verl.utils.sid_constraints import make_sid_prefix_allowed_tokens_fn


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np


def get_hash(x):
    x = [str(_) for _ in x]
    return "-".join(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "/home/scur1249/Office_Products_checkpoint/merged",
    info_file: str = "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt",
    category: str = "Office_Products",
    test_data_path: str = "./data/Amazon/test/Office_Products_5_2016-10-2018-11_for_test.csv",
    item_file: str = "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt",
    index_file: str = "./data/Amazon/index/Office_Products.index.json",
    result_json_data: str = "./temp/test_results_Qwen3.json",
    batch_size: int = 1,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 10,
    padding_side: str = "left",
):
    os.environ["FLASH_ATTENTION"] = "1"
    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    set_seed(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "Video_Games": "video games",
    }
    if category in category_dict:
        category = category_dict[category]
    else:
        category = "items"
    print(category)

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = True
    # model.config.attention_mode = "flash"
    model.eval()
    llm = LLM(
        model=base_model,
        max_model_len=2048,
        max_num_seqs=32,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prefix_allowed_tokens_fn = make_sid_prefix_allowed_tokens_fn(
        tokenizer,
        index_file=index_file,
        info_file=info_file,
        eos_token_id=tokenizer.eos_token_id,
        answer_separator="</think>\n\n",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side

    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = Reasoning_Eval_Dataset(
        data_file=test_data_path,
        item_file=item_file,
        index_file=index_file,
        sample=-1,
        tokenizer=tokenizer,
        max_len=2048,
        category=category,
        test=True,
        seed=seed,
    )

    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    # Debugging only
    # encodings = encodings[:500]
    # test_data = test_data[:500]

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # * 为了可复现性，不做采样
        skip_special_tokens=False,
    )

    def evaluate(
        encodings,
        num_beams=10,
        max_new_tokens=256,
        length_penalty=1.0,
        padding_side="right",
        **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        # print(f"maxLen of input_ids: {maxLen}")

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for _ in encodings:
            L = len(_["input_ids"])
            if padding_side == "left":
                padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            elif padding_side == "right":
                padding_encodings["input_ids"].append(_["input_ids"] + [tokenizer.pad_token_id] * (maxLen - L))
            else:
                raise ValueError("Invalid padding_side. Choose 'left' or 'right'.")

            attention_mask.append([0] * (maxLen - L) + [1] * L)
        # print(f"maxLen of input_ids: {maxLen}")

        with torch.no_grad():
            generate_kwargs = {
                "input_ids": torch.tensor(padding_encodings["input_ids"]).to(device),
                "attention_mask": torch.tensor(attention_mask).to(device),
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "num_return_sequences": num_beams,
                "output_scores": True,
                "return_dict_in_generate": True,
                "early_stopping": True,
                "length_penalty": length_penalty,
                # "temperature": args.temperature,
                "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            }

            generation_output = model.generate(**generate_kwargs)

        batched_completions = generation_output.sequences[:, maxLen:]

        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(
                batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)

        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs

    model = model.to(device)

    from tqdm import tqdm

    outputs = []
    # cots = []

    prompts = tokenizer.batch_decode([_["input_ids"] for _ in encodings], skip_special_tokens=False)
    with torch.no_grad():
        outputs = llm.generate(
            prompts,
            sampling_params,
        )
    cots = [cot.outputs[0].text.strip() for cot in outputs]
    cots = [cot[: cot.find("</think>")].strip() for cot in cots]
    prompt_cot = [prompt + cot.outputs[0].text.strip() for prompt, cot in zip(prompts, outputs)]
    prompt_cot = [text[: text.find("</think>")].strip() + "\n</think>\n\n" for text in prompt_cot]
    # reasoning_output = tokenizer(prompt_cot, return_tensors="pt", padding=True, truncation=True, max_length=2560)
    for i in range(len(encodings)):
        # * Update encodings with new input_ids including CoT
        reasoning_output = tokenizer(prompt_cot[i], return_tensors="pt", padding=True, truncation=True, max_length=2560)
        encodings[i]["input_ids"] = reasoning_output["input_ids"][0].tolist()
        encodings[i]["attention_mask"] = reasoning_output["attention_mask"][0].tolist()
    del llm

    # encodings = sorted(encodings, key=lambda x: len(x['input_ids']))    # Sort by length for efficient batching
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size : (i + 1) * batch_size])

    outputs = []
    for idx, encodings in enumerate(tqdm(new_encodings, desc="Generating reasoning trajectories and items...")):
        # Use standard evaluation
        # print(f"Processing batch {idx+1}/{len(new_encodings)}...")
        output = evaluate(
            encodings,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            padding_side=padding_side,
        )

        outputs = outputs + output

    for i, test in enumerate(test_data):
        # print(outputs[i])
        test["prompt_cot"] = prompt_cot[i]
        test["predict"] = outputs[i]
        test["cot"] = cots[i]

    for i in range(len(test_data)):
        if "dedup" in test_data[i]:
            test_data[i].pop("dedup")
    with open(result_json_data, "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
