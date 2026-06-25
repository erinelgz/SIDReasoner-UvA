from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


SID_PATTERN = re.compile(r"<[^>]+>")


def extract_sid_tokens(text: str) -> list[str]:
    return SID_PATTERN.findall(text or "")


def _dedupe_preserve_order(sequences: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    seen = set()
    unique = []
    for sequence in sequences:
        if sequence in seen:
            continue
        seen.add(sequence)
        unique.append(sequence)
    return unique


def _coerce_sid_sequence(value) -> tuple[str, ...] | None:
    if isinstance(value, str):
        parts = extract_sid_tokens(value)
    elif isinstance(value, (list, tuple)):
        parts = [part for part in value if isinstance(part, str)]
    else:
        return None

    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return None
    return tuple(parts)


def load_sid_sequences(
    *,
    index_file: str | None = None,
    info_file: str | None = None,
    expected_depth: int | None = 3,
) -> list[tuple[str, ...]]:
    sequences = []

    if index_file:
        with Path(index_file).open("r") as f:
            index_data = json.load(f)

        if isinstance(index_data, dict):
            raw_values = index_data.values()
        elif isinstance(index_data, list):
            raw_values = index_data
        else:
            raise ValueError(f"Unsupported SID index format in {index_file}: {type(index_data)!r}")

        for raw_value in raw_values:
            sequence = _coerce_sid_sequence(raw_value)
            if sequence is not None:
                sequences.append(sequence)

    if info_file:
        with Path(info_file).open("r") as f:
            for line in f:
                semantic_id = line.split("\t", 1)[0].strip()
                sequence = _coerce_sid_sequence(semantic_id)
                if sequence is not None:
                    sequences.append(sequence)

    if expected_depth is not None:
        sequences = [sequence for sequence in sequences if len(sequence) == expected_depth]

    sequences = _dedupe_preserve_order(sequences)
    if not sequences:
        source = index_file or info_file or "<none>"
        raise ValueError(f"No valid SID sequences loaded from {source}")
    return sequences


class TokenTrie:
    def __init__(self):
        self.children: dict[int, TokenTrie] = {}
        self.terminal = False

    def insert(self, token_ids: Iterable[int]) -> None:
        node = self
        inserted = False
        for token_id in token_ids:
            node = node.children.setdefault(int(token_id), TokenTrie())
            inserted = True
        if inserted:
            node.terminal = True

    def _node_for_prefix(self, token_ids: Iterable[int]) -> "TokenTrie | None":
        node = self
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node

    def allowed_next_token_ids(self, token_ids: Iterable[int], eos_token_id: int | None = None) -> list[int]:
        node = self._node_for_prefix(token_ids)
        if node is None:
            return [eos_token_id] if eos_token_id is not None else []

        allowed = list(node.children.keys())
        if node.terminal and eos_token_id is not None:
            allowed.append(int(eos_token_id))
        return allowed

    def contains(self, token_ids: Iterable[int]) -> bool:
        node = self._node_for_prefix(token_ids)
        return bool(node and node.terminal)


def _encode_text(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_token_trie(
    tokenizer,
    sid_sequences: Iterable[tuple[str, ...]],
) -> TokenTrie:
    trie = TokenTrie()
    for sid_sequence in sid_sequences:
        sid_text = "".join(sid_sequence)
        token_ids = _encode_text(tokenizer, sid_text)
        trie.insert(token_ids)
    return trie


def build_token_trie_from_sid_files(
    tokenizer,
    *,
    index_file: str | None = None,
    info_file: str | None = None,
    expected_depth: int | None = 3,
) -> TokenTrie:
    sid_sequences = load_sid_sequences(index_file=index_file, info_file=info_file, expected_depth=expected_depth)
    return build_token_trie(tokenizer, sid_sequences)


def _find_last_sublist(sequence: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    needle_len = len(needle)
    for start in range(len(sequence) - needle_len, -1, -1):
        if sequence[start : start + needle_len] == needle:
            return start
    return None


def _find_last_separator(sequence: list[int], separators: list[list[int]]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for separator in separators:
        position = _find_last_sublist(sequence, separator)
        if position is None:
            continue
        if best is None or position > best[0]:
            best = (position, len(separator))
    return best


def encode_separator_variants(tokenizer, answer_separator: str = "</think>\n\n") -> list[list[int]]:
    candidates = [answer_separator]
    if not answer_separator.startswith("\n"):
        candidates.append("\n" + answer_separator)

    encoded = []
    for candidate in candidates:
        token_ids = _encode_text(tokenizer, candidate)
        if token_ids and token_ids not in encoded:
            encoded.append(token_ids)
    return encoded


def make_sid_prefix_allowed_tokens_fn(
    tokenizer,
    *,
    index_file: str | None = None,
    info_file: str | None = None,
    trie: TokenTrie | None = None,
    eos_token_id: int | None = None,
    answer_separator: str = "</think>\n\n",
    fail_on_missing_separator: bool = True,
):
    if trie is None:
        trie = build_token_trie_from_sid_files(tokenizer, index_file=index_file, info_file=info_file)

    separators = encode_separator_variants(tokenizer, answer_separator)
    eos_token_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    vocab_size = getattr(tokenizer, "vocab_size", None)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        if hasattr(input_ids, "tolist"):
            input_ids_list = input_ids.tolist()
        else:
            input_ids_list = list(input_ids)

        separator_match = _find_last_separator(input_ids_list, separators)
        if separator_match is None:
            if fail_on_missing_separator:
                decoded = tokenizer.decode(input_ids_list[-128:], skip_special_tokens=False)
                raise ValueError(f"SID answer separator not found in generated prefix: {decoded!r}")
            if vocab_size is None:
                return []
            return list(range(vocab_size))

        position, separator_len = separator_match
        generated_after_separator = input_ids_list[position + separator_len :]
        return trie.allowed_next_token_ids(generated_after_separator, eos_token_id=eos_token_id)

    return prefix_allowed_tokens_fn


class SIDConstrainedLogitsProcessor:
    def __init__(
        self,
        *,
        trie: TokenTrie,
        answer_separator_ids: list[list[int]],
        eos_token_id: int | None,
    ):
        self.trie = trie
        self.answer_separator_ids = answer_separator_ids
        self.eos_token_id = eos_token_id

    def __call__(self, *args):
        if len(args) == 2:
            token_ids, scores = args
            full_token_ids = _to_token_list(token_ids)
        elif len(args) == 3:
            prompt_token_ids, token_ids, scores = args
            full_token_ids = _to_token_list(prompt_token_ids) + _to_token_list(token_ids)
        else:
            raise TypeError(f"Unexpected logits processor signature with {len(args)} arguments")

        separator_match = _find_last_separator(full_token_ids, self.answer_separator_ids)
        if separator_match is None:
            return scores

        position, separator_len = separator_match
        generated_after_separator = full_token_ids[position + separator_len :]
        allowed = self.trie.allowed_next_token_ids(generated_after_separator, eos_token_id=self.eos_token_id)
        allowed = [token_id for token_id in allowed if 0 <= token_id < scores.shape[-1]]
        if not allowed:
            return scores

        masked_scores = scores.new_full(scores.shape, float("-inf"))
        masked_scores[allowed] = scores[allowed]
        return masked_scores


def _to_token_list(token_ids) -> list[int]:
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def build_sid_logits_processor(
    tokenizer,
    *,
    index_file: str | None = None,
    info_file: str | None = None,
    eos_token_id: int | None = None,
    answer_separator: str = "</think>\n\n",
) -> SIDConstrainedLogitsProcessor:
    trie = build_token_trie_from_sid_files(tokenizer, index_file=index_file, info_file=info_file)
    return SIDConstrainedLogitsProcessor(
        trie=trie,
        answer_separator_ids=encode_separator_variants(tokenizer, answer_separator),
        eos_token_id=tokenizer.eos_token_id if eos_token_id is None else eos_token_id,
    )
