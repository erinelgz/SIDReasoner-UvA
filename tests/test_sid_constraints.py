import unittest

import torch

from verl.utils.sid_constraints import SIDConstrainedLogitsProcessor, TokenTrie, make_sid_prefix_allowed_tokens_fn


class FakeTokenizer:
    eos_token_id = 0
    vocab_size = 32

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % self.vocab_size for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class SIDConstraintTests(unittest.TestCase):
    def test_token_trie_transitions_and_terminal(self):
        trie = TokenTrie()
        trie.insert([4, 5])
        trie.insert([4, 6])
        self.assertEqual(set(trie.allowed_next_token_ids([])), {4})
        self.assertEqual(set(trie.allowed_next_token_ids([4])), {5, 6})
        self.assertEqual(trie.allowed_next_token_ids([4, 5], eos_token_id=0), [0])
        self.assertTrue(trie.contains([4, 6]))
        self.assertFalse(trie.contains([4]))

    def test_prefix_function_rejects_missing_separator(self):
        tokenizer = FakeTokenizer()
        trie = TokenTrie()
        trie.insert([7])
        callback = make_sid_prefix_allowed_tokens_fn(
            tokenizer,
            trie=trie,
            answer_separator="END",
            fail_on_missing_separator=True,
        )
        with self.assertRaisesRegex(ValueError, "separator not found"):
            callback(0, torch.tensor([1, 2, 3]))

    def test_prefix_function_uses_tokens_after_separator(self):
        tokenizer = FakeTokenizer()
        trie = TokenTrie()
        trie.insert([7, 8])
        callback = make_sid_prefix_allowed_tokens_fn(tokenizer, trie=trie, answer_separator="END")
        separator = tokenizer.encode("END")
        self.assertEqual(callback(0, torch.tensor([9, *separator])), [7])
        self.assertEqual(callback(0, torch.tensor([9, *separator, 7])), [8])

    def test_logits_processor_masks_disallowed_tokens(self):
        trie = TokenTrie()
        trie.insert([3])
        processor = SIDConstrainedLogitsProcessor(trie=trie, answer_separator_ids=[[8, 9]], eos_token_id=0)
        scores = torch.arange(6, dtype=torch.float32)
        masked = processor([8, 9], scores)
        self.assertTrue(torch.isfinite(masked[3]))
        self.assertTrue(torch.isneginf(masked[2]))


if __name__ == "__main__":
    unittest.main()
