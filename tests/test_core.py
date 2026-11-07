from __future__ import annotations

import unittest
from pathlib import Path
import torch

from local_llm.data import load_examples
from local_llm.model import MiniGPT, count_parameters
from local_llm.retrieval import LocalRetriever
from local_llm.assistant import ProgrammingAssistant
from local_llm.tokenizer import ByteTokenizer, BOS_ID, EOS_ID, VOCAB_SIZE

ROOT = Path(__file__).resolve().parents[1]


class TokenizerTests(unittest.TestCase):
    def test_round_trip_utf8(self):
        tokenizer = ByteTokenizer()
        sample = "Hello, Python! नमस्ते 🚀"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_special_token_ids(self):
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode("test", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(tokenizer.vocab_size, VOCAB_SIZE)


class DatasetAndRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = load_examples(ROOT / "dataset" / "programming_qa.jsonl")
        cls.retriever = LocalRetriever(cls.examples)

    def test_dataset_has_valid_examples(self):
        self.assertGreaterEqual(len(self.examples), 50)
        self.assertTrue(all(item["question"] and item["answer"] for item in self.examples))

    def test_retriever_finds_binary_search(self):
        result = self.retriever.search("Explain binary search", limit=1)
        self.assertTrue(result)
        self.assertIn("binary search", str(result[0]["question"]).lower())
        self.assertGreater(float(result[0]["score"]), 0.0)


class ModelTests(unittest.TestCase):
    def test_forward_and_loss(self):
        model = MiniGPT(vocab_size=VOCAB_SIZE, block_size=32, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
        x = torch.randint(0, VOCAB_SIZE, (2, 16))
        y = torch.randint(0, VOCAB_SIZE, (2, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, VOCAB_SIZE))
        self.assertIsNotNone(loss)
        self.assertGreater(count_parameters(model), 0)

    def test_generation_length(self):
        model = MiniGPT(vocab_size=VOCAB_SIZE, block_size=32, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
        prompt = torch.tensor([[BOS_ID, 72, 105]], dtype=torch.long)
        output = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=5)
        self.assertEqual(output.shape[0], 1)
        self.assertGreaterEqual(output.shape[1], prompt.shape[1] + 1)
        self.assertLessEqual(output.shape[1], prompt.shape[1] + 3)


class AssistantCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = ProgrammingAssistant(
            ROOT / "dataset" / "programming_qa.jsonl",
            ROOT / "artifacts" / "minigpt.pt",
            device="cpu",
        )

    def test_demo_checkpoint_loads(self):
        self.assertTrue(self.assistant.model_loaded)

    def test_known_question_uses_local_reference(self):
        result = self.assistant.answer("What is binary search?")
        self.assertEqual(result["source"], "Local programming knowledge base")
        self.assertIn("sorted data", str(result["answer"]))


if __name__ == "__main__":
    unittest.main()
