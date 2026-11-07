"""Combines local GPT generation with a transparent local knowledge retriever."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .retrieval import LocalRetriever
from .tokenizer import ByteTokenizer, BOS_ID, EOS_ID

if TYPE_CHECKING:
    from .model import MiniGPT


class ProgrammingAssistant:
    def __init__(
        self,
        dataset_path: str | Path,
        checkpoint_path: str | Path,
        device: str | None = None,
        enable_model: bool = True,
    ) -> None:
        import json
        from .data import load_examples

        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.examples = load_examples(self.dataset_path)
        self.retriever = LocalRetriever(self.examples)
        self.tokenizer = ByteTokenizer()
        self.enable_model = enable_model
        self.device = device or "cpu"
        if enable_model:
            import torch

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MiniGPT | None = None
        self.model_loaded = False
        self.checkpoint_info: dict[str, object] = {}
        if self.enable_model and self.checkpoint_path.exists():
            self.load_checkpoint()

    def load_checkpoint(self) -> bool:
        if not self.enable_model:
            return False
        try:
            import torch

            from .model import MiniGPT

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            config = checkpoint.get("model_config", {})
            self.model = MiniGPT(**config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.checkpoint_info = checkpoint.get("training_info", {})
            self.model_loaded = True
            return True
        except Exception as exc:  # Keep the web app available if a checkpoint is damaged.
            self.model = None
            self.model_loaded = False
            self.checkpoint_info = {"load_error": str(exc)}
            return False

    def _generate(self, question: str) -> str:
        if self.model is None:
            return ""
        import torch

        prompt = f"### User:\n{question.strip()}\n### Assistant:\n"
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_tensor,
            max_new_tokens=260,
            temperature=0.75,
            top_k=35,
            stop_token_id=EOS_ID,
        )[0].tolist()
        generated_ids = output[len(ids):]
        text = self.tokenizer.decode(generated_ids).strip()
        text = text.split("### User:")[0].split("### Assistant:")[0].strip()
        return re.sub(r"\n{3,}", "\n\n", text)

    def answer(self, question: str) -> dict[str, object]:
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")
        matches = self.retriever.search(question, limit=3)
        best = matches[0] if matches else None
        score = float(best["score"]) if best else 0.0
        generated = self._generate(question) if self.model_loaded else ""

        # Prefer a directly relevant, human-written reference when the match is strong.
        # Keep the raw local model output visible so the training experiment is inspectable.
        if best and score >= 0.18:
            answer_text = str(best["answer"])
            source = "Local programming knowledge base"
            matched_question = str(best["question"])
        elif best and score >= 0.07:
            answer_text = (
                "I could not find a strong match, so treat this as a starting point. The closest "
                "reference in the local programming dataset is:\n\n" + str(best["answer"])
            )
            source = "Closest local reference"
            matched_question = str(best["question"])
        else:
            answer_text = "This question is not well represented in the included dataset. "
            if self.model_loaded:
                answer_text += (
                    "The raw MiniGPT draft is shown below for experimentation, but this tiny "
                    "from-scratch model can produce incoherent or incorrect text. "
                )
            answer_text += "Add a correct question-answer example to dataset/programming_qa.jsonl and train again for better coverage."
            source = "Experimental MiniGPT output — verify carefully"
            matched_question = None

        return {
            "answer": answer_text,
            "source": source,
            "matched_question": matched_question,
            "similarity": round(score, 3),
            "model_output": generated or None,
            "model_loaded": self.model_loaded,
        }
