"""Dataset loading and next-token training batch creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .tokenizer import ByteTokenizer, BOS_ID, EOS_ID

if TYPE_CHECKING:
    import torch


PROMPT_TEMPLATE = "### User:\n{question}\n### Assistant:\n{answer}\n"


def load_examples(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError(f"Line {line_number} must contain non-empty question and answer fields")
            examples.append({"question": question, "answer": answer})
    if not examples:
        raise ValueError("Dataset is empty")
    return examples


def build_training_stream(examples: list[dict[str, str]], tokenizer: ByteTokenizer) -> torch.Tensor:
    import torch

    token_ids: list[int] = []
    for item in examples:
        text = f"### User:\n{item['question']}\n### Assistant:\n{item['answer']}\n"
        token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.tensor(token_ids, dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if data.numel() <= block_size + 1:
        raise ValueError("Training dataset is too short for the selected block_size")
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)
