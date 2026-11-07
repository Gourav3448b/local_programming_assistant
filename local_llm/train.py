"""Train MiniGPT from random initialization; no pretrained weights or API calls."""

from __future__ import annotations

import argparse
import json
import random
import time
import os
from pathlib import Path
import torch

from .data import build_training_stream, load_examples, get_batch
from .model import MiniGPT, count_parameters
from .tokenizer import ByteTokenizer, EOS_ID, BOS_ID, VOCAB_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "programming_qa.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "minigpt.pt"


def evaluate_loss(model: MiniGPT, data: torch.Tensor, block_size: int, batch_size: int,
                  device: torch.device, batches: int = 8) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            if loss is not None:
                losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.num_threads))
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda")

    examples = load_examples(args.dataset)
    tokenizer = ByteTokenizer()
    if len(examples) < 3:
        raise ValueError("Add at least three examples so a validation example can be held out.")

    # Split by complete Q&A records so validation examples never appear in training windows.
    split_at = max(1, min(len(examples) - 1, int(len(examples) * 0.92)))
    train_data = build_training_stream(examples[:split_at], tokenizer)
    val_data = build_training_stream(examples[split_at:], tokenizer)
    if train_data.numel() < args.block_size + 2 or val_data.numel() <= args.block_size + 1:
        raise ValueError("Dataset is too small for this block_size. Add examples or reduce --block-size.")

    model_config = {
        "vocab_size": VOCAB_SIZE,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    model = MiniGPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    print("=" * 64)
    print("MiniGPT training from scratch")
    print(f"Device: {device}")
    print(f"Examples: {len(examples)} ({split_at} train / {len(examples) - split_at} validation) | "
          f"Training bytes/tokens: {len(train_data):,}")
    print(f"Vocabulary: {VOCAB_SIZE} byte/special tokens")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Steps: {args.steps} | Context: {args.block_size}")
    print("No API calls and no pretrained checkpoint are used.")
    print("=" * 64)

    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Training loss was not calculated")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val_loss = evaluate_loss(model, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - started
            print(f"step {step:>5}/{args.steps} | train loss {loss.item():.4f} | "
                  f"validation estimate {val_loss:.4f} | {elapsed:.1f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_config": model_config,
        "model_state_dict": model.cpu().state_dict(),
        "training_info": {
            "examples": len(examples),
            "steps": args.steps,
            "training_tokens": int(train_data.numel()),
            "final_train_loss": float(loss.item()),
            "device": str(device),
            "trained_from_scratch": True,
            "created_unix": int(time.time()),
        },
    }
    torch.save(checkpoint, output_path)
    print(f"\nSaved checkpoint: {output_path}")
    print("Training complete. Start the Flask app to chat with the local assistant.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local MiniGPT programming assistant.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "block_size", "d_model", "n_heads", "n_layers", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")
    if args.d_model % args.n_heads != 0:
        parser.error("--d-model must be divisible by --n-heads")
    return args


if __name__ == "__main__":
    train(parse_args())
