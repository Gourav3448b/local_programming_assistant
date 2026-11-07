# Training Guide — Train MiniGPT From Scratch

## Goal

A demonstration checkpoint is included at `artifacts/minigpt.pt` (800 optimizer steps, initialized from random weights). The app can use it immediately. Running the trainer with its default output path replaces it with a new training run.

The goal is to understand and implement a small language model without downloading pretrained weights or sending text to an external API.

## What “from scratch” means here

- The Transformer parameters are initialized randomly.
- The tokenizer uses a fixed byte vocabulary rather than a pretrained tokenizer.
- The project trains the next-token prediction objective on `dataset/programming_qa.jsonl`.
- It does not load a Hugging Face checkpoint or call an external model endpoint.

The PyTorch library supplies tensor operations and neural-network building blocks; it does not supply the trained model weights.

## Training steps

1. Read each JSONL line as a question and answer.
2. Format it into a prompt/response text sequence.
3. Add BOS and EOS markers and convert UTF-8 bytes to token IDs.
4. Sample fixed-length windows from the concatenated training stream.
5. Feed input tokens through the causal Transformer.
6. Compare logits against the sequence shifted by one byte.
7. Calculate cross-entropy loss.
8. Backpropagate gradients and update parameters with AdamW.
9. Periodically estimate validation loss on a held-out tail of the stream.
10. Save model configuration, learned weights, and training metadata to `artifacts/minigpt.pt`.

## First training run

Run:

```bash
python -m local_llm.train --steps 1200
```

A short test run can confirm that the pipeline works:

```bash
python -m local_llm.train --steps 20 --d-model 64 --n-heads 4 --n-layers 2 --block-size 128 --batch-size 8 --output artifacts/minigpt_test.pt
```

The test checkpoint uses a different architecture configuration from the default and is saved under a different filename. To load it as the app checkpoint, either train with the default output path or rename/copy it to `artifacts/minigpt.pt` after deciding you are satisfied with the experiment.

## How to improve the model

### Improve the dataset first

More high-quality and varied data is usually more valuable than simply increasing the number of steps. Include:

- Beginner, intermediate, and advanced questions.
- Correct examples and expected outputs.
- Common bugs and explanations of their causes.
- Different phrasings of the same concept.
- Edge cases and conceptual comparisons.

Keep training and validation examples separate for a more rigorous experiment. This starter project uses a small tail split only as a rough validation-loss estimate.

### Tune model size and training

- Increase `--d-model` for wider representations.
- Increase `--n-layers` for more Transformer blocks.
- Increase `--block-size` for longer context, at higher memory and compute cost.
- Adjust `--learning-rate` if loss becomes unstable or training is too slow.
- Compare checkpoints rather than assuming the largest model is best.

A larger network is not automatically better when the dataset is tiny. It may memorize examples without learning robust patterns.

## Understanding the training log

- **train loss:** Cross-entropy on the sampled training batch. It should generally trend downward, though individual readings fluctuate.
- **validation estimate:** Loss on sampled windows from the reserved tail. It is a rough diagnostic, not a full benchmark.
- **steps:** Number of optimizer updates, not epochs.
- **parameters:** Number of trainable scalar weights in the model.

If train loss falls while validation loss rises, the model may be overfitting. Add data, improve data diversity, reduce model capacity, or apply regularization.

## Model generation settings

Inference currently uses temperature 0.75 and top-k 35. Lower temperature makes output less random; top-k limits sampling to the most likely next tokens. These settings do not guarantee correctness.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
Activate the virtual environment and install PyTorch using the correct command for your machine. Then verify with `python -c "import torch; print(torch.__version__)"`.

### `ModuleNotFoundError: No module named 'flask'`
Run `python -m pip install Flask` in the active virtual environment.

### Training is slow
Use CPU settings with fewer layers or smaller width for experiments. A CUDA-capable GPU can speed up training when PyTorch is installed with compatible CUDA support.

### App says `Knowledge mode`
The checkpoint is missing or could not be loaded. Run the training command, ensure `artifacts/minigpt.pt` exists, then restart Flask. Check the terminal output for a load error.

### Model output is gibberish
This is possible and expected with a tiny byte-level model and limited training data. Add more diverse, correct examples and train for longer. The app uses the local retriever for strong matches so common starter topics can still return curated explanations.
