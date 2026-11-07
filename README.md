# CodeMentor — AI-Based Programming Learning Assistant Using LLMs

A local programming tutor built with **Python, PyTorch, and Flask**. It includes a compact GPT-style Transformer that you train from randomly initialized weights on the supplied programming Q&A dataset. It does not use Gemini, OpenAI, Hugging Face hosted inference, or any other model API.

> **Important:** This is an educational, from-scratch mini language model — not a small copy of ChatGPT. The included dataset is deliberately small, so free-form generations can be inaccurate or incomplete. The app also has a local TF-IDF knowledge retriever; when it finds a strong match, it shows the curated reference answer rather than pretending that a weak model generation is reliable.

## Features

- GPT-style decoder-only Transformer implemented in `local_llm/model.py`.
- UTF-8 byte tokenizer implemented without a third-party tokenizer library.
- Training from random initialization using PyTorch next-token prediction.
- Included programming Q&A dataset covering Python, OOP, DSA, algorithms, DBMS, SQL, APIs, Git, debugging, and related topics.
- Local TF-IDF retrieval over the same dataset to improve answers for known topics.
- Flask backend with a separate HTML, CSS, and JavaScript frontend.
- Local inference; no AI API key is needed.
- CPU support; CUDA is used automatically when available.
- Training logs, model checkpoint, project report, architecture notes, and interview questions.

## 1. Requirements

- Python 3.10 or newer (Python 3.11/3.12 is a good general choice; Python 3.13 also works with a compatible PyTorch release).
- PyTorch 2.2 or newer.
- Flask 3.x.
- 4 GB RAM minimum is recommended for the small CPU configuration; more memory gives more room for experiments.

The app itself runs locally after dependencies and a trained checkpoint are present. You do not need an internet connection to ask questions. Internet is needed only if you have to download Python packages during setup.

## 2. Setup

Open a terminal in the project folder.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation is difficult on your machine, use the official PyTorch installation selector to choose the correct CPU/CUDA command for your operating system and Python version. Then install Flask with `pip install Flask`.

## 3. Train your own model

The included `artifacts/minigpt.pt` is a small demo checkpoint trained from random initialization for 800 steps on the included dataset. You can run the app immediately, then train again to reproduce or continue your experiment. Training learns to predict the next byte in programming question-and-answer examples. The training command overwrites the checkpoint at its default output path.

```bash
python -m local_llm.train
```

This uses the default settings (1,200 training steps) and writes:

```text
artifacts/minigpt.pt
```

For a quick smoke test, use a smaller run:

```bash
python -m local_llm.train --steps 20 --d-model 64 --n-heads 4 --n-layers 2 --block-size 128 --batch-size 8 --output artifacts/minigpt_test.pt
```

The quick command is for checking that the training pipeline works. It is not enough to produce a useful language model. To experiment with quality, increase steps, add good training examples, and compare training and validation loss. More steps do not automatically fix a small or repetitive dataset.

Useful training options:

| Option | Default | Meaning |
|---|---:|---|
| `--steps` | `1200` | Number of optimizer updates |
| `--batch-size` | `8` | Training sequences per update |
| `--block-size` | `128` | Maximum context length in byte tokens |
| `--d-model` | `64` | Transformer embedding width |
| `--n-heads` | `4` | Attention heads |
| `--n-layers` | `2` | Transformer blocks |
| `--learning-rate` | `0.0003` | AdamW learning rate |
| `--dropout` | `0.1` | Dropout probability |
| `--device` | `auto` | `auto` or `cpu` |

`d-model` must be divisible by `n-heads`. The `--device cpu` option forces CPU training. CUDA is selected automatically when available and `--device auto` is used.

## 4. Run the assistant

After training finishes:

```bash
python app.py
```

The server port is read from `PORT` in the project-root `.env` file (default: `5000`). Change that value to use a different port. Open **http://127.0.0.1:<PORT>** in your browser. Keep the terminal open while using the app. Press `Ctrl+C` in the terminal to stop it.

The Flask server binds to `127.0.0.1`, so it is intended for local use on your own computer.

### If you have not trained the model yet

The app can still start in **knowledge mode** using the included local Q&A dataset. The status indicator will say `Knowledge mode`. Run the training command and restart Flask to load your checkpoint.

## 5. Add your own training data

Edit `dataset/programming_qa.jsonl`. Each line must be one valid JSON object with `question` and `answer` fields:

```json
{"question":"What is a pointer?","answer":"A pointer is a variable that stores a memory address..."}
```

Guidelines:

1. Keep one question-answer pair per line.
2. Use clear, correct, self-contained explanations.
3. Include examples and edge cases when useful.
4. Avoid contradictory answers and copied material you do not have permission to use.
5. Add enough varied examples; a few examples cannot teach a general-purpose model.

After editing the dataset, train again. The retriever reads the dataset when the Flask app starts, so restart Flask after dataset edits.

## 6. Project structure

```text
local_programming_assistant/
├── app.py                       # Flask application and JSON endpoints
├── local_llm/
│   ├── model.py                 # GPT-style Transformer architecture
│   ├── tokenizer.py             # UTF-8 byte tokenizer
│   ├── data.py                  # Dataset parsing and batches
│   ├── train.py                 # Training loop and checkpoint saving
│   ├── retrieval.py             # Local TF-IDF search
│   └── assistant.py             # Generation + retrieval response logic
├── dataset/
│   └── programming_qa.jsonl     # Included training and retrieval examples
├── artifacts/
│   └── minigpt.pt               # Included from-scratch demo checkpoint
├── templates/
│   └── index.html               # Flask page template
├── static/
│   ├── css/style.css            # UI styles
│   └── js/app.js                # Chat interaction
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING_GUIDE.md
│   ├── TRAINING_RUN.md
│   ├── PROJECT_REPORT.md
│   └── INTERVIEW_QUESTIONS.md
├── tests/
│   └── test_core.py
├── requirements.txt
├── train_windows.bat
├── run_windows.bat
├── train_unix.sh
└── run_unix.sh
```

## 7. How an answer is produced

1. The browser sends the user's question to Flask at `/api/chat`.
2. `ProgrammingAssistant` searches the local Q&A dataset using TF-IDF similarity.
3. If `artifacts/minigpt.pt` exists, the trained MiniGPT generates a draft locally.
4. If a knowledge-base result is strongly relevant, the app displays that curated answer and exposes the raw model draft in an expandable section. Otherwise, it explains that the dataset needs a closer example and exposes any raw model draft as experimental output.
5. Flask returns JSON and the browser renders the response.

This design is intentionally transparent: a curated retrieval answer is not labelled as a model-generated answer.

## 8. Run tests

```bash
python -m unittest discover -s tests -v
```

Tests cover byte-tokenizer round trips, model forward/generation shapes, dataset parsing, and retrieval.

## 9. Important limitations

- The model is small and is trained on a compact educational dataset. It will not have the broad coding knowledge of a large pretrained LLM.
- Byte-level tokenization is simple and supports all UTF-8 text, but is less compute-efficient than modern subword tokenizers.
- The included examples are not a large-scale training corpus. Model quality depends strongly on dataset size, correctness, diversity, training time, and hardware.
- A falling training loss does not prove that answers are correct. Inspect outputs and keep a separate validation set when expanding the project.
- No generated code is executed automatically. Review code before running it.
- There is no web search, external API, account system, or cloud deployment in this project.

## 10. Suggested extensions

- Add a larger licensed programming dataset and a separate held-out validation set.
- Implement a tokenizer such as BPE and compare tokens-per-second and output quality.
- Add code formatting, syntax highlighting, and safe code execution in an isolated sandbox.
- Add a lesson planner, quizzes, difficulty levels, and progress tracking.
- Compare training from scratch with fine-tuning a small openly licensed base model as a separate experiment.

See `docs/` for the architecture, training guide, project report, and viva/interview questions.
