# System Architecture

## 1. High-level design

CodeMentor is a local application with four main layers:

```text
Browser UI (HTML / CSS / JavaScript)
                |
                | HTTP JSON request
                v
          Flask backend
                |
                v
       ProgrammingAssistant
          /          \
         /            \
 Local TF-IDF       MiniGPT inference
 retriever          (PyTorch)
         \            /
          \          /
           Response selection
                |
                v
          JSON response
```

The browser does not call a model provider. Flask and PyTorch run on the same computer.

## 2. Language model

The model is a decoder-only Transformer inspired by the GPT architecture. It is initialized randomly and trained on the project's own text examples.

### Components

1. **Byte tokenizer:** Converts UTF-8 input into byte IDs from 0 to 255. ID 256 is end-of-sequence and ID 257 is beginning-of-sequence. This guarantees that new words can be represented without an unknown-word token.
2. **Token embedding:** Maps token IDs to vectors of size `d_model`.
3. **Positional embedding:** Adds information about token positions up to `block_size`.
4. **Causal self-attention:** Allows each position to attend to itself and earlier positions, not future positions.
5. **Feed-forward network:** Applies a two-layer MLP with GELU activation to each position.
6. **Residual connections and layer normalization:** Help stabilize optimization and information flow.
7. **Language-model head:** Produces a score for every vocabulary item at every sequence position.

### Training objective

The model predicts the next byte. If the input sequence is:

```text
The answer is 42
```

The learning target is the same sequence shifted one token to the right. Cross-entropy loss measures how well the model predicts each next token. AdamW updates the model parameters to reduce this loss.

The causal attention mask prevents the model from seeing future target bytes while training.

## 3. Retrieval component

`LocalRetriever` creates TF-IDF-style vectors from the questions and answers in the local JSONL dataset. It ranks examples by cosine similarity to the user's query. This provides a reliable reference when the user's question is close to an included topic.

Retrieval is not another LLM and does not train neural weights. It is a lightweight information-retrieval component that complements the small language model.

## 4. Response selection

- Strong match: show the matching curated dataset answer.
- Moderate match: show the closest local reference with a clear qualification if model generation is not useful.
- Weak/no match: explain that the dataset needs a closer example and expose the raw local model draft as experimental output, if a checkpoint is loaded.
- When a model is trained, its raw draft can be inspected below the main answer.

## 5. Main modules

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, input validation, JSON response |
| `local_llm/model.py` | Attention, Transformer blocks, MiniGPT |
| `local_llm/tokenizer.py` | UTF-8 byte encoding and decoding |
| `local_llm/data.py` | JSONL loading and batch construction |
| `local_llm/train.py` | Training loop, evaluation estimate, checkpoint |
| `local_llm/retrieval.py` | TF-IDF similarity search |
| `local_llm/assistant.py` | Model loading, local generation, response selection |
| `templates/index.html` | UI markup |
| `static/css/style.css` | Visual design and responsive layout |
| `static/js/app.js` | Chat requests and UI interactions |

## 6. API routes

### `GET /`
Returns the chat interface.

### `GET /api/status`
Returns whether a local checkpoint loaded, device name, dataset size, and basic training metadata.

### `POST /api/chat`
Request:

```json
{"message":"Explain binary search"}
```

Success response fields include `answer`, `source`, `matched_question`, `similarity`, `model_output`, and `model_loaded`.

## 7. Security and privacy

The server binds to `127.0.0.1`, has a small request-size limit, validates user input, and does not run submitted code. This is intended as a local educational prototype, not a hardened multi-user production service. Do not expose the development server directly to the public internet.
