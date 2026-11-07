# Project Report

## Title

**AI-Based Programming Learning Assistant Using LLMs**

## Abstract

This project presents a local programming learning assistant implemented in Python. It demonstrates the core workflow of a small language model: tokenization, causal Transformer architecture, next-token prediction, model training, checkpoint storage, and inference. The model is initialized randomly and trained on a compact programming question-and-answer dataset using PyTorch. A Flask application provides a browser-based interface, while a local TF-IDF retriever supplements model output with relevant curated explanations. The system does not require an external AI API key or hosted model endpoint.

## Problem statement

Programming learners often need explanations of concepts, examples, and help identifying mistakes. Hosted AI tools can be convenient, but they may require API credentials, internet access, or usage fees. This project explores how a small language model can be implemented and trained locally as an educational demonstration.

## Objectives

1. Implement a compact GPT-style decoder-only Transformer in PyTorch.
2. Create a tokenizer that can represent arbitrary UTF-8 input.
3. Train model parameters from random initialization using next-token prediction.
4. Build a local question-answering flow for programming topics.
5. Create a Flask web application with separate frontend files.
6. Document model design, training, limitations, and future improvements.

## Proposed solution

The solution combines a mini GPT model with a local knowledge retriever. The model generates text based on patterns learned from the training corpus. The retriever searches the included Q&A dataset and can provide a trusted reference when a close match exists. The user interacts with both components through the Flask web interface.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural-network layers, tensors, loss, optimization |
| Flask | Local web server and JSON endpoints |
| HTML | Page structure |
| CSS | Responsive visual design |
| JavaScript | Chat interface and API requests |
| JSONL | Training data format |

## Methodology

### Data preparation

Programming questions and answers are stored one JSON object per line. Each example is formatted with user and assistant markers, encoded into UTF-8 bytes, and terminated with special sequence markers.

### Model training

The decoder-only Transformer reads a context window and predicts the next byte. Cross-entropy loss measures prediction error. Gradients are computed through backpropagation, and AdamW updates model parameters.

### Inference

At inference time, the user's question is formatted using the same prompt markers and encoded. The trained model samples one byte at a time until it reaches an end marker or generation limit.

### Retrieval assistance

A local TF-IDF retriever ranks the included examples. A strongly relevant result can be displayed as the main response while the model's raw generation remains inspectable.

## Expected outcome

The project demonstrates an end-to-end local language-model pipeline, including model architecture, data handling, training, checkpointing, inference, retrieval, and web integration. It is particularly suitable for demonstrating foundational LLM concepts in an academic project or viva.

## Limitations

The corpus and network are intentionally small. The model does not have the scale, broad knowledge, or reliability of a commercial LLM. Byte tokenization is inefficient compared with common subword tokenizers. The included validation estimate is simple and should be replaced with a carefully curated held-out benchmark for serious evaluation. The tool does not execute user code.

## Future scope

- Expand and curate a larger dataset under suitable licenses.
- Use a subword tokenizer such as a trained BPE tokenizer.
- Build a dedicated held-out evaluation set and report perplexity or task accuracy.
- Add safe code execution in an isolated container.
- Add quiz generation, hints, difficulty selection, and learner progress.
- Compare a from-scratch model with a fine-tuned open-weight model as a separate study.

## Conclusion

The project provides a transparent and fully local educational implementation of a small GPT-style programming assistant. It shows how raw text is tokenized, how a causal Transformer learns next-token prediction, and how a trained checkpoint can be integrated into a practical Flask application. Its primary value is demonstrating the engineering and learning process, not replacing a large general-purpose assistant.
