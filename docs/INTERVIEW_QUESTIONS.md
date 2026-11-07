# Viva / Interview Questions and Suggested Answers

## Project overview

### 1. What is the objective of this project?
To build a local programming learning assistant and demonstrate the complete workflow of a small language model, from tokenization and training to inference and web integration.

### 2. Why did you avoid an external API?
The goal was to understand model training and inference locally, avoid API credentials and per-request charges, and make the application usable without sending prompts to an external provider.

### 3. Is this model trained from scratch?
Yes. Its weights are initialized randomly and optimized on the included dataset. It does not load pretrained language-model weights. PyTorch provides the tensor and neural-network operations.

### 4. Is this model comparable to ChatGPT?
No. It is a small educational model trained on a compact dataset. It demonstrates the architecture and training process but does not have comparable scale, knowledge, or answer quality.

## LLM and Transformer concepts

### 5. What does LLM stand for?
Large Language Model. It is a neural model trained on large amounts of text to model language and generate text. This project uses a much smaller language model for educational purposes.

### 6. What is a Transformer?
A neural-network architecture that uses attention to process relationships between tokens. It can model context more flexibly than sequential recurrent architectures.

### 7. What is a decoder-only Transformer?
A Transformer that uses causal attention and is commonly trained to predict the next token. GPT-style language models use this design.

### 8. What is self-attention?
Self-attention lets each position create a weighted combination of information from earlier positions in the sequence. In causal language modeling, future positions are masked.

### 9. Why is causal masking required?
The model should predict the next token without seeing that future token. A causal mask prevents information leakage from future positions during training and generation.

### 10. What are query, key, and value vectors?
They are learned projections of the token representations. Queries are compared with keys to obtain attention weights, and those weights combine value vectors.

### 11. What are embeddings?
Embeddings map discrete token IDs to dense learned vectors that the neural network can process.

### 12. What is positional encoding or positional embedding?
It supplies information about token order. This implementation uses learned positional embeddings up to a fixed context length.

### 13. What is a tokenizer?
A tokenizer converts text into discrete model inputs. This project uses UTF-8 bytes, so every text character can be represented as one or more byte IDs.

### 14. Why use byte-level tokenization?
It is dependency-free and avoids out-of-vocabulary words. Its disadvantage is that sequences are longer than with efficient subword tokenizers.

## Training concepts

### 15. What is next-token prediction?
The model receives a sequence and learns to predict the next token at each position. The target sequence is the input shifted by one token.

### 16. What is cross-entropy loss?
It measures how much probability the model assigns to the correct next token. Lower loss means better prediction on the measured data, but it does not guarantee useful or correct answers.

### 17. What is backpropagation?
It calculates gradients of the loss with respect to model parameters using the chain rule, allowing an optimizer to update those parameters.

### 18. What is an optimizer?
An optimizer applies parameter updates using the gradients. This project uses AdamW, which combines adaptive gradient updates with decoupled weight decay.

### 19. What is an epoch and what is a training step?
An epoch is one pass through a dataset. A training step is one optimizer update using a batch. This starter trainer samples random windows, so its `steps` count optimizer updates rather than complete epochs.

### 20. What is overfitting?
Overfitting occurs when a model learns the training examples too specifically and performs worse on unseen data. A widening gap between training and validation loss can be a warning sign.

### 21. Why is validation loss important?
It estimates performance on data not used for parameter updates. It can help identify overfitting, although a small or poorly chosen validation set can be misleading.

### 22. What is a checkpoint?
A checkpoint stores learned parameters and configuration so the model can be loaded for inference without training again.

## Application and architecture

### 23. Why did you use Flask?
Flask is a lightweight Python web framework. It exposes local routes for the browser interface and JSON responses from the assistant.

### 24. Why separate HTML, CSS, and JavaScript?
Separation of concerns makes the UI easier to maintain and lets the Python backend focus on model and application logic.

### 25. What is TF-IDF retrieval doing in the project?
It ranks local question-answer examples by term importance and cosine similarity. It is a retrieval technique, not a generative language model.

### 26. Why combine retrieval with the mini language model?
The tiny model may generate weak or unreliable text. Retrieval can supply a relevant curated answer for known topics, while the raw model output remains available for inspection.

### 27. Does the app need internet access to answer questions?
No, not after dependencies and the model checkpoint are installed. The Flask server, model, and retriever operate locally.

### 28. Does the project execute submitted code?
No. It only provides explanations and text responses. Automatically running user-submitted code would require a carefully isolated sandbox.

### 29. How would you improve this project?
I would expand and validate the dataset, train a BPE tokenizer, improve the evaluation setup, tune model hyperparameters, add code-aware features, and implement safe sandboxed execution.

### 30. How would you evaluate the assistant?
Use a held-out test set with representative programming questions, measure loss/perplexity for language modeling, and manually or automatically assess answer correctness, relevance, clarity, and code validity. Retrieval accuracy should be evaluated separately from generation quality.
