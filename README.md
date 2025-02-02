# Attention Mechanism Implementation

A PyTorch implementation that demonstrates how attention mechanisms in neural networks can understand contextual relationships between words, similar to human comprehension. This implementation focuses specifically on learning size-related contextual relationships in sentences.

## Quick Start

1. Install PyTorch:
```bash
pip install torch
```

2. Save the code as `attention.py` and run:
```python
python attention.py
```

## Code Structure

The implementation consists of three main classes:

### 1. AttentionWithGradient
The core attention mechanism that:
- Embeds words into vectors
- Computes attention scores using Query, Key, Value transformations
- Produces contextualized representations

### 2. SizeLoss
Custom loss function that:
- Encourages attention to size-related words ('elephant', 'car', 'too')
- Penalizes excessive attention to irrelevant words
- Guides the model to learn meaningful size relationships

### 3. AttentionTrainer
Training framework that:
- Handles vocabulary creation and token conversion
- Manages the training loop
- Analyzes and visualizes attention patterns

## Example Output

When you run the code, you'll see the training progress and attention analysis:

```
Starting training...
Epoch: 0, Loss: 1.9872

Attention Pattern Analysis:
Token Importance Ranking:
Token: elephant        Attention Weight: 0.2515
Token: car            Attention Weight: 0.1971
Token: too            Attention Weight: 0.1090
Token: pink           Attention Weight: 0.0515
...

Combined weight for size-related words: 0.5576

Epoch: 20, Loss: 1.5634
...
```

## Understanding the Implementation

### 1. Word Embeddings
Each word is converted to a numerical vector:
```python
self.embedding = nn.Embedding(vocab_size, embedding_dim)
```

### 2. Attention Mechanism
Transforms embeddings into Query, Key, and Value vectors:
```python
Q = self.W_q(embedded)  # Query transformation
K = self.W_k(embedded)  # Key transformation
V = self.W_v(embedded)  # Value transformation
```

### 3. Attention Scoring
Computes attention weights through:
```python
attention_scores = torch.matmul(target_Q, K.transpose(-2, -1)) / self.scale
attention_weights = torch.softmax(attention_scores, dim=-1)
```

### 4. Loss Calculation
Custom loss function focuses on size-related words:
```python
size_loss = torch.mean(1.0 - size_related_weights)
other_loss = torch.mean(other_weights)
loss = size_loss + other_loss
```

## Customization

You can modify the example sentence or adjust parameters:

```python
# Change the input sentence
sentence = "The pink elephant tried to get into the car but it was too"

# Adjust training parameters
trainer = AttentionTrainer(sentence, embedding_dim=4)
trainer.train(epochs=200)  # Change number of epochs
```

## Parameters

- Embedding dimension: 4
- Learning rate: 0.001
- Optimizer: Adam
- Default epochs: 200

## License

MIT License

## Citation

```bibtex
@misc{attention_implementation,
  author = {Aekanun},
  title = {Attention Mechanism Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aekanun2020/attention-demo}
}
```