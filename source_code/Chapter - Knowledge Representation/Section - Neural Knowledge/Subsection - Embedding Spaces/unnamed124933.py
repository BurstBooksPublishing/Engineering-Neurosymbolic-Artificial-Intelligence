import torch
import torch.nn as nn

# Define the vocabulary
vocab = ["cat", "dog", "animal", "pet"]
vocab_size = len(vocab)
embedding_dim = 5  # Each word will be represented by a 5-dimensional vector

# Create an embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Assign indices to words
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Example: Get the embedding for the word 'cat'
cat_idx = torch.tensor([word_to_idx['cat']])
cat_embedding = embedding_layer(cat_idx)

print("Embedding for 'cat':", cat_embedding)