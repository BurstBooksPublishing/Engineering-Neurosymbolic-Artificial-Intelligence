import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class SymbolicEmbedding(nn.Module):
    def __init__(self, num_symbols, embedding_dim):
        super(SymbolicEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_symbols, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Example usage
num_symbols = 100  # Assume 100 different symbolic tags
embedding_dim = 768  # Same as BERT's hidden size

symbolic_embedding = SymbolicEmbedding(num_symbols, embedding_dim)

# Example symbolic tags for a sentence
symbolic_tags = torch.tensor([1, 23, 45, 67, 89])  # Random example tags

embedded_symbols = symbolic_embedding(symbolic_tags)

# Now feed this into a transformer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Hello, this is an example sentence."
inputs = tokenizer(sentence, return_tensors="pt")

outputs = model(inputs, inputs_embeds=embedded_symbols.unsqueeze(0))