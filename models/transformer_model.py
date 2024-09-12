import torch
import torch.nn as nn

# Multi-head Attention and Transformer Block implemented here.
class MultiHeadAttention(nn.Module):
    # Implementation here as shown in the prior responses

class TransformerBlock(nn.Module):
    # Implementation as explained above

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_hidden_dim, num_layers, num_classes=1, max_len=250):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        return self.sigmoid(self.fc(x))

