import torch
import torch.nn as nn
import torch.nn.functional as F

class sentimentBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_size):
        """
        Bidirectional LSTM for sentiment analysis.
        
        Parameters:
          embedding_matrix: Pretrained embedding matrix or dummy matrix (shape: [num_words, embed_dim]).
          hidden_dim: Hidden state dimension.
          output_size: Number of sentiment classes (e.g. 3).
        """
        super(sentimentBiLSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        # Create embedding layer initialized with the given matrix.
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embeddings.
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # Fully connected layer; input dimension is hidden_dim*2 due to bidirectionality.
        self.fc = nn.Linear(hidden_dim * 2, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_length] (integer indices)
        embeds = self.embedding(x)            # → [batch_size, seq_length, embed_dim]
        lstm_out, _ = self.lstm(embeds)         # → [batch_size, seq_length, hidden_dim*2]
        lstm_out = lstm_out[:, -1, :]           # Take last time step → [batch_size, hidden_dim*2]
        out = self.fc(lstm_out)                 # → [batch_size, output_size]
        return out
