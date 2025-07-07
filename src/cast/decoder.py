import torch
import torch.nn as nn

class LinguisticAlignmentDecoder(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=256):
        super(LinguisticAlignmentDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoded_features):
        """
        encoded_features: (batch, time, dim)
        """
        output, _ = self.lstm(encoded_features)
        logits = self.classifier(output)
        return logits  # shape: (batch, time, vocab_size)
