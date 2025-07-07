import torch
import torch.nn as nn

class CrossDomainTransfer(nn.Module):
    def __init__(self, input_dim, shared_dim=256, private_dim=128):
        super(CrossDomainTransfer, self).__init__()
        self.shared_encoder = nn.Linear(input_dim, shared_dim)
        self.private_encoder = nn.Linear(input_dim, private_dim)
        self.merge = nn.Linear(shared_dim + private_dim, input_dim)

    def forward(self, x):
        """
        x: (batch, time, input_dim)
        """
        shared = torch.relu(self.shared_encoder(x))
        private = torch.relu(self.private_encoder(x))
        concat = torch.cat([shared, private], dim=-1)
        return self.merge(concat)
