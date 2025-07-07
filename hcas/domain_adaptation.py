import torch
import torch.nn as nn

class DomainAdaptationLayer(nn.Module):
    def __init__(self, input_dim, domain_embedding_dim=32):
        super(DomainAdaptationLayer, self).__init__()
        self.domain_proj = nn.Embedding(num_embeddings=10, embedding_dim=domain_embedding_dim)
        self.linear = nn.Linear(input_dim + domain_embedding_dim, input_dim)

    def forward(self, x, domain_id):
        """
        x: (batch, time, input_dim)
        domain_id: (batch,) integer tensor
        """
        domain_emb = self.domain_proj(domain_id)  # (batch, domain_embedding_dim)
        domain_emb = domain_emb.unsqueeze(1).repeat(1, x.size(1), 1)  # align time dimension
        x = torch.cat([x, domain_emb], dim=-1)
        return self.linear(x)
