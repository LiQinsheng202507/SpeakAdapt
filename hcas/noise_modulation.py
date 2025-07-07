import torch
import torch.nn as nn

class NoiseAwareFeatureModulation(nn.Module):
    def __init__(self, feature_dim, noise_dim=16):
        super(NoiseAwareFeatureModulation, self).__init__()
        self.noise_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, noise_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.gamma = nn.Linear(noise_dim, feature_dim)
        self.beta = nn.Linear(noise_dim, feature_dim)

    def forward(self, x):
        """
        x: (batch, time, feature_dim)
        """
        noise_repr = self.noise_encoder(x.transpose(1, 2)).squeeze(-1)  # (batch, noise_dim)
        gamma = self.gamma(noise_repr).unsqueeze(1)
        beta = self.beta(noise_repr).unsqueeze(1)
        return x * gamma + beta
