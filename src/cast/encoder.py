import torch
import torch.nn as nn

class MultiScaleAudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_scales=3):
        super(MultiScaleAudioEncoder, self).__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=2 ** (i + 1), stride=1, padding='same'),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for i in range(num_scales)
        ])
        self.proj = nn.Linear(hidden_dim * num_scales, hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, time_steps, feature_dim)
        """
        x = x.transpose(1, 2)  # to (batch, feature_dim, time)
        features = [scale(x) for scale in self.scales]
        concat = torch.cat(features, dim=1)  # concat along channel
        concat = concat.transpose(1, 2)      # back to (batch, time, dim)
        return self.proj(concat)
