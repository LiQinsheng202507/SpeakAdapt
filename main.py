import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.cast.encoder import MultiScaleAudioEncoder
from src.cast.attention import ContextAwareAttention
from src.cast.decoder import LinguisticAlignmentDecoder

from src.hcas.noise_modulation import NoiseAwareFeatureModulation
from src.hcas.domain_adaptation import DomainAdaptationLayer
from src.hcas.transfer_learning import CrossDomainTransfer

# ─────────────────────────────────────────────────────────────── #
# 模型组合：CAST + HCAS
# ─────────────────────────────────────────────────────────────── #

class CAST_HCAS_Model(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(CAST_HCAS_Model, self).__init__()
        self.encoder = MultiScaleAudioEncoder(input_dim=input_dim)
        self.noise_mod = NoiseAwareFeatureModulation(feature_dim=256)
        self.attn = ContextAwareAttention(dim=256)
        self.domain_adapt = DomainAdaptationLayer(input_dim=256)
        self.transfer = CrossDomainTransfer(input_dim=256)
        self.decoder = LinguisticAlignmentDecoder(input_dim=256, vocab_size=vocab_size)

    def forward(self, x, domain_id):
        x = self.encoder(x)
        x = self.noise_mod(x)
        x = self.attn(x)
        x = self.domain_adapt(x, domain_id)
        x = self.transfer(x)
        logits = self.decoder(x)
        return logits

# ─────────────────────────────────────────────────────────────── #
# 简化数据生成（用于测试训练流程）
# ─────────────────────────────────────────────────────────────── #

def get_dummy_data(num_samples=32, time_steps=100, input_dim=80, vocab_size=30):
    X = torch.randn(num_samples, time_steps, input_dim)
    y = torch.randint(0, vocab_size, (num_samples, time_steps))
    domain_ids = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(X, y, domain_ids), batch_size=8)

# ─────────────────────────────────────────────────────────────── #
# 训练流程
# ─────────────────────────────────────────────────────────────── #

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        x, y, domain_ids = [item.to(device) for item in batch]
        optimizer.zero_grad()
        logits = model(x, domain_ids)

        # 平铺处理 loss（交叉熵假设已对齐标签）
        loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Training batch complete, loss = {loss.item():.4f}")

# ─────────────────────────────────────────────────────────────── #
# 主运行函数
# ─────────────────────────────────────────────────────────────── #

def main():
    input_dim = 80
    vocab_size = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CAST_HCAS_Model(input_dim=input_dim, vocab_size=vocab_size).to(device)
    dataloader = get_dummy_data(input_dim=input_dim, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):  # 示例 3 个 epoch
        print(f"Epoch {epoch+1}")
        train(model, dataloader, optimizer, criterion, device)

if __name__ == "__main__":
    main()
