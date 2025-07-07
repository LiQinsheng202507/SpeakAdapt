import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attention_weights(attention_matrix, save_path=None, title="Attention Map"):
    """
    显示或保存注意力图
    Args:
        attention_matrix: (T_q, T_k) torch.Tensor or np.ndarray
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Key Steps")
    plt.ylabel("Query Steps")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()

def plot_feature_map(features, title="Mel Features", save_path=None):
    """
    可视化 log-Mel 特征
    Args:
        features: (time, freq)
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(features.T, aspect='auto', origin='lower', cmap='magma')
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
