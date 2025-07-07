import os
import librosa
import numpy as np
import soundfile as sf
import torch

def load_audio(file_path, sample_rate=16000):
    """
    读取音频文件并重采样
    """
    waveform, sr = sf.read(file_path)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
    return waveform, sample_rate

def extract_mel_features(waveform, sample_rate=16000, n_mels=80, win_length=400, hop_length=160):
    """
    从原始波形提取 Mel 频谱特征
    """
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0
    )
    log_mel_spec = librosa.power_to_db(spectrogram, ref=np.max)
    return log_mel_spec.T  # shape: (time, n_mels)

def normalize(features, mean=None, std=None):
    """
    对特征进行均值方差归一化
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-8), mean, std

def preprocess_file(file_path, sample_rate=16000, normalize_audio=True):
    """
    预处理单个音频文件：加载、特征提取、标准化
    """
    waveform, sr = load_audio(file_path, sample_rate)
    features = extract_mel_features(waveform, sample_rate)
    if normalize_audio:
        features, _, _ = normalize(features)
    return torch.tensor(features, dtype=torch.float)

def batch_preprocess_from_folder(folder_path, ext=".wav", max_files=None):
    """
    批量处理一个目录下的音频文件
    """
    processed = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith(ext)]
    for idx, fname in enumerate(file_list):
        if max_files and idx >= max_files:
            break
        fpath = os.path.join(folder_path, fname)
        features = preprocess_file(fpath)
        processed.append(features)
    return processed
