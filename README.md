# 🎙️ CAST + HCAS: Context-Aware Speech Recognition Framework

A deep learning framework for building **robust, multilingual, and domain-adaptable speech recognition systems**. This project introduces:

- **CAST (Context-Aware Speech Transformer)** — a transformer-based encoder-decoder architecture with multi-scale acoustic encoding and attention mechanisms.
- **HCAS (Hierarchical Contextual Adaptation Strategy)** — an extensible strategy that incorporates noise-aware modulation, domain adaptation, and cross-domain transfer learning.

---

## 🚀 Features

- ✅ Multi-scale feature encoder for robust speech representations
- ✅ Context-aware attention module for noisy or accented speech
- ✅ Domain adaptation via lightweight embedding injection
- ✅ Cross-domain transfer learning support
- ✅ Evaluation with WER, CER, BLEU and attention visualization
- ✅ Modular codebase with PyTorch and Torchaudio

---

## 🧱 Project Structure

CAS adaptation modules ├── datasets/ # Data loaders & feature extraction ├── evaluation/ # Metrics & visualization tools ├── configs/ # Model and training config ├── experiments/ # Sample experiments ├── checkpoints/ # Pretrained models (optional) ├── main.py # Training & inference pipeline ├── requirements.txt # Dependency list ├── setup.py # Package installer └── README.md # You're here!
---

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourname/CAST-HCAS-SpeechRecognition.git
cd CAST-HCAS-SpeechRecognition
2. Install dependencies
pip install -r requirements.txt
pip install -e .

🏁 Quick Start
Run dummy training:

python main.py
Or using console command:
cast-hcas-train
This uses synthetic data. To integrate real speech datasets like LibriSpeech or CommonVoice, customize datasets/preprocessing.py.

📊 Evaluation
The evaluation/ module provides:

compute_wer, compute_cer, compute_bleu in metrics.py
plot_attention_weights, plot_feature_map in visualization.py

📦 Model Components
Module	Path	Description
🎧 Encoder	src/cast/encoder.py	Multi-scale Conv1D + Linear encoder
🧠 Attention	src/cast/attention.py	Context-aware multihead self-attention
🔡 Decoder	src/cast/decoder.py	LSTM decoder with vocab projection
🔊 Noise Modulation	src/hcas/noise_modulation.py	Per-sample noise-conditioned feature reshaping
🧩 Domain Adaptation	src/hcas/domain_adaptation.py	Domain embedding injection
🌍 Transfer Learning	src/hcas/transfer_learning.py	Shared-private feature decomposition
📁 Example Experiments
experiments/
├── multilingual/         # Transcription across languages
├── noisy_real_time/      # Noisy live speech evaluation
└── domain_specific/      # Healthcare / customer support cases
📜 License
This project is released under the MIT License.
🤝 Contributing
PRs and issues are welcome! Feel free to fork and adapt for your domain-specific speech needs.
