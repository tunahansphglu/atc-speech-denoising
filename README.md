# Deep Learning for Noise Reduction in Aviation Communications

This repository contains the complete implementation of a speech
enhancement framework evaluated on aviation-realistic compound noise.
Five methods are compared:

| Model | Type | Params |
|---|---|---|
| Spectral Subtraction | Classical DSP | — |
| Wiener Filter | Classical DSP | — |
| LSTM Spectral Masking | Deep Learning (trained) | 2.83 M |
| Wave-U-Net Lite | Deep Learning (trained) | 3.87 M |
| Conv-TasNet | Deep Learning (pre-trained) | 5.07 M |

### Key Results (N = 40 test samples)

| Model | PESQ | STOI | SI-SNRi (dB) | SDRi (dB) |
|---|---|---|---|---|
| Noisy Input | 1.276 | 0.861 | 0.00 | 0.00 |
| Spectral Subtraction | 1.315 | 0.860 | +0.71 | +0.78 |
| Wiener Filter | 1.675 | 0.881 | +6.62 | +6.88 |
| **LSTM Masking** | **1.792** | **0.902** | +7.67 | +6.49 |
| **Wave-U-Net Lite** | 1.616 | 0.894 | **+7.87** | −18.81 |
| ~~Conv-TasNet~~ | ~~1.073~~ | ~~0.161~~ | ~~−32.90~~ | ~~−112.5~~ |

> **Main finding:** Pre-trained Conv-TasNet catastrophically degrades all
> 40 test samples due to domain mismatch (trained on speaker separation,
> not denoising). Domain-matched models outperform classical baselines
> significantly. See the paper for full analysis.

---

## Repository Structure

```
atc-speech-denoising/
├── notebooks/
│   └── ATC_Denoising_Final.ipynb   # Complete Kaggle/Colab notebook
├── atc_denoising_results
```

---

## Quickstart

###  Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload Notebook** → select `notebooks/ATC_Denoising_Final.ipynb`
3. **Runtime → Change Runtime Type → GPU (T4)**
4. **Runtime → Run All**

### Kaggle / Colab (used for paper results)

| Setting | Value |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| Runtime | ~30 min (Train + Eval) |

## Reproducing Paper Results

All experiments use `SEED = 42` (set in `src/config.py`).
The evaluation set is fully deterministic: given the same seed,
the same 40 noisy-clean pairs are generated.

```python
from src.config import Config
from src.dataset import load_clean_wavs, build_dataset

# Load 40 clean utterances from LibriSpeech validation-clean
clean_wavs = load_clean_wavs(n=Config.train.n_samples)

# Build deterministic noisy-clean pairs
test_items = build_dataset(clean_wavs, seed=Config.train.seed)

# test_items[i] = {'id', 'clean', 'noisy', 'noise_type', 'snr_db'}
```

To reproduce a specific noise/SNR combination:

```python
from src.noise_synthesis import NoiseSynthesizer
import numpy as np

synth = NoiseSynthesizer(seed=42, sr=16000)

# Engine noise at 0 dB SNR
clean = np.zeros(64000, dtype=np.float32)  # replace with real speech
noise = synth.generate("engine", n_samples=len(clean))
noisy = synth.mix_at_snr(clean, noise, snr_db=0)
```

## Noise Categories

| Category | Description | Key Parameters |
|---|---|---|
| `engine` | Turbofan broadband + 120/240 Hz tones | Pink noise + sinusoids |
| `radio` | VHF band-limited + AM residue + squelch | 300–3400 Hz BP filter |
| `cockpit` | Fan hum (400 Hz) + aircon hiss | Tonal + HP filter |
| `rfi` | Sparse interference bursts | 3–12 bursts, 10–100 ms |

---

## Evaluation Metrics

| Metric | Range | Higher = Better |
|---|---|---|
| PESQ (wideband, ITU-T P.862) | −0.5 to 4.5 | ✓ |
| STOI | 0 to 1 | ✓ |
| SI-SNRi (dB) | −∞ to +∞ | ✓ |
| SDRi (dB) | −∞ to +∞ | ✓ |

---
