# 🎙️ Darija Speech Data Preparation Pipeline

> **GIZA++ · Thutmose · NeMo · ITN Benchmark**  
> A full end-to-end pipeline for preparing Moroccan Darija speech corpora — from raw audio to aligned, normalized, and annotated datasets.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Outputs](#-outputs)
- [Benchmark Results](#-benchmark-results)
- [Project Structure](#-project-structure)
- [Notes & Configuration](#-notes--configuration)

---

## 🔍 Overview

This notebook implements a **production-ready NLP/speech pipeline** specifically designed for **Moroccan Darija** — a low-resource dialect mixing Arabic, French, and English.

It was built to run on **Kaggle** (GPU-enabled kernel) and covers the full data preparation workflow needed before fine-tuning ASR or NLP models:

| Stage | Description |
|-------|-------------|
| Audio preprocessing | Standardization → Denoising → Normalization |
| Text processing | Spoken-to-written conversion, Arabic normalization |
| Alignment | Token-level alignment via **GIZA++** |
| Semiotic tagging | **ITN** (Inverse Text Normalization) with two methods |
| Benchmarking | Rule-based vs. supervised model comparison |
| Export | Formatted Excel + CSV + ZIP output |

---

## 🏗️ Pipeline Architecture

```
Raw Audio (.wav)
       │
       ▼
[STEP 1] Audio Standardization
   └── 16kHz mono 16-bit PCM (librosa)
       │
       ▼
[STEP 2] Audio Denoising
   └── NVIDIA NeMo (SchroedingerBridge model)
       Sample-based to avoid timeout on large corpora
       │
       ▼
[STEP 3] LUFS Normalization (EBU R128)
   └── pyloudnorm — broadcast-ready loudness
       │
       ▼
[STEP 3.5] Audio Data Augmentation
   └── Gain · Noise · Time shift · Speed · Pitch
       Polarity · Clipping · Band-pass · Reverb
       │
       ▼
[STEP 4] Language Identification (token-level)
   ├── Method A: Rule-based (Arabic/French/English patterns)
   └── Method B: Supervised XLM-RoBERTa classifier
       │
       ▼
[STEP 5] Text Normalization + Semiotic Labeling
   └── Spoken → Written form (numbers, dates, money, etc.)
       │
       ▼
[STEP 6] GIZA++ Token Alignment
   └── Spoken tokens ↔ Written tokens alignment table
       │
       ▼
[STEP 8] Full Corpus Processing
   └── All steps applied per row, results aggregated
       │
       ▼
[STEP 9] Export
   └── Excel (styled) · CSV (UTF-8) · ZIP download
```

---

## ✨ Features

### Audio Pipeline
- 🎧 **Standardization** to 16kHz mono PCM (handles any input format via librosa)
- 🔇 **NeMo Denoising** using `nvidia/se_den_sb_16k_small` — real neural denoising, not spectral subtraction
- 📊 **EBU R128 loudness normalization** (-23 LUFS target)
- 🔀 **Data augmentation** with 9 controlled transformations

### Text & NLP Pipeline
- 🔤 **Arabic/Darija normalization** — maps variant Unicode characters to canonical forms
- 🔢 **ITN (Inverse Text Normalization)** — converts written numbers/dates/money to spoken form
- 🏷️ **Semiotic class detection** — 12 classes: `CARDINAL`, `MONEY`, `DATE`, `ORDINAL`, `MEASURE`, `DECIMAL`, `LETTER`, `TIME`, `PUNCT`, `TELEPHONE`, `ADDRESS`, `ELECTRONIC`
- 🌍 **Language ID** — detects Arabic, Darija, French, English tokens

### Alignment
- ⚙️ **GIZA++** word alignment (industry standard for SMT)
- 🔄 **Thutmose-style rules** as fallback

### Benchmarking
- 📈 **Side-by-side comparison**: GIZA++/Thutmose rules vs. supervised XLM-RoBERTa
- Metrics: Accuracy + Macro F1 per semiotic class

---

## 📦 Prerequisites

### Hardware
- **GPU required** (Kaggle GPU kernel or equivalent)
- Minimum 8GB VRAM recommended for NeMo

### Input Files (upload to Kaggle Dataset)
| File | Description |
|------|-------------|
| `corpus.zip` | ZIP of `.wav` audio files |
| `Results.xlsx` | Excel with columns: `audio_id`, `ground_truth`, `qwen3-asr-checkpoint*` |
| `ner_results.zip` | *(optional)* Pre-computed NER results |
| `itn_annotated_all.parquet` | *(optional)* Labeled ITN data for supervised training |

### Auto-installed packages
The notebook installs all dependencies on first run:

```
nemo_toolkit[asr,tts,audio]   # NVIDIA NeMo
librosa · soundfile · pydub · pyloudnorm
transformers · datasets · accelerate · seqeval
openpyxl · xlsxwriter · tqdm
numpy==1.26.4 · pandas==2.2.2 · pyarrow==20.0.0
```

> ⚠️ **Kaggle restart required** after first install — the notebook will tell you when.

---

## 🚀 Quick Start

1. **Upload** the notebook to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. **Add your dataset** via *Add Data* (your `.wav` zip + `Results.xlsx`)
3. **Enable GPU** in Settings → Accelerator → GPU T4 or P100
4. **Run all cells** — on first run, it installs deps and exits; restart and run again
5. **Download output** from the Output tab → `Darija_Pipeline_GIZA_Thutmose_Benchmark.zip`

---

## 📖 Step-by-Step Guide

### STEP 0 — Install Dependencies

The `pip_install()` helper installs packages safely in the Kaggle environment with compatibility pinning. A sentinel file prevents reinstallation on subsequent runs.

```python
# Key: scientific stack is force-reinstalled LAST to ensure compatibility
pip_install('numpy==1.26.4', 'pandas==2.2.2', 'pyarrow==20.0.0',
            extra_args=['--force-reinstall', '--no-cache-dir'])
```

### STEP 0.5 — Load Files

Auto-discovers your uploaded files by walking `/kaggle/input`:
- Finds `.zip` (audio), `.xlsx` (transcriptions), NER zip
- Extracts `.wav` files to `darija_corpus/audio_raw/`

### STEP 1 — Audio Standardization

```python
# Converts any WAV to: 16kHz · mono · 16-bit PCM
# Rejects: silent files (<1e-5 RMS), files shorter than 0.1s
standardize_audio(input_path, output_path, sr=16000)
```

### STEP 2 — NeMo Denoising

Uses `SchroedingerBridgeAudioToAudioModel` from NVIDIA NeMo. Processes a configurable sample (`NEMO_SAMPLE_SIZE = 10` by default), copies standardized audio for the rest to keep the pipeline fast.

```python
NEMO_ENHANCEMENT_MODEL = "nvidia/se_den_sb_16k_small"
NEMO_SAMPLE_SIZE = 10  # adjust for full denoising
```

### STEP 3 — LUFS Normalization

Targets `-23 LUFS` (EBU R128 broadcast standard) using `pyloudnorm`.

### STEP 3.5 — Audio Augmentation

9 augmentation types applied with controlled probability:

| Transform | Description |
|-----------|-------------|
| Gain | ±6dB random gain |
| Noise | Gaussian noise addition |
| Time shift | Random temporal shift |
| Speed | 0.9×–1.1× speed change |
| Pitch | ±2 semitones |
| Polarity | Signal inversion |
| Clipping | Soft/hard clipping |
| Band-pass | Frequency filtering |
| Reverb | Room simulation |

### STEP 4 — Language Identification

Two methods compared:
- **Rules-first**: regex patterns for Arabic/Darija/French/English character sets
- **Supervised**: fine-tuned XLM-RoBERTa for token classification

```python
USE_SUPERVISED_LANG_ID = True  # toggle in config
```

### STEP 4d — Benchmark: Rules vs. Supervised

Outputs a comparison DataFrame with accuracy + macro F1 for both methods.

### STEP 5 — Text Normalization + ITN

The `spoken_to_written()` function handles:
- Arabic letter normalization (hamza variants, alef, teh marbuta...)
- Number conversion (Eastern/Western Arabic numerals)
- Currency detection (درهم → MAD)
- Date patterns, measurement units

### STEP 6 — GIZA++ Alignment

Builds a token-level alignment table between spoken and written forms. Falls back to length-based alignment if GIZA++ is unavailable.

### STEP 8 — Full Corpus Processing

Applies all steps per row and assembles a rich result DataFrame with 19 columns per entry.

### STEP 9 — Export

Generates:
- Styled Excel with RTL layout (right-to-left for Arabic)
- UTF-8 CSV
- ZIP archive for download

---

## 📤 Outputs

After running the full pipeline, you get:

```
📁 darija_corpus/outputs/
├── corpus_giza_thutmose_benchmark.xlsx   # Main annotated corpus (styled, RTL)
└── corpus_pipeline_giza_thutmose_benchmark.csv  # Raw CSV (UTF-8-sig)

📦 /kaggle/working/
└── Darija_Pipeline_GIZA_Thutmose_Benchmark.zip  # All outputs zipped
```

### Excel Columns

| Column | Description |
|--------|-------------|
| `audio_id` | File identifier |
| `spoken` | Cleaned spoken form |
| `written` | Written normalized form |
| `written_normalized` | With number normalization |
| `lang_AR_DA_pct` | Arabic/Darija percentage |
| `lang_FR_pct` | French percentage |
| `lang_EN_pct` | English percentage |
| `fr_en_tokens` | Foreign language tokens detected |
| `alignment_pairs` | GIZA++ alignment (spoken→written) |
| `semiotic_classes` | ITN labels found |
| `normalization_applied` | Was text normalized? |
| `token_count` | Number of tokens |
| `ground_truth_original` | Original transcription |
| `qwen_transcription` | ASR model output |
| `lang_id_method` | Which LangID method was used |

---

## 📊 Benchmark Results

The pipeline compares two ITN methods:

| Method | What it does |
|--------|-------------|
| `giza_thutmose_rules` | Deterministic regex + GIZA++ rules |
| `supervised_itn_xlm_roberta` | Fine-tuned XLM-RoBERTa on ITN parquet |

Results printed to console and saved to Excel sheet `Benchmark_Lang_ID`.

---

## 📁 Project Structure

```
darija-pipeline/
│
├── README.md                                    # This file
├── Darija_Pipeline_ITN_AUG_BENCHMARK.ipynb     # Main Kaggle notebook
│
├── docs/
│   ├── SETUP.md          # Detailed environment setup
│   └── ARCHITECTURE.md   # Deep dive into each component
│
└── scripts/
    └── verify_env.py     # Quick environment check script
```

---

## ⚙️ Notes & Configuration

### Key constants to tune

```python
NEMO_SAMPLE_SIZE = 10         # Increase for more denoising coverage
NEMO_ENHANCEMENT_MODEL = "nvidia/se_den_sb_16k_small"
USE_SUPERVISED_LANG_ID = True  # False → faster rule-based
RUN_ITN_SUPERVISED_TRAINING = True  # False → rules only
SEED = 42
```

### Kaggle-specific notes

- The notebook uses a **sentinel file** (`/kaggle/working/.darija_deps_installed_v2`) to skip reinstallation on rerun
- NeMo has a known conflict with `pyarrow` extension registration — the `patch_nemo_runtime_compat()` function fixes this
- The scientific stack (`numpy/pandas/scipy/pyarrow`) must be force-reinstalled **last** because NeMo shifts versions

### Semiotic Classes

```
CARDINAL · MONEY · DATE · ORDINAL · MEASURE
DECIMAL · LETTER · TIME · PUNCT · TELEPHONE · ADDRESS · ELECTRONIC
```

---

## 🤝 Contributing

PRs welcome. Key areas for improvement:
- Extend augmentation transforms
- Add more semiotic classes for Darija-specific entities
- Improve GIZA++ fallback alignment quality

---

*Pipeline built for Moroccan Darija NLP research — handles the unique code-switching between Arabic, French, and English common in Darija speech.*
