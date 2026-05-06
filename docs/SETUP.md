# ⚙️ Setup Guide

## Running on Kaggle (Recommended)

### 1. Create a new notebook

Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) → **New Notebook** → Upload `Darija_Pipeline_ITN_AUG_BENCHMARK.ipynb`

### 2. Configure the kernel

| Setting | Value |
|---------|-------|
| Language | Python |
| Accelerator | **GPU (T4 or P100)** ← required for NeMo |
| Internet | On |
| Persistence | Files and variables |

### 3. Add your dataset

Click **Add Data** → Upload Dataset:

```
your-dataset/
├── corpus.zip          # all your .wav audio files
├── Results.xlsx        # transcriptions (audio_id, ground_truth, qwen3-*)
├── ner_results.zip     # optional: pre-computed NER
└── itn_annotated_all.parquet  # optional: for supervised ITN training
```

### 4. First run

Run all cells. The installer will:
1. Install NeMo + all dependencies (~10 min)
2. Run a sanity check
3. **Exit the kernel** with `os._exit(0)` — this is intentional

### 5. Restart and run again

After the kernel restarts, run all cells again. This time the sentinel file exists and installation is skipped. The pipeline runs normally.

---

## Running Locally (Advanced)

> Not recommended — NeMo has complex CUDA requirements. Use Kaggle or a cloud GPU.

```bash
# Create environment
conda create -n darija python=3.10
conda activate darija

# Install CUDA toolkit (if not present)
conda install -c nvidia cuda-toolkit=11.8

# Install NeMo
pip install nemo_toolkit[asr,tts,audio]

# Install audio packages
pip install librosa==0.10.2.post1 soundfile pydub pyloudnorm

# Install NLP packages
pip install transformers>=4.44 datasets>=3.0.0 accelerate seqeval

# Pin scientific stack
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.14.1 \
            scikit-learn==1.5.2 pyarrow==20.0.0 \
            --force-reinstall --no-cache-dir

# Install data tools
pip install openpyxl xlsxwriter tqdm jupyter
```

Then update paths in STEP 0.5:
```python
# Replace Kaggle paths:
KAGGLE_INPUT = '/your/local/data'
BASE_DIR = '/your/local/output'
```

---

## Troubleshooting

### `RuntimeError: Dependency import check failed`
The pyarrow version is wrong. Run this cell separately:
```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'pyarrow==20.0.0', '--force-reinstall', '--no-cache-dir'])
```
Then restart kernel.

### NeMo model fails to load
NeMo will print a clear error. The pipeline **does not silently fall back** — this is intentional. Check:
- Is GPU enabled in Kaggle settings?
- Is `NEMO_ENHANCEMENT_MODEL` set to a valid HuggingFace model ID?

### No WAV files found
Verify your ZIP structure. The loader looks for `.wav` files at any depth inside the zip. Confirm:
```python
import zipfile
with zipfile.ZipFile('/kaggle/input/your-dataset/corpus.zip') as z:
    print([n for n in z.namelist() if n.endswith('.wav')][:5])
```

### Excel not found
The loader searches for `Results.xlsx` by name. If your file has a different name, set manually:
```python
EXCEL_PATH = '/kaggle/input/your-dataset/your_file.xlsx'
```
