# 🏗️ Architecture Deep Dive

## Why these components?

### NVIDIA NeMo for Denoising

NeMo's `SchroedingerBridgeAudioToAudioModel` is a **diffusion-based speech enhancement** model. Unlike traditional spectral subtraction:
- It operates directly on waveforms (not spectrograms)
- Handles non-stationary noise well (traffic, crowd, music)
- Produces natural-sounding output without musical noise artifacts

The pipeline uses the `nvidia/se_den_sb_16k_small` checkpoint — a compact model that runs on Kaggle's T4 GPU without OOM issues.

**Why sample-only mode?** NeMo inference is ~3–5× slower than real-time on T4. Processing 1000 files would take hours. The default `NEMO_SAMPLE_SIZE=10` processes a quality-check sample while copying standardized (not denoised) audio for the rest.

### GIZA++ for Alignment

GIZA++ (Och & Ney, 2003) is the standard tool for **word alignment in Statistical Machine Translation**. Here it aligns:
- **Source**: spoken/Darija form (as pronounced)
- **Target**: written form (normalized Arabic script)

This alignment is critical for:
- Training TTS systems that need phoneme-grapheme correspondences
- Building ITN (Inverse Text Normalization) rules
- Data augmentation (swap spoken↔written forms)

### XLM-RoBERTa for Language ID and ITN

`xlm-roberta-base` (Conneau et al., 2020) was pretrained on 100 languages including Arabic and French — making it a natural fit for Darija code-switching.

The pipeline fine-tunes two separate classifiers:
1. **Language ID**: token-level classification (ARABIC / DARIJA / FRENCH / ENGLISH / MIXED)
2. **ITN semiotic labeling**: span-level classification (12 semiotic classes)

Both use the same training pattern: `span: {token} </s> sentence: {full_sentence}` — this gives the model both local (token) and global (sentence) context.

---

## Semiotic Classes

| Class | Examples |
|-------|---------|
| `CARDINAL` | ثلاثة، 3، ثلاث |
| `MONEY` | 200 درهم، 50 MAD |
| `DATE` | 15 مارس 2024، يناير |
| `ORDINAL` | الثالث، 3ème |
| `MEASURE` | 5 كيلو، 100km |
| `DECIMAL` | 3.5، ٣٫٥ |
| `LETTER` | أ، B |
| `TIME` | 3 ديعية، 15h30 |
| `PUNCT` | ، . ؟ ! |
| `TELEPHONE` | 0661234567، +212 |
| `ADDRESS` | شارع محمد الخامس |
| `ELECTRONIC` | test@gmail.com، www.example.ma |

---

## Data Augmentation Strategy

### Audio augmentation (STEP 3.5)

Augmentation is applied **after** normalization, not before, to ensure augmented files maintain proper loudness levels. Transforms are applied with independent probabilities so combinations occur naturally.

### ITN augmentation (inside STEP 5)

The `augment_itn_span()` function generates variants for training:
- Converts Western ↔ Eastern Arabic numerals (0-9 ↔ ٠-٩)
- Normalizes currency spellings (درهم / dh / MAD)
- Normalizes measurement units (kg / كيلو)
- Applies Arabic letter normalization

Factor-2 augmentation means each training example generates up to 2 additional variants.

---

## Benchmark Design

### Language ID benchmark (STEP 4d)

Tests the same held-out split with both methods. Reports per-class precision/recall/F1 using `sklearn.classification_report`.

### ITN benchmark (STEP 4d)

Uses a 70/15/15 train/val/test split with stratification. The test set is evaluated with both methods — rules can only label what they recognize, so incomparable spans are filtered before computing rule-based metrics.

```python
# Only evaluate rules on spans they actually classified
keep = [i for i, y in enumerate(y_rule) if y >= 0]
```

This gives an honest comparison: rules are fast and precise on known patterns; supervised learning covers the long tail.

---

## Output Format

### Excel design decisions

- **RTL layout**: `ws.sheet_view.rightToLeft = True` — correct for Arabic text
- **Alternating row colors**: light blue / white for readability
- **Column widths**: pre-tuned for Arabic text (wider for text columns, narrow for percentages)
- **Header**: dark blue / white font for visibility

### CSV encoding

`encoding='utf-8-sig'` adds a BOM (Byte Order Mark) — this makes the CSV open correctly in Excel on Windows without Arabic characters appearing as mojibake.
