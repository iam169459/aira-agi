# Tiny From Scratch

A small, fully local educational language model trained **from random initialization** with PyTorch. Designed as a first experiment on a Mac with 16 GB RAM.

> This is a learning project, not a capable general-purpose assistant. With a small corpus it will generate low-quality, often incoherent text.

## Features

- Character-level tokenizer built from your dataset
- Decoder-only Transformer with causal self-attention
- Training and validation loss
- Checkpoint saved locally to `tiny_model.pt`
- Text generation via `generate.py`
- No pretrained weights or API calls

## Requirements

- macOS with Python 3.10+ (Python 3.11 or 3.12 recommended)
- PyTorch
- About 16 GB RAM recommended for the included conservative settings; actual use depends on the Mac chip and other apps

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check whether Apple GPU acceleration is available:

```bash
python -c "import platform, torch; print('Architecture:', platform.machine()); print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

The scripts use Apple's MPS backend when available; otherwise they use CPU.

## Dataset

`data.txt` is a tiny demo corpus so the project can run immediately. It is only a smoke test. Replace it with a larger corpus of text you have permission to use. For meaningful experiments, start with at least hundreds of thousands of characters and ideally several megabytes of clean, diverse text.

- Remove duplicates and corrupted text.
- Do not include private or sensitive information.
- Check the dataset license and terms.
- This demo splits the text into a training segment and a validation segment. For serious evaluation, prepare a separate validation file to avoid accidental leakage.

## Train

```bash
python train.py
```

This initializes weights randomly, trains the model, prints loss periodically, and saves `tiny_model.pt`.

## Generate

```bash
python generate.py --prompt "A language model" --tokens 300
```

The prompt must use characters present in the training vocabulary. If you change `data.txt`, retrain before generating.

## Tuning for memory

Edit the constants at the top of `train.py`:

- If memory pressure is high, reduce `BATCH_SIZE` from 8 to 4.
- Reduce `BLOCK_SIZE` from 64 to 32 if needed.
- Increase model dimensions only after the small configuration works.

The included model is roughly in the low-million-parameter range depending on vocabulary size. Parameter count is printed at startup. Training quality depends much more on dataset size, data quality, and compute than on parameter count alone.

## Offline use

Install Python packages before disconnecting from the internet. After installation, training and generation use local files only and do not make network requests.

## Project structure

```text
tiny-from-scratch/
├── data.txt
├── train.py
├── generate.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

The code is licensed under the MIT License. Dataset contents retain their own licenses and terms.
