#!/usr/bin/env python3
"""Grow Aira's corpus, then report whether the data budget is healthy.

Drop new .txt/.md files into data/incoming/ (or pass paths / a Kaggle
dataset directory), then run:

    python scripts/grow_corpus.py                 # merge + stats report
    python scripts/grow_corpus.py --dry-run       # report only
    python scripts/grow_corpus.py /kaggle/input   # scan a specific dir

New paragraphs are deduped against the existing corpus, so re-runs are
idempotent; beast_corpus.txt is backed up to .bak before the first
write. The report prints the params-per-token ratio - THE number that
decides whether more training can help (healthy is ~20-50; Aira's
current 423KB corpus sits around 1,300, which is why val loss stalls).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import ROOT, PRESETS, grow_corpus  # noqa: E402


def estimate_params(cfg):
    """~12*L*E^2 matrix weights + embeddings (matches the exact count
    to within ~1% - good enough for a budget report)."""
    n_embd = cfg["n_embd"]
    vocab = cfg.get("vocab_size", 512) if cfg.get("tokenizer") == "bpe" \
        else 128
    return 12 * cfg["n_layer"] * n_embd * n_embd + 2 * vocab * n_embd


def report(root, chars_per_token=6.2, healthy=35):
    corpus = os.path.join(root, "data", "beast_corpus.txt")
    chars = 0
    if os.path.exists(corpus):
        with open(corpus, encoding="utf-8") as fh:
            chars = len(fh.read())
    tokens = chars / chars_per_token
    print(f"\n=== corpus report ({corpus}) ===")
    print(f"characters: {chars:,}   ~tokens (BPE est.): {tokens:,.0f}")
    if tokens < 1:
        print("corpus is empty - nothing to report")
        return
    for preset in ("nova", "titan"):
        cfg = PRESETS.get(preset)
        if not cfg:
            continue
        params = estimate_params(cfg)
        ppt = params / tokens
        need_chars = params / healthy * chars_per_token
        verdict = ("HEALTHY" if ppt <= 50 else
                   "tight" if ppt <= 200 else "DATA-STARVED")
        print(f"{preset:>6}: ~{params:,} params -> ~{ppt:,.0f} per token "
              f"[{verdict}]")
        if ppt > 50:
            print(f"         target ~{healthy} per token needs "
                  f"~{need_chars / 1e6:,.1f} MB of text "
                  f"(~{need_chars / chars:,.0f}x the current corpus)")
    print("drop more .txt/.md into data/incoming (or attach a Kaggle "
          "dataset) and re-run to grow it")


def main():
    parser = argparse.ArgumentParser(
        description="merge new text into the training corpus + stats")
    parser.add_argument("paths", nargs="*", default=None,
                        help="files/dirs to scan (default: data/incoming "
                             "and /kaggle/input when present)")
    parser.add_argument("--root", default=ROOT,
                        help="repo root containing data/ (default: this repo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what WOULD be added without writing")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="skip files smaller than this (default: 200)")
    args = parser.parse_args()
    grow_corpus(root=args.root, sources=args.paths or None,
                dry_run=args.dry_run, min_chars=args.min_chars)
    report(args.root)


if __name__ == "__main__":
    main()
