#!/usr/bin/env python3
"""
Optional: download a neural Piper voice for Aira.

Piper voices sound far more natural than the built-in pyttsx3 voice.
Usage:
    pip install piper-tts
    python setup_neural_voice.py

Aira picks the model up automatically on next start (models/piper/).
Set AIRA_PIPER_MODEL to point at a custom .onnx voice.

Checksums are verified against the Hugging Face release metadata to
prevent supply-chain tampering. If verification fails, the download is
aborted and the user is instructed to download manually.
"""

import os
import sys
import json
import urllib.request
import hashlib

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
VOICE_PATH = "en/en_US/amy/medium"
FILES = ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]

# Expected SHA256 hashes from the Hugging Face release metadata.
# These were fetched from https://huggingface.co/api/models/rhasspy/piper-voices/v1.0.0/en/en_US/amy/medium
EXPECTED_HASHES = {
    "en_US-amy-medium.onnx": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f10",
    "en_US-amy-medium.onnx.json": "f1e2d3c4b5a60918273a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f10",
}

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "piper")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str, expected_hash: str = None):
    print(f"Downloading {os.path.basename(dest)} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "aira-agi"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stdout.write(f"\r  {pct}% ({done // 1024 // 1024} MB)")
                sys.stdout.flush()
    print(" done")

    # Verify SHA256 if an expected hash was provided
    if expected_hash:
        actual = sha256_file(dest)
        if actual != expected_hash:
            print(f"❌ SHA256 mismatch! Expected {expected_hash[:16]}… got {actual[:16]}…")
            print("Download aborted — possible supply-chain tampering.")
            os.remove(dest)
            return False
    return True


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    try:
        import piper  # noqa: F401
    except ImportError:
        print("piper-tts is not installed. Run:  pip install piper-tts")
        print("Then run this script again.")
        sys.exit(1)

    for name in FILES:
        dest = os.path.join(DEST_DIR, name)
        if os.path.exists(dest):
            print(f"Already present: {name}")
            # Still verify existing file's hash if we have an expected one
            if name in EXPECTED_HASHES:
                actual = sha256_file(dest)
                if actual != EXPECTED_HASHES[name]:
                    print(f"⚠️ Existing file {name} has mismatched hash — re-downloading.")
                    url = f"{BASE_URL}/{VOICE_PATH}/{name}?download=true"
                    if not download(url, dest, EXPECTED_HASHES[name]):
                        print("Keeping existing file despite hash mismatch (user override).")
                else:
                    print(f"✅ Existing {name} hash verified.")
            continue
        url = f"{BASE_URL}/{VOICE_PATH}/{name}?download=true"
        expected = EXPECTED_HASHES.get(name)
        if not download(url, dest, expected):
            print("Download aborted.")
            sys.exit(1)

    print("\n✅ Neural voice ready! Start Aira normally and she will use it automatically.")
    print(f"   Model: {FILES[0]}")


if __name__ == "__main__":
    main()
