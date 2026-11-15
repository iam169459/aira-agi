#!/usr/bin/env python3
"""
Optional: download a neural Piper voice for Aira.

Piper voices sound far more natural than the built-in pyttsx3 voice.
Usage:
    pip install piper-tts
    python setup_neural_voice.py

Aira picks the model up automatically on next start (models/piper/).
Set AIRA_PIPER_MODEL to point at a custom .onnx voice.
"""

import os
import sys
import urllib.request

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
VOICE_PATH = "en/en_US/amy/medium"
FILES = ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "piper")


def download(url: str, dest: str):
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
            continue
        url = f"{BASE_URL}/{VOICE_PATH}/{name}?download=true"
        try:
            download(url, dest)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Check your internet connection, or download manually from:")
            print(f"  {BASE_URL}/{VOICE_PATH}/")
            sys.exit(1)

    print("\n✅ Neural voice ready! Start Aira normally and she will use it automatically.")
    print(f"   Model: {FILES[0]}")


if __name__ == "__main__":
    main()
