import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.model import EmotionAwareGPT, fuse_legacy_state_dict


def main():
    parser = argparse.ArgumentParser()
    _root = os.path.dirname(os.path.abspath(__file__))
    _aira = os.path.join(_root, "aira_model.pt")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            _root,
            "aira_model.pt" if os.path.exists(_aira) else "tiny_model.pt",
        ),
    )
    parser.add_argument("--prompt", default="A language model")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Checkpoints are created by this project and should be treated as trusted local files.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    chars = checkpoint["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise SystemExit(
            "Prompt contains characters not in training vocabulary: "
            + repr("".join(unknown))
            + ". Use a prompt with characters found in data/code_corpus.txt."
        )

    model = EmotionAwareGPT(len(chars), checkpoint["config"])
    model.load_state_dict(fuse_legacy_state_dict(checkpoint["model"]))
    model = model.to(device).eval()
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
    output = model.generate(idx, args.tokens, temperature=args.temperature)[0].tolist()
    print("".join(itos[i] for i in output))


if __name__ == "__main__":
    main()
