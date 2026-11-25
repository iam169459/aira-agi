import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.agent import load_model


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
    model, codec = load_model(args.checkpoint, device)
    if codec.type_name == "char":
        unknown = sorted(set(args.prompt) - set(codec.stoi))
        if unknown:
            raise SystemExit(
                "Prompt contains characters not in training vocabulary: "
                + repr("".join(unknown))
                + ". Use a prompt with characters found in the training "
                  "data, or train with --tokenizer bpe (byte-level BPE "
                  "has no unknown characters)."
            )
    ids = codec.encode(args.prompt)
    if not ids:
        raise SystemExit("Prompt produced no tokens.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.generate(idx, args.tokens, temperature=args.temperature)[0].tolist()
    print(codec.decode(output))


if __name__ == "__main__":
    main()
