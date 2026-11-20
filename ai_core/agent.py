"""AiraAgent: the thinking brain — real model generation, not templates.

Wraps the EmotionAwareGPT checkpoint (trained via train.py) into an
agent with bounded conversation memory, mood conditioning, and modern
sampling (top-k / nucleus / repetition penalty). When no checkpoint is
available it degrades gracefully to the mood-aware template voice, so
the agent always answers.
"""
import json
import os
from collections import deque

import torch

from ai_core.model import EmotionAwareGPT, fuse_legacy_state_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(ROOT, "tiny_model.pt")

MOOD_TEMPLATES = {
    "stressed": "I can hear you're feeling stressed. Take a deep breath. How can I help?",
    "angry": "I sense your frustration. Let me help you with that. What seems to be the problem?",
    "excited": "That sounds great! Tell me more about what's got you so pumped!",
    "happy": "I'm glad to hear it! What would you like to do next?",
    "calm": "I can sense you're in a calm state. How can I assist you today?",
    "neutral": "I'm listening. What's on your mind?",
    "tired": "You sound tired. Maybe you should rest. Anything I can handle first?",
    "sad": "I'm sorry you're feeling down. I'm here if you want to talk.",
}

EMOTION_IDS = {name: i for i, name in enumerate(
    ["neutral", "happy", "sad", "angry", "excited", "calm", "stressed"])}


class AiraAgent:
    """Model-backed conversational agent with memory and mood."""

    def __init__(self, checkpoint=None, memory_turns=20, device="cpu",
                 memory_path=None):
        self.checkpoint_path = checkpoint or DEFAULT_CHECKPOINT
        self.device = device
        self.memory = deque(maxlen=memory_turns)
        self.memory_path = memory_path
        self.emotion = "neutral"
        self.generation = {
            "temperature": 0.85,
            "top_k": 64,
            "top_p": 0.92,
            "repetition_penalty": 1.15,
            "max_tokens": 140,
        }
        self.model = None
        self.stoi = {}
        self.itos = {}
        self._load_checkpoint()
        self._load_memory()

    def _load_checkpoint(self):
        path = self.checkpoint_path
        if not path or not os.path.exists(path):
            return
        self.model, self.stoi, self.itos = load_model(path, self.device)

    def _load_memory(self):
        if not self.memory_path or not os.path.exists(self.memory_path):
            return
        try:
            with open(self.memory_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.memory.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    def save_memory(self):
        """Persist the conversation memory as JSONL for restarts."""
        if not self.memory_path:
            return None
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as fh:
            for turn in self.memory:
                fh.write(json.dumps(turn, ensure_ascii=False) + "\n")
        return self.memory_path

    @property
    def memory_turns(self):
        return len(self.memory)

    @property
    def powered_by_model(self):
        return self.model is not None

    def remember(self, role, text):
        """Append a turn to the bounded conversation memory."""
        self.memory.append({"role": role, "text": text})

    def recall(self, turns=None):
        """The last `turns` exchanges, oldest first."""
        if turns is None:
            return list(self.memory)
        return list(self.memory)[-int(turns):]

    def _encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            text = "".join(c for c in text if c in self.stoi)
            if not text:
                return None
        return torch.tensor([[self.stoi[c] for c in text]],
                            dtype=torch.long, device=self.device)

    def _decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def think(self, text, emotion=None, max_tokens=None):
        """Answer `text` using the model, mood-conditioned.

        Falls back to the template voice when no checkpoint is loaded or
        the text cannot be encoded. Records both turns in memory.
        """
        emotion = emotion or self.emotion
        self.emotion = emotion
        self.remember("user", text)
        if not self.powered_by_model:
            response = self._template_response(text, emotion)
        else:
            response = self._model_response(text, emotion, max_tokens)
        self.remember("aira", response)
        if self.memory_path:
            self.save_memory()
        return response

    def _model_response(self, text, emotion, max_tokens):
        opts = dict(self.generation)
        if max_tokens:
            opts["max_tokens"] = max_tokens
        context_parts = [f"{t['role']}: {t['text']}\n"
                         for t in self.recall(8)]
        context_parts.append("aira: ")
        context = "".join(context_parts)
        idx = self._encode(context)
        if idx is None or idx.size(1) >= self.model.block_size:
            context = f"user: {text}\naira: "
            idx = self._encode(context)
        if idx is None or idx.size(1) >= self.model.block_size:
            return self._template_response(text, emotion)
        emotion_id = EMOTION_IDS.get(emotion)
        budget = min(opts["max_tokens"],
                     self.model.block_size - idx.size(1) - 1)
        if budget <= 0:
            return self._template_response(text, emotion)
        out = self.model.generate(
            idx, budget,
            temperature=opts["temperature"],
            emotion_id=emotion_id,
            top_k=opts["top_k"],
            top_p=opts["top_p"],
            repetition_penalty=opts["repetition_penalty"],
        )[0].tolist()
        tail = self._decode(out[idx.size(1):])
        return self._clean_response(tail, text, emotion)

    def _clean_response(self, tail, text, emotion):
        cleaned = tail.split("\n")[0].strip()[:400]
        if len(cleaned) < 2 or not self._looks_like_text(cleaned):
            return self._template_response(text, emotion)
        return cleaned

    @staticmethod
    def _looks_like_text(candidate):
        """Coherence gate: reject symbol-heavy noise from an
        undertrained checkpoint so the template voice answers instead."""
        natural = sum(c.isalnum() or c in " .,!?'-\"()'" for c in candidate)
        non_space = max(1, len(candidate))
        return natural / non_space >= 0.65

    def _template_response(self, text, emotion):
        response = MOOD_TEMPLATES.get(emotion, MOOD_TEMPLATES["neutral"])
        lowered = text.lower()
        if "thank" in lowered:
            return "You're welcome! Anything else I can help with?"
        if any(w in lowered for w in ("help", "assist", "support")):
            if emotion in ("stressed", "angry", "sad"):
                return ("I'm here to help you. What specifically do you "
                        "need assistance with?")
            return "I'd be happy to help you with that! What do you need?"
        return response


def load_model(path, device="cpu"):
    """Load an EmotionAwareGPT checkpoint saved by train.py."""
    checkpoint = torch.load(path, map_location="cpu",
                            weights_only=False)
    chars = checkpoint["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    model = EmotionAwareGPT(len(chars), checkpoint["config"])
    state = checkpoint["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(fuse_legacy_state_dict(state))
    model = model.to(device).eval()
    return model, stoi, itos
