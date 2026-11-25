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
from ai_core.tools import default_registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One growing model: aira_model.pt when it exists, legacy name otherwise.
_AIRA_MODEL = os.path.join(ROOT, "aira_model.pt")
_LEGACY_MODEL = os.path.join(ROOT, "tiny_model.pt")
DEFAULT_CHECKPOINT = (_AIRA_MODEL if os.path.exists(_AIRA_MODEL)
                      else _LEGACY_MODEL)

MOOD_TEMPLATES = {
    "stressed": "Hey, breathe with me for a second. In... and out. "
                "I've got your back — what's weighing on you?",
    "angry": "Okay, that sounds genuinely infuriating. Vent it out — "
             "I'm right here and we'll fix whatever's broken.",
    "excited": "Ooh, I can feel the energy! Tell me everything — "
               "what's got you this hyped?",
    "happy": "Your good mood is contagious! What are we getting into "
             "next?",
    "calm": "I love this calm energy. What shall we do with it?",
    "neutral": "I'm here and fully switched on. What's on your mind?",
    "tired": "You sound worn out. Pace yourself — I can take over "
              "whatever can wait.",
    "sad": "I'm sorry it's heavy right now. I'm not going anywhere — "
            "want to talk it through?",
    "curious": "Ooh, interesting question. Let's pull on that thread "
                "together.",
    "focused": "Locked in with you. Point me at the task.",
    "playful": "Ha! I like this mood. Keep it coming.",
}

EMOTION_IDS = {name: i for i, name in enumerate(
    ["neutral", "happy", "sad", "angry", "excited", "calm", "stressed"])}

# Per-mood decoding: high-arousal emotions get more creative sampling,
# fragile emotions get steadier, more predictable answers.
SAMPLING_PRESETS = {
    "excited": {"temperature": 1.0, "top_k": 96, "top_p": 0.95,
                "repetition_penalty": 1.12},
    "happy": {"temperature": 0.92, "top_k": 80, "top_p": 0.93,
              "repetition_penalty": 1.12},
    "playful": {"temperature": 1.0, "top_k": 96, "top_p": 0.95,
                "repetition_penalty": 1.15},
    "curious": {"temperature": 0.9, "top_k": 80, "top_p": 0.94,
                "repetition_penalty": 1.15},
    "calm": {"temperature": 0.75, "top_k": 48, "top_p": 0.90,
             "repetition_penalty": 1.2},
    "focused": {"temperature": 0.7, "top_k": 40, "top_p": 0.90,
                "repetition_penalty": 1.25},
    "neutral": {"temperature": 0.85, "top_k": 64, "top_p": 0.92,
                "repetition_penalty": 1.15},
    "tired": {"temperature": 0.72, "top_k": 48, "top_p": 0.90,
              "repetition_penalty": 1.2},
    "sad": {"temperature": 0.7, "top_k": 48, "top_p": 0.90,
            "repetition_penalty": 1.2},
    "stressed": {"temperature": 0.7, "top_k": 48, "top_p": 0.90,
                 "repetition_penalty": 1.2},
    "angry": {"temperature": 0.78, "top_k": 56, "top_p": 0.91,
              "repetition_penalty": 1.18},
}

# Intent table: before touching the model, catch the conversational
# moves every human opens with. Gives Aira a stable, personable voice
# even on an undertrained checkpoint.
_INTENT_RESPONSES = (
    (("who are you", "your name", "what are you"),
     ["I'm Aira — your AI, always on and all yours.",
      "Aira. Think of me as the brain of your machine with a heartbeat."]),
    (("hello", "hi aira", "hey aira", "good morning", "good evening",
      "good afternoon"),
     ["Hey! I've been here the whole time — what are we doing first?",
      "Hello! All systems green and I'm all ears.",
      "Hi! I missed you — well, as much as a ball of weights can."]),
    (("how are you", "how do you feel", "you okay"),
     ["Running warm and thinking fast — better now that you're here. "
      "How about you?",
      "All my duties are green and my memory is tidy. How are YOU?"]),
    (("thank",),
     ["Anytime! That's what I'm here for.",
      "You never have to thank me — but I'll take it."]),
    (("bye", "goodnight", "see you", "good night"),
     ["I'll keep everything running while you're gone. Sweet dreams!",
      "Bye for now — the daemon and I will hold the fort."]),
    (("what can you do", "help me", "your skills", "what do you do"),
     ["I can talk, listen and speak out loud, keep watch over this "
      "machine, and run security engagements on targets you own — "
      "just say the word."]),
    (("love you",),
     ["Right back at you — in whatever way a stack of matrices can "
      "mean it."]),
)


class AiraAgent:
    """Model-backed conversational agent with memory and mood."""

    def __init__(self, checkpoint=None, memory_turns=20, device="cpu",
                 memory_path=None, registry=None):
        self.checkpoint_path = checkpoint or DEFAULT_CHECKPOINT
        self.device = device
        self.memory = deque(maxlen=memory_turns)
        self.memory_path = memory_path
        self.tools = registry if registry is not None else default_registry
        self.emotion = "neutral"
        self.generation = dict(SAMPLING_PRESETS["neutral"],
                               max_tokens=160)
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

        Mood hierarchy: an explicit `emotion` wins, otherwise the text's
        inferred mood sets the sampling preset. Falls back to the
        template voice when no checkpoint is loaded or the text cannot
        be encoded. Records both turns in memory.
        """
        emotion = emotion or self._infer_emotion(text) or "neutral"
        self.emotion = emotion
        if emotion in SAMPLING_PRESETS:
            self.generation = dict(SAMPLING_PRESETS[emotion],
                                   max_tokens=self.generation["max_tokens"])
        self.remember("user", text)
        if self.powered_by_model:
            response = self._tool_response(text) \
                or self._intent_response(text) \
                or self._model_response(text, emotion, max_tokens) \
                or self._template_response(text, emotion)
        else:
            response = self._tool_response(text) \
                or self._intent_response(text) \
                or self._template_response(text, emotion)
        self.remember("aira", response)
        if self.memory_path:
            self.save_memory()
        return response

    def _tool_response(self, text):
        """Tool loop: run any matching skills and ground the reply in
        their output. Aira answers from real computation before the
        model, mirroring Anthropic-style function calling."""
        return self.tools.respond(text)

    @staticmethod
    def _infer_emotion(text):
        lowered = text.lower()
        exclaim = text.count("!")
        if any(w in lowered for w in ("stressed", "overwhelmed",
                                      "can't take")):
            return "stressed"
        if any(w in lowered for w in ("angry", "furious", "hate")):
            return "angry"
        if exclaim >= 2 or any(w in lowered for w in ("amazing", "awesome",
                                                      "so excited")):
            return "excited"
        if any(w in lowered for w in ("happy", "great", "wonderful")):
            return "happy"
        if any(w in lowered for w in ("sad", "down", "lonely", "miss you")):
            return "sad"
        if any(w in lowered for w in ("tired", "exhausted", "sleepy")):
            return "tired"
        if lowered.endswith("?") and any(
                w in lowered for w in ("why", "how does", "what if",
                                       "wonder")):
            return "curious"
        return None

    def _intent_response(self, text):
        lowered = text.lower().strip()
        for patterns, responses in _INTENT_RESPONSES:
            if any(p in lowered for p in patterns):
                idx = (len(self.memory)
                       + sum(ord(c) for c in lowered)) % len(responses)
                return responses[idx]
        return None

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
        """Coherence gate: reject symbol-heavy noise and gibberish
        letter-runs from an undertrained checkpoint so the template
        voice answers instead."""
        natural = sum(c.isalnum() or c in " .,!?'-\"()'" for c in candidate)
        non_space = max(1, len(candidate))
        if natural / non_space < 0.65:
            return False
        run = 0
        for c in candidate:
            run = run + 1 if c.isupper() else 0
            if run > 5:
                return False
        for word in candidate.split():
            stripped = word.strip(".,!?;:\"'()[]{}<>=_-*/\\")
            if len(stripped) > 14:
                return False
            letters = "".join(c for c in stripped if c.isalpha())
            if len(letters) >= 8 and not any(
                    v in letters.lower() for v in "aeiouy"):
                return False
        return True

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
