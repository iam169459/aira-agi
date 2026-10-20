# Purple AI

A new integrated AI agent built on top of your workspace.

## What's included

- **Voice-to-voice** (`purple_ai/action/voice_io.py`) using Whisper + pyttsx3
- **Emotion-aware responses** (`purple_ai/mood_engine/mood_state.py`) — valence/arousal/dominance with decay
- **Screen perception** (`purple_ai/perception/emotion_perception.py`) — microphone-based emotion + image capture hooks
- **Desktop automation** (`purple_ai/action/desktop_automation.py`) — mouse/keyboard control guarded by SafeMode
- **Self-healing code** (`purple_ai/self_healing/self_heal.py`) — syntax diagnosis + minimal repair
- **Safe-mode guardrails** (`purple_ai/safe_mode/guardrails.py`) — blocks destructive actions and dangerous paths
- **Adaptive tone** — response style changes based on current mood/emotion state
- **Mood persistence** across sessions (saved to `mood.json`)

## Quick start

```bash
python purple_ai/run.py
```

From Python:

```python
from purple_ai.core.brain import PurpleAgent
agent = PurpleAgent(safe=True)
agent.observe_audio("audio.wav")
print(agent.mood.describe())
agent.self_check("some_file.py")
```

## Architecture

```
purple_ai/
├── core/brain.py          # Agent loop (mood + self-healing + guardrails)
├── mood_engine/           # Persistent valence/arousal/dominance
├── perception/            # Emotion detection + screen hooks
├── action/                # Desktop automation + voice I/O
├── safe_mode/             # Guardrails
├── self_healing/          # Auto-diagnose / repair
└── run.py                 # Entry point
```

## Safety notes

- Safe-mode is ON by default (`safe=True`).
- All desktop actions pass through `SafeModeGuard.allow_desktop_action()`.
- Code repair is limited to syntax / import injection only — no dangerous mutations.
- Voice and screen modules are wrappers around your existing `modules/` assets.
