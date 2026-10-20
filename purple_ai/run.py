"""Launch Purple AI: voice-to-voice, emotion-aware, self-healing agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purple_ai.core.brain import PurpleAgent
from purple_ai.mood_engine.mood_state import MoodState


def main():
    print("=" * 60)
    print("PURPLE AI - Self-Healing, Emotion-Aware Agent")
    print("Features: Voice-to-voice  |  Screen perception  |  Desktop automation")
    print("          Emotion awareness  |  Safe-mode guardrails  |  Self-healing")
    print("=" * 60)

    agent = PurpleAgent(safe=True)
    print(f"Mood loaded: {agent.mood.describe()}")
    print("Safe-mode guardrails enabled.")
    print("Use agent.observe_audio() for emotion input.")
    print("Use agent.self_check('path.py') for self-healing.")
    print("Use agent.respond('prompt') for adaptive answers.")


if __name__ == "__main__":
    main()
