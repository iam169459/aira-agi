#!/usr/bin/env python3
"""
Aira - Unified Launcher

Cross-platform interactive menu for all Aira features.
Replaces the bash-only start_voice_ai.sh so it works on Windows too.

Usage:
    python aira.py          # interactive menu
    python aira.py girl     # jump straight into girl voice conversation
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MENU = """
========================================================================
  Aira - Offline Voice AI Assistant
========================================================================

  Talk to Aira (girl persona):
    1. Voice conversation      (voice-to-voice)
    2. Text conversation       (type, she answers with voice)
    3. Voice command mode

  Classic assistant:
    4. Voice conversation
    5. Text conversation
    6. Voice command mode

  More:
    7. Persona demo
    8. List training configurations
    9. Train a model
   10. Generate text from a checkpoint
   11. Install neural voice (Piper)
   12. Run AGI core tests

    0. Exit
"""

DIRECT_COMMANDS = {
    "girl": ("voice_ai_girl.py", ["--mode", "voice"]),
    "text": ("voice_ai_girl.py", ["--mode", "text"]),
    "command": ("voice_ai_girl.py", ["--mode", "command"]),
    "voice": ("voice_ai.py", ["--mode", "voice"]),
    "demo": ("demo_girl_persona.py", ["--interactive"]),
    "test": ("tests/test_agi_core.py", []),
}


def run(script, args=()):
    cmd = [sys.executable, os.path.join(REPO_ROOT, script), *args]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        pass


def train_menu():
    choices = {
        "1": ("coding", "Coding model (base size, code corpus)"),
        "2": ("daily", "Daily conversation model (base size)"),
        "3": ("voice", "Voice model (nova: BPE + RoPE, GPU recommended)"),
        "4": ("large", "Large model (nova preset — the standard choice)"),
    }
    print("\nTrain which model? (all grow the ONE model: aira_model.pt)")
    for key, (_, label) in choices.items():
        print(f"  {key}. {label}")
    choice = input("Choice: ").strip()
    if choice not in choices:
        print("Cancelled.")
        return
    kind, label = choices[choice]
    print(f"\nGenerating and starting {label.lower()}...")
    generated = os.path.join(REPO_ROOT, f"train_{kind}.py")
    with open(generated, "w") as f:
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "train_configs.py"),
             "--generate", kind],
            stdout=f, cwd=REPO_ROOT,
        )
    run(f"train_{kind}.py")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target in DIRECT_COMMANDS:
            script, args = DIRECT_COMMANDS[target]
            run(script, args)
            return
        print(f"Unknown shortcut '{target}'. Available: {', '.join(DIRECT_COMMANDS)}")
        return

    while True:
        print(MENU)
        choice = input("Select an option: ").strip()

        if choice == "1":
            run("voice_ai_girl.py", ["--mode", "voice"])
        elif choice == "2":
            run("voice_ai_girl.py", ["--mode", "text"])
        elif choice == "3":
            run("voice_ai_girl.py", ["--mode", "command"])
        elif choice == "4":
            run("voice_ai.py", ["--mode", "voice"])
        elif choice == "5":
            run("voice_ai.py", ["--mode", "text"])
        elif choice == "6":
            run("voice_ai.py", ["--mode", "command"])
        elif choice == "7":
            run("demo_girl_persona.py", ["--all"])
        elif choice == "8":
            run("train_configs.py", ["--list"])
        elif choice == "9":
            train_menu()
        elif choice == "10":
            run("generate.py")
        elif choice == "11":
            run("scripts/setup_neural_voice.py")
        elif choice == "12":
            run("scripts/test_agi_core.py")
        elif choice == "0":
            print("Goodbye!")
            return
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
