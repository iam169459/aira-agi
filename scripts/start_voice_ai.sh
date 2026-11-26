#!/bin/bash

# Start Voice AI Assistant
# This script provides easy access to the voice AI functionality

# Run from the project root so all relative paths work
cd "$(dirname "$0")/.." || exit 1

echo "========================================================================"
echo "  Aira Voice AI Assistant"
echo "========================================================================"
echo ""
echo "Select an option:"
echo "  1. Start voice conversation (voice-to-voice)"
echo "  2. Start text conversation with voice output"
echo "  3. Start voice command mode"
echo "  4. Show training configurations"
echo "  5. Train coding model"
echo "  6. Train daily conversation model"
echo "  7. Train voice model"
echo "  8. Generate text (from existing model)"
echo "  9. Exit"
echo ""

read -p "Enter your choice (1-9): " choice

echo ""

case $choice in
    1)
        echo "Starting voice conversation mode..."
        python voice_ai.py --mode voice
        ;;
    2)
        echo "Starting text conversation with voice output..."
        python voice_ai.py --mode text
        ;;
    3)
        echo "Starting voice command mode..."
        python voice_ai.py --mode command
        ;;
    4)
        echo "Available training configurations (via the modern trainer):"
        python train_configs.py --list
        ;;
    5)
        echo "Training coding model (base size on the code corpus)..."
        echo "Grows the ONE model: aira_model.pt (auto-resumes)."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train.py --preset base --data code_corpus.txt
        ;;
    6)
        echo "Training daily conversation model (base size)..."
        echo "Grows the ONE model: aira_model.pt (auto-resumes)."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train.py --preset base
        ;;
    7)
        echo "Training voice model (nova: byte-level BPE + RoPE)..."
        echo "Needs a GPU for long runs; grows aira_model.pt."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train.py --preset nova
        ;;
    8)
        echo "Generating text from the model (prefers aira_model.pt)..."
        python generate.py
        ;;
    9)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        ;;
esac
