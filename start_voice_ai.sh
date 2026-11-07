#!/bin/bash

# Start Voice AI Assistant
# This script provides easy access to the voice AI functionality

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
        echo "Available training configurations:"
        python train_configs.py --list
        ;;
    5)
        echo "Training coding model..."
        echo "This will train a model optimized for coding assistance."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train_configs.py --generate coding > train_coding.py
        python train_coding.py
        ;;
    6)
        echo "Training daily conversation model..."
        echo "This will train a model optimized for daily conversations."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train_configs.py --generate daily > train_daily.py
        python train_daily.py
        ;;
    7)
        echo "Training voice model..."
        echo "This will train a model optimized for voice interactions."
        read -p "Press Enter to start or Ctrl+C to cancel..." 
        python train_configs.py --generate voice > train_voice.py
        python train_voice.py
        ;;
    8)
        echo "Generating text from existing model..."
        python generate.py --checkpoint tiny_model.pt --prompt "Hello, how are you?"
        ;;
    9)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        ;;
esac
