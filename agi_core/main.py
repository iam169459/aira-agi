"""
AGI Core Main Entry Point

This is the main entry point for the AGI Core system.
It provides a command-line interface and demonstration of AGI capabilities.
"""

import sys
import os
import time
import argparse
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_core.integration import AGIAgent
from agi_core import __version__, AGI_NAME


def print_header():
    """Print the AGI header"""
    print("=" * 70)
    print(f"  {AGI_NAME} v{__version__}")
    print("  Artificial General Intelligence Core")
    print("=" * 70)
    print()


def print_menu():
    """Print the main menu"""
    print("\nMain Menu:")
    print("  1. Start conversation")
    print("  2. Test reasoning engine")
    print("  3. Test memory system")
    print("  4. Test planning system")
    print("  5. Test decision making")
    print("  6. Show AGI state")
    print("  7. Learn a concept")
    print("  8. Run demo")
    print("  9. Exit")
    print()


def start_conversation(agent: AGIAgent):
    """Start an interactive conversation with the AGI"""
    print("\n" + "=" * 70)
    print("  Conversation Mode")
    print("  Type 'quit' or 'exit' to return to menu")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"{agent.name}: Goodbye!")
                break
            
            result = agent.converse(user_input)
            
            print(f"\n{agent.name}: {result['response']}")
            
            # Show additional info if verbose
            if result.get('intent'):
                print(f"  [Intent: {result['intent'].get('type', 'unknown')}, Confidence: {result['intent'].get('confidence', 0):.1f}]")
            if result.get('mood'):
                print(f"  [Mood: {result['mood'].get('emotion', 'unknown')}]")
            
        except KeyboardInterrupt:
            print(f"\n{agent.name}: Conversation ended.")
            break
        except Exception as e:
            print(f"Error: {e}")


def test_reasoning(agent: AGIAgent):
    """Test the reasoning engine"""
    print("\n" + "=" * 70)
    print("  Reasoning Engine Test")
    print("=" * 70)
    
    problems = [
        "What is the meaning of life?",
        "How do I solve a quadratic equation?",
        "Explain the concept of artificial intelligence",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?"
    ]
    
    for problem in problems:
        print(f"\nProblem: {problem}")
        chain = agent.think(problem, max_steps=5)
        
        print(f"  Conclusion: {chain.get_conclusion()}")
        print(f"  Confidence: {chain.confidence:.2f}")
        print(f"  Steps: {len(chain.steps)}")
        
        time.sleep(0.5)


def test_memory(agent: AGIAgent):
    """Test the memory system"""
    print("\n" + "=" * 70)
    print("  Memory System Test")
    print("=" * 70)
    
    # Store some information
    print("\nStoring information...")
    agent.remember("The capital of France is Paris", "fact")
    agent.remember("Artificial intelligence is the simulation of human intelligence in machines", "definition")
    agent.learn("AI", "Artificial Intelligence - the field of computer science focused on creating intelligent machines")
    
    # Recall information
    print("\nRecalling information...")
    results = agent.recall("France")
    print(f"  Found {len(results)} results about France")
    for result in results:
        print(f"    - {result}")
    
    results = agent.recall("AI")
    print(f"  Found {len(results)} results about AI")
    for result in results:
        print(f"    - {result}")
    
    # Show memory stats
    state = agent.get_state()
    print(f"\nMemory Stats:")
    print(f"  STM items: {state['memory_stats']['stm']['total_items']}")
    print(f"  LTM concepts: {state['memory_stats']['ltm']['total_concepts']}")


def test_planning(agent: AGIAgent):
    """Test the planning system"""
    print("\n" + "=" * 70)
    print("  Planning System Test")
    print("=" * 70)
    
    goals = [
        "Write a research paper on AI",
        "Learn to play the piano",
        "Build a website"
    ]
    
    for goal in goals:
        print(f"\nGoal: {goal}")
        plan = agent.plan(goal)
        
        if 'error' not in plan:
            print(f"  Plan ID: {plan['plan_id']}")
            print(f"  Number of actions: {len(plan['actions'])}")
            for i, action in enumerate(plan['actions'][:3]):  # Show first 3 actions
                print(f"    {i+1}. {action['name']}: {action['description'][:50]}...")
        else:
            print(f"  Error: {plan['error']}")


def test_decision_making(agent: AGIAgent):
    """Test the decision making system"""
    print("\n" + "=" * 70)
    print("  Decision Making Test")
    print("=" * 70)
    
    decision = agent.decide(
        "Which programming language should I learn first?",
        ["Python", "JavaScript", "Java", "C++"]
    )
    
    print(f"\nDecision: {decision['decision_id']}")
    print(f"  Chosen: {decision['chosen_option']}")
    print(f"  Reasoning: {decision['reasoning']}")


def show_state(agent: AGIAgent):
    """Show the current AGI state"""
    print("\n" + "=" * 70)
    print("  AGI State")
    print("=" * 70)
    
    state = agent.get_state()
    
    print("\nMemory System:")
    print(f"  STM: {state['memory_stats']['stm']['total_items']} items")
    print(f"  LTM: {state['memory_stats']['ltm']['total_concepts']} concepts")
    print(f"  Episodic: {state['memory_stats']['episodic']['total_experiences']} experiences")
    
    print("\nPlanning System:")
    print(f"  Total goals: {state['planning_stats']['total_goals']}")
    print(f"  Active goals: {state['planning_stats']['active_goals']}")
    
    print("\nWorld Model:")
    print(f"  Entities: {state['world_model_stats']['entities']}")
    print(f"  Relationships: {state['world_model_stats']['relationships']}")
    
    print("\nMeta Learning:")
    print(f"  Learning experiences: {state['meta_learning_stats']['total_experiences']}")
    print(f"  Success rate: {state['meta_learning_stats']['success_rate']:.1%}")
    
    if state.get('mood'):
        print("\nMood:")
        print(f"  Emotion: {state['mood']['emotion']}")
        print(f"  Valence: {state['mood']['valence']:.2f}")
        print(f"  Arousal: {state['mood']['arousal']:.2f}")


def run_demo(agent: AGIAgent):
    """Run a comprehensive demo of AGI capabilities"""
    print("\n" + "=" * 70)
    print("  AGI Core Demo")
    print("=" * 70)
    
    print("\n1. Conversation")
    print("-" * 70)
    result = agent.converse("Hello! What is artificial intelligence?")
    print(f"Q: Hello! What is artificial intelligence?")
    print(f"A: {result['response']}")
    
    print("\n2. Reasoning")
    print("-" * 70)
    chain = agent.think("If all humans are mortal and Socrates is human, is Socrates mortal?")
    print(f"Q: If all humans are mortal and Socrates is human, is Socrates mortal?")
    print(f"A: {chain.get_conclusion()}")
    print(f"  Confidence: {chain.confidence:.2f}")
    
    print("\n3. Memory")
    print("-" * 70)
    agent.learn("AGI", "Artificial General Intelligence - AI with human-like cognitive abilities")
    results = agent.recall("AGI")
    print(f"Learned: AGI = Artificial General Intelligence")
    print(f"Recall: {results[0] if results else 'No results'}")
    
    print("\n4. Planning")
    print("-" * 70)
    plan = agent.plan("Write a book")
    print(f"Goal: Write a book")
    print(f"Plan: {len(plan.get('actions', []))} steps created")
    
    print("\n5. Decision Making")
    print("-" * 70)
    decision = agent.decide("Which is better, tea or coffee?", ["Tea", "Coffee"])
    print(f"Q: Which is better, tea or coffee?")
    print(f"A: {decision['chosen_option']}")
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"{AGI_NAME} - Artificial General Intelligence Core"
    )
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    parser.add_argument('--demo', action='store_true', help='Run demo and exit')
    parser.add_argument('--conversation', action='store_true', help='Start conversation mode')
    parser.add_argument('--no-purple', action='store_true', help='Disable Purple AI integration')
    
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print_header()
        print(f"Version: {__version__}")
        print(f"AGI Name: {AGI_NAME}")
        sys.exit(0)
    
    # Initialize AGI Agent
    print_header()
    print("Initializing AGI Core...")
    
    agent = AGIAgent(use_purple=not args.no_purple)
    agent.initialize()
    
    print(f"{agent.name} ready!")
    
    # Run demo if requested
    if args.demo:
        run_demo(agent)
        sys.exit(0)
    
    # Start conversation if requested
    if args.conversation:
        start_conversation(agent)
        sys.exit(0)
    
    # Interactive menu
    while True:
        try:
            print_menu()
            choice = input("Enter your choice (1-9): ").strip()
            
            if choice == '1':
                start_conversation(agent)
            elif choice == '2':
                test_reasoning(agent)
            elif choice == '3':
                test_memory(agent)
            elif choice == '4':
                test_planning(agent)
            elif choice == '5':
                test_decision_making(agent)
            elif choice == '6':
                show_state(agent)
            elif choice == '7':
                concept = input("Enter concept name: ").strip()
                definition = input("Enter definition: ").strip()
                agent.learn(concept, definition)
                print(f"Learned: {concept}")
            elif choice == '8':
                run_demo(agent)
            elif choice == '9':
                print(f"\n{agent.name}: Shutting down...")
                agent.shutdown()
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 9.")
            
        except KeyboardInterrupt:
            print(f"\n{agent.name}: Shutting down...")
            agent.shutdown()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
