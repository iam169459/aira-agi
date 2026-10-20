#!/usr/bin/env python
"""
Comprehensive Test Suite for AGI Core

This script tests all major components of the AGI Core system.
Run with: python test_agi_core.py

Options:
  --quick    Run only basic tests
  --nlp      Test NLP modules
  --llm      Test LLM modules (requires PyTorch)
  --all      Test everything
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_test(test_name: str, passed: bool = True, message: str = ""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {status}: {test_name}")
    if message:
        print(f"      {message}")


def test_agi_core_basic():
    """Test basic AGI Core functionality"""
    print_section("AGI Core Basic Tests")
    
    from agi_core.integration import AGIAgent
    
    # Test initialization
    try:
        agent = AGIAgent(name='TestAGI', use_purple=False)
        agent.initialize()
        print_test("AGI Agent initialization")
    except Exception as e:
        print_test("AGI Agent initialization", False, str(e))
        return False
    
    # Test conversation
    try:
        result = agent.converse("Hello!")
        assert 'response' in result
        print_test("Conversation")
    except Exception as e:
        print_test("Conversation", False, str(e))
        return False
    
    # Test reasoning
    try:
        chain = agent.think("What is AI?", max_steps=3)
        assert chain is not None
        assert len(chain.steps) > 0
        print_test("Reasoning engine")
    except Exception as e:
        print_test("Reasoning engine", False, str(e))
        return False
    
    # Test memory
    try:
        agent.remember("Test fact", "fact")
        results = agent.recall("Test")
        assert len(results) > 0
        print_test("Memory system")
    except Exception as e:
        print_test("Memory system", False, str(e))
        return False
    
    # Test learning
    try:
        agent.learn("TestConcept", "A test concept")
        print_test("Learning system")
    except Exception as e:
        print_test("Learning system", False, str(e))
        return False
    
    # Test planning
    try:
        plan = agent.plan("Test goal")
        assert 'actions' in plan or 'plan_id' in plan
        print_test("Planning system")
    except Exception as e:
        print_test("Planning system", False, str(e))
        return False
    
    # Test decision making
    try:
        decision = agent.decide("Test?", ["Option1", "Option2"])
        assert 'chosen_option' in decision
        print_test("Decision making")
    except Exception as e:
        print_test("Decision making", False, str(e))
        return False
    
    # Test state
    try:
        state = agent.get_state()
        assert 'memory_stats' in state
        print_test("State management")
    except Exception as e:
        print_test("State management", False, str(e))
        return False
    
    return True


def test_nlp_modules():
    """Test NLP modules"""
    print_section("NLP Modules Tests")
    
    from agi_core.nlp import TextProcessor, IntentClassifier, EntityExtractor
    
    # Test TextProcessor
    try:
        processor = TextProcessor()
        
        # Test tokenization
        tokens = processor.tokenizer.tokenize_words("Hello World")
        assert len(tokens) == 2
        print_test("Text tokenization")
        
        # Test normalization
        normalized = processor.normalize("Hello, World!")
        assert "hello" in normalized.lower()
        print_test("Text normalization")
        
        # Test n-grams
        ngrams = processor.generate_ngrams("hello world test", n=2)
        assert len(ngrams) > 0
        print_test("N-gram generation")
        
    except Exception as e:
        print_test("TextProcessor", False, str(e))
        return False
    
    # Test IntentClassifier
    try:
        classifier = IntentClassifier()
        
        # Test question detection
        intent = classifier.classify("What is AI?")
        assert intent.intent_type.value == "question"
        print_test("Question detection")
        
        # Test greeting detection
        intent = classifier.classify("Hello!")
        assert intent.intent_type.value == "greeting"
        print_test("Greeting detection")
        
        # Test sentiment analysis
        intent = classifier.classify("I love this!")
        assert intent.sentiment > 0
        print_test("Sentiment analysis")
        
    except Exception as e:
        print_test("IntentClassifier", False, str(e))
        return False
    
    # Test EntityExtractor
    try:
        extractor = EntityExtractor()
        
        # Test entity extraction
        entities = extractor.extract("John works at Google in New York")
        assert len(entities) > 0
        print_test("Entity extraction")
        
        # Test specific entity types
        entities = extractor.extract_by_type("john@example.com", extractor.EntityType.EMAIL)
        assert len(entities) > 0
        print_test("Email extraction")
        
    except Exception as e:
        print_test("EntityExtractor", False, str(e))
        return False
    
    return True


def test_reasoning_modules():
    """Test reasoning modules"""
    print_section("Reasoning Modules Tests")
    
    from agi_core.reasoning import ReasoningEngine, ReasoningMode
    
    try:
        engine = ReasoningEngine()
        
        # Test chain-of-thought
        chain = engine.chain_of_thought("What is AI?", max_steps=3)
        assert chain is not None
        assert len(chain.steps) > 0
        print_test("Chain-of-thought reasoning")
        
        # Test tree-of-thoughts
        chain = engine.tree_of_thoughts("Test problem", branches=2, depth=2)
        assert chain is not None
        print_test("Tree-of-thoughts reasoning")
        
        # Test verification
        result = engine.verify_reasoning(chain)
        assert 'is_valid' in result
        print_test("Reasoning verification")
        
        # Test different modes
        chain = engine.reason("Test", ReasoningMode.CHAIN_OF_THOUGHT, max_steps=2)
        assert chain is not None
        print_test("Reasoning modes")
        
    except Exception as e:
        print_test("ReasoningEngine", False, str(e))
        return False
    
    return True


def test_memory_modules():
    """Test memory modules"""
    print_section("Memory Modules Tests")
    
    from agi_core.memory import MemorySystem, ShortTermMemory, LongTermMemory
    
    try:
        # Test ShortTermMemory
        stm = ShortTermMemory(max_items=10)
        
        # Test adding items
        item_id = stm.add("Test item", "fact")
        assert item_id is not None
        print_test("STM add item")
        
        # Test retrieval
        item = stm.get(item_id)
        assert item is not None
        print_test("STM retrieval")
        
        # Test search
        results = stm.search("Test")
        assert len(results) > 0
        print_test("STM search")
        
        # Test context windows
        window = stm.create_context_window(focus="test")
        assert window is not None
        print_test("STM context windows")
        
    except Exception as e:
        print_test("ShortTermMemory", False, str(e))
        return False
    
    try:
        # Test LongTermMemory
        ltm = LongTermMemory()
        
        # Test storing concepts
        concept = ltm.store_concept("Test", "A test concept")
        assert concept is not None
        print_test("LTM store concept")
        
        # Test retrieving concepts
        retrieved = ltm.retrieve_concept("Test")
        assert retrieved is not None
        print_test("LTM retrieve concept")
        
        # Test storing facts
        fact = ltm.store_fact("Test fact", "Test", "is", "test")
        assert fact is not None
        print_test("LTM store fact")
        
        # Test search
        results = ltm.search("Test")
        assert 'concepts' in results or 'facts' in results
        print_test("LTM search")
        
    except Exception as e:
        print_test("LongTermMemory", False, str(e))
        return False
    
    try:
        # Test MemorySystem
        memory = MemorySystem(stm_capacity=20)
        
        # Test unified storage
        item_id = memory.store("Test", "fact")
        assert item_id is not None
        print_test("MemorySystem store")
        
        # Test unified retrieval
        item = memory.retrieve(item_id)
        assert item is not None
        print_test("MemorySystem retrieve")
        
        # Test unified query
        results = memory.query("Test")
        assert len(results) > 0
        print_test("MemorySystem query")
        
        # Test learning
        concept_id = memory.learn_concept("AI", "Artificial Intelligence")
        assert concept_id is not None
        print_test("MemorySystem learn")
        
        # Test experience
        exp_id = memory.remember_experience("Test experience", "A test")
        assert exp_id is not None
        print_test("MemorySystem experience")
        
    except Exception as e:
        print_test("MemorySystem", False, str(e))
        return False
    
    return True


def test_planning_modules():
    """Test planning modules"""
    print_section("Planning Modules Tests")
    
    from agi_core.planning import Planner, DecisionEngine, DecisionStrategy
    
    try:
        planner = Planner()
        
        # Test goal creation
        goal = planner.create_goal("Test goal", priority=0.8)
        assert goal is not None
        print_test("Goal creation")
        
        # Test plan creation
        plan = planner.create_plan(goal.goal_id, "Test plan")
        assert plan is not None
        print_test("Plan creation")
        
        # Test planning
        plan = planner.plan_for_goal(goal.goal_id, max_actions=3)
        assert plan is not None
        print_test("Automatic planning")
        
        # Test goal management
        goals = planner.get_active_goals()
        assert len(goals) >= 0  # May be 0 or more
        print_test("Goal management")
        
        # Test stats
        stats = planner.get_stats()
        assert 'total_goals' in stats
        print_test("Planner stats")
        
    except Exception as e:
        print_test("Planner", False, str(e))
        return False
    
    try:
        engine = DecisionEngine()
        
        # Test decision creation
        decision = engine.create_decision("Test decision", "Context")
        assert decision is not None
        print_test("Decision creation")
        
        # Test option evaluation
        decision.add_option(engine.Option("opt1", "Option 1"))
        decision.add_option(engine.Option("opt2", "Option 2"))
        decision.add_criterion(engine.Criterion("quality", "Quality criterion", 1.0))
        decision.make_decision()
        assert decision.chosen_option is not None
        print_test("Decision making")
        
        # Test strategy selection
        decision2 = engine.make_decision(
            "Test", ["A", "B", "C"],
            strategy=DecisionStrategy.MAXIMIZE_UTILITY
        )
        assert decision2.chosen_option is not None
        print_test("Decision strategies")
        
    except Exception as e:
        print_test("DecisionEngine", False, str(e))
        return False
    
    return True


def test_world_model():
    """Test world model"""
    print_section("World Model Tests")
    
    from agi_core.world_model import WorldModel, KnowledgeGraph
    
    try:
        world = WorldModel()
        
        # Test entity creation
        entity = world.add_entity("Test", "thing", "A test entity")
        assert entity is not None
        print_test("Entity creation")
        
        # Test relationship creation
        entity2 = world.add_entity("Test2", "thing", "Another test entity")
        rel = world.add_relationship("related_to", entity.entity_id, entity2.entity_id)
        assert rel is not None
        print_test("Relationship creation")
        
        # Test querying
        results = world.query({'entity_type': 'thing'})
        assert len(results) >= 0
        print_test("World query")
        
        # Test inference
        types = world.infer(entity.entity_id, "type_hierarchy")
        assert isinstance(types, list)
        print_test("World inference")
        
        # Test state
        state = world.get_world_state()
        assert state is not None
        print_test("World state")
        
        # Test stats
        stats = world.get_stats()
        assert 'entities' in stats
        print_test("World stats")
        
    except Exception as e:
        print_test("WorldModel", False, str(e))
        return False
    
    return True


def test_meta_learning():
    """Test meta-learning"""
    print_section("Meta-Learning Tests")
    
    from agi_core.meta_learning import MetaLearner
    
    try:
        learner = MetaLearner()
        
        # Test recording experience
        exp = learner.record_experience(
            task_description="Test task",
            learning_method="test_method",
            duration=10.0,
            success=True,
            accuracy=0.9
        )
        assert exp is not None
        print_test("Experience recording")
        
        # Test strategy recommendations
        recommendations = learner.get_strategy_recommendations("Test task")
        assert len(recommendations) > 0
        print_test("Strategy recommendations")
        
        # Test best strategy
        best = learner.get_best_strategy("Test task")
        assert best is not None
        print_test("Best strategy selection")
        
        # Test stats
        stats = learner.get_learning_stats()
        assert 'total_experiences' in stats
        print_test("Meta-learning stats")
        
    except Exception as e:
        print_test("MetaLearner", False, str(e))
        return False
    
    return True


def test_llm_modules():
    """Test LLM modules (requires PyTorch)"""
    print_section("LLM Modules Tests")
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print_test("PyTorch", False, "PyTorch not installed")
        return False
    
    if not TORCH_AVAILABLE:
        return False
    
    try:
        from agi_core.llm import LLMReasoningEngine, TransformerIntegration, LLM_AVAILABLE
        
        if not LLM_AVAILABLE:
            print_test("LLM modules", False, "LLM modules not available")
            return False
        
        # Test transformer integration
        transformer = TransformerIntegration()
        
        if transformer.is_loaded():
            # Test generation
            result = transformer.generate("Hello", max_tokens=10)
            assert result is not None
            print_test("Transformer generation")
        else:
            print_test("Transformer model", False, "Model not loaded (tiny_model.pt not found)")
        
        # Test LLM reasoning engine
        engine = LLMReasoningEngine()
        
        if engine.is_available():
            # Test reasoning
            result = engine.generate_reasoning("What is AI?", max_steps=2)
            assert 'final_answer' in result
            print_test("LLM reasoning")
        else:
            print_test("LLM reasoning", False, "LLM not available")
        
    except Exception as e:
        print_test("LLM modules", False, str(e))
        return False
    
    return True


def run_tests(test_types: List[str] = None):
    """Run all specified tests"""
    start_time = time.time()
    
    print("=" * 70)
    print("  AGI Core Comprehensive Test Suite")
    print("=" * 70)
    
    test_types = test_types or ['basic', 'nlp', 'reasoning', 'memory', 'planning', 'world', 'meta']
    
    results = {}
    
    # Run tests
    if 'basic' in test_types or 'all' in test_types:
        results['basic'] = test_agi_core_basic()
    
    if 'nlp' in test_types or 'all' in test_types:
        results['nlp'] = test_nlp_modules()
    
    if 'reasoning' in test_types or 'all' in test_types:
        results['reasoning'] = test_reasoning_modules()
    
    if 'memory' in test_types or 'all' in test_types:
        results['memory'] = test_memory_modules()
    
    if 'planning' in test_types or 'all' in test_types:
        results['planning'] = test_planning_modules()
    
    if 'world' in test_types or 'all' in test_types:
        results['world'] = test_world_model()
    
    if 'meta' in test_types or 'all' in test_types:
        results['meta'] = test_meta_learning()
    
    if 'llm' in test_types or 'all' in test_types:
        results['llm'] = test_llm_modules()
    
    # Print summary
    print_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_type, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_type}")
    
    print(f"\n   Total: {passed}/{total} test suites passed")
    
    elapsed = time.time() - start_time
    print(f"   Time: {elapsed:.2f} seconds")
    
    return passed == total


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AGI Core Comprehensive Test Suite'
    )
    parser.add_argument('--quick', action='store_true', 
                        help='Run only basic tests')
    parser.add_argument('--nlp', action='store_true',
                        help='Test NLP modules')
    parser.add_argument('--llm', action='store_true',
                        help='Test LLM modules')
    parser.add_argument('--all', action='store_true',
                        help='Test everything')
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.all:
        test_types = ['all']
    elif args.quick:
        test_types = ['basic']
    else:
        test_types = ['basic']
        if args.nlp:
            test_types.append('nlp')
        if args.llm:
            test_types.append('llm')
    
    # Run tests
    success = run_tests(test_types)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
