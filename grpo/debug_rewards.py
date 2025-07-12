import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the actual functions from grpo_utils
try:
    import grpo_utils
    print("Successfully imported grpo_utils")
    print(f"Available functions: {[f for f in dir(grpo_utils) if not f.startswith('_')]}")
    
    # Test the format reward function
    test_responses = [
        "This is a simple response.",
        "Let me think about this step by step. First, I need to consider...",
        "The answer is 42.",
        "We have a sequence of 50, 400, and 750. Let k = 50.",
        "Therefore, the solution is x = 5.",
        "I would need to find out which characters I can put in the environment variable."
    ]
    
    print("\nTesting calculate_format_reward function:")
    for i, response in enumerate(test_responses):
        reward = grpo_utils.calculate_format_reward(response)
        print(f"Response {i+1}: {reward:.3f}")
        print(f"  Text: {response[:50]}...")
    
    # Test the calculate_rewards function
    print("\nTesting calculate_rewards function:")
    validation_objects = [
        {"expected_answer": "42", "validate_function": lambda x, y: "42" in x.lower()},
        {"expected_answer": "50", "validate_function": lambda x, y: "50" in x.lower()},
        {"expected_answer": "solution", "validate_function": lambda x, y: "solution" in x.lower()},
        {"expected_answer": "answer", "validate_function": lambda x, y: "answer" in x.lower()},
        {"expected_answer": "think", "validate_function": lambda x, y: "think" in x.lower()},
        {"expected_answer": "environment", "validate_function": lambda x, y: "environment" in x.lower()}
    ]
    
    rewards = grpo_utils.calculate_rewards(test_responses, validation_objects)
    print(f"Final rewards: {rewards}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 