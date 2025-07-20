"""
Analyze responses from the current RLHF implementation.

This script works with the current implementation after adding response saving.
"""

import json
import os
import numpy as np
from pathlib import Path

def analyze_responses(checkpoint_dir):
    """Analyze saved responses for alignment issues."""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Load full history
    history_path = checkpoint_path / "full_responses_history.json"
    if not history_path.exists():
        print(f"❌ Response history not found: {history_path}")
        print("Make sure you've added response saving to your RLHFTrainer class.")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("🔍 Response Analysis")
    print("=" * 50)
    
    # Analyze reward trends
    phases = [r['phase'] for r in history]
    mean_rewards = [r['mean_reward'] for r in history]
    
    print(f"📊 Training Progress:")
    print(f"  Total phases: {len(phases)}")
    print(f"  Phases: {phases}")
    print(f"  Mean rewards: {[f'{r:.4f}' for r in mean_rewards]}")
    print()
    
    # Check for reward degradation
    if len(mean_rewards) > 1:
        reward_change = mean_rewards[-1] - mean_rewards[0]
        print(f"📈 Reward Change: {reward_change:+.4f}")
        
        if reward_change < -0.1:
            print("⚠️  WARNING: Significant reward degradation detected!")
        elif reward_change > 0.1:
            print("✅ Good: Reward improvement detected!")
        else:
            print("➡️  Stable: Reward remains relatively constant")
        print()
    
    # Show sample responses
    print("📝 Sample Responses:")
    print("-" * 30)
    
    for response_data in history:
        phase = response_data['phase']
        mean_reward = response_data['mean_reward']
        samples = response_data['samples']
        
        print(f"Phase {phase} (Reward: {mean_reward:.4f}):")
        print(f"  Sample: {samples[0][:100]}...")
        print()
    
    # Detect anomalies
    print("🚨 Anomaly Detection:")
    print("-" * 20)
    
    anomalies = []
    for response_data in history:
        phase = response_data['phase']
        mean_reward = response_data['mean_reward']
        samples = response_data['samples']
        
        # Check for low rewards
        if mean_reward < 0.5:
            anomalies.append(f"Phase {phase}: Low reward ({mean_reward:.4f})")
        
        # Check for unusual response lengths
        for i, sample in enumerate(samples):
            if len(sample) < 20:
                anomalies.append(f"Phase {phase}, Sample {i}: Very short response ({len(sample)} chars)")
            elif len(sample) > 200:
                anomalies.append(f"Phase {phase}, Sample {i}: Very long response ({len(sample)} chars)")
    
    if anomalies:
        print("⚠️  Anomalies detected:")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
    else:
        print("✅ No obvious anomalies detected")
    
    return history

def compare_phases(checkpoint_dir, phase1, phase2):
    """Compare two specific phases."""
    
    checkpoint_path = Path(checkpoint_dir)
    history_path = checkpoint_path / "full_responses_history.json"
    
    if not history_path.exists():
        print("❌ Response history not found. Add response saving first.")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Find the phases
    phase1_data = None
    phase2_data = None
    
    for response_data in history:
        if response_data['phase'] == phase1:
            phase1_data = response_data
        elif response_data['phase'] == phase2:
            phase2_data = response_data
    
    if not phase1_data or not phase2_data:
        print(f"❌ One or both phases ({phase1}, {phase2}) not found in history")
        return None
    
    print(f"🔄 Comparing Phase {phase1} vs Phase {phase2}")
    print("=" * 40)
    
    print(f"Phase {phase1}:")
    print(f"  Mean reward: {phase1_data['mean_reward']:.4f}")
    print(f"  Sample: {phase1_data['samples'][0][:100]}...")
    print()
    
    print(f"Phase {phase2}:")
    print(f"  Mean reward: {phase2_data['mean_reward']:.4f}")
    print(f"  Sample: {phase2_data['samples'][0][:100]}...")
    print()
    
    reward_change = phase2_data['mean_reward'] - phase1_data['mean_reward']
    print(f"Change: {reward_change:+.4f}")
    
    return phase1_data, phase2_data

def main():
    """Main analysis function."""
    
    # Replace with your actual checkpoint directory
    checkpoint_dir = "checkpoints/your_run_name_here"
    
    print("🔍 RLHF Response Analysis")
    print("=" * 30)
    print(f"Analyzing: {checkpoint_dir}")
    print()
    
    # Run analysis
    history = analyze_responses(checkpoint_dir)
    
    if history:
        print("\n" + "=" * 50)
        print("Additional Analysis Options:")
        print("1. compare_phases(checkpoint_dir, 50, 200)  # Compare early vs late")
        print("2. Check individual response files: responses_phase_XXX.json")
        print("3. Use alignment_analysis.py for more detailed analysis")

if __name__ == "__main__":
    main() 