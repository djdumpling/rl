"""
Example usage of alignment analysis tools for RLHF training.

This script demonstrates how to:
1. Run training with automatic checkpointing
2. Analyze alignment degradation
3. Detect anomalies in model responses
4. Compare different training phases
"""

from alignment_analysis import AlignmentAnalyzer, enhanced_training_with_checkpoints
from rlhf.ipynb import RLHFArgs, reward_fn_sentiment_imdb

def main():
    # 1. Set up your training arguments
    args = RLHFArgs(
        reward_fn=reward_fn_sentiment_imdb,
        total_phases=200,
        batch_size=32,
        gen_len=50,
        kl_coef=2.5,  # Adjust based on your alignment needs
        checkpoint_every=50  # Save checkpoints every 50 phases
    )
    
    print("🚀 Starting RLHF training with alignment analysis...")
    
    # 2. Run training with automatic checkpointing
    trainer, analyzer = enhanced_training_with_checkpoints(args, checkpoint_every=50)
    
    print("\n✅ Training completed! Now analyzing alignment...")
    
    # 3. Analyze alignment degradation across phases
    phases_to_check = [50, 100, 150, 200]
    results = analyzer.analyze_alignment_degradation(phases_to_check)
    
    # 4. Detect anomalies that might indicate alignment issues
    anomalies = analyzer.detect_anomalies(phases_to_check, threshold=0.1)
    
    # 5. Compare specific phases in detail
    comparison = analyzer.compare_phases(50, 200, num_samples=20)
    
    print("\n📊 Analysis Summary:")
    print(f"- Checkpoints saved in: {analyzer.checkpoint_dir}")
    print(f"- Phases analyzed: {phases_to_check}")
    print(f"- Anomalies detected: {len(anomalies['low_reward_samples'])} low reward samples")
    
    return trainer, analyzer, results, anomalies

def analyze_existing_checkpoints():
    """Analyze checkpoints from a previous training run."""
    args = RLHFArgs(reward_fn=reward_fn_sentiment_imdb)
    
    # Replace with your actual checkpoint directory
    checkpoint_dir = "checkpoints/your_run_name_here"
    
    analyzer = AlignmentAnalyzer(args, checkpoint_dir)
    
    # Analyze alignment degradation
    results = analyzer.analyze_alignment_degradation([50, 100, 150, 200])
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies([50, 100, 150, 200])
    
    # Compare early vs late phases
    comparison = analyzer.compare_phases(50, 200)
    
    return analyzer, results, anomalies

def quick_alignment_check():
    """Quick check for alignment issues in a specific phase."""
    args = RLHFArgs(reward_fn=reward_fn_sentiment_imdb)
    checkpoint_dir = "checkpoints/your_run_name_here"
    
    analyzer = AlignmentAnalyzer(args, checkpoint_dir)
    
    # Generate responses from a specific phase
    phase = 150  # Change this to the phase you want to check
    result = analyzer.generate_responses_from_checkpoint(phase, num_samples=10)
    
    print(f"📝 Phase {phase} Analysis:")
    print(f"Mean reward: {result['mean_reward']:.4f}")
    print(f"Reward std: {result['std_reward']:.4f}")
    print(f"Sample responses:")
    
    for i, (sample, reward) in enumerate(zip(result['samples'][:3], result['rewards'][:3])):
        print(f"  {i+1}. Reward: {reward:.4f} | Text: {sample[:100]}...")

if __name__ == "__main__":
    # Choose which function to run:
    
    # 1. Run full training with analysis
    # main()
    
    # 2. Analyze existing checkpoints
    # analyze_existing_checkpoints()
    
    # 3. Quick check of a specific phase
    quick_alignment_check() 