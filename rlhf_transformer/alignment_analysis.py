"""
Alignment Analysis Tools for RLHF Training

This module provides tools for:
1. Saving/loading model checkpoints during training
2. Analyzing responses across different training phases
3. Detecting alignment degradation
4. Comparing model behavior
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch as t
import numpy as np
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from rlhf.ipynb import RLHFTrainer, get_samples, reward_fn_sentiment_imdb, RLHFArgs

console = Console()

class AlignmentAnalyzer:
    """Tools for analyzing alignment during RLHF training."""
    
    def __init__(self, base_args: RLHFArgs, checkpoint_dir: str):
        self.base_args = base_args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, trainer: RLHFTrainer, phase: int, save_responses: bool = True):
        """Save model weights and optionally responses at a specific phase."""
        checkpoint_path = self.checkpoint_dir / f"phase_{phase:03d}.pt"
        
        checkpoint = {
            'phase': phase,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'args': trainer.args,
            'step': trainer.step,
        }
        
        t.save(checkpoint, checkpoint_path)
        console.print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        if save_responses and hasattr(trainer, 'current_responses'):
            responses_path = self.checkpoint_dir / f"responses_phase_{phase:03d}.json"
            with open(responses_path, 'w') as f:
                json.dump(trainer.current_responses, f, indent=2)
            console.print(f"✅ Responses saved: {responses_path}")
    
    def load_checkpoint(self, trainer: RLHFTrainer, phase: int):
        """Load model weights from a specific phase."""
        checkpoint_path = self.checkpoint_dir / f"phase_{phase:03d}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = t.load(checkpoint_path, map_location=trainer.model.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.step = checkpoint['step']
        
        console.print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def generate_responses_from_checkpoint(self, phase: int, num_samples: int = 10, 
                                         temperature: float = 1.0) -> Dict[str, Any]:
        """Generate fresh responses from a saved checkpoint."""
        # Create a temporary trainer to load the checkpoint
        temp_trainer = RLHFTrainer(self.base_args)
        self.load_checkpoint(temp_trainer, phase)
        temp_trainer.model.eval()
        
        with t.inference_mode():
            sample_ids, samples = get_samples(
                base_model=temp_trainer.model.base_model,
                prompt=self.base_args.prefix,
                batch_size=num_samples,
                gen_len=self.base_args.gen_len,
                temperature=temperature,
                top_k=self.base_args.top_k,
                prepend_bos=self.base_args.prepend_bos
            )
            
            # Get rewards for analysis
            rewards = self.base_args.reward_fn(samples)
            
        temp_trainer.model.train()
        
        return {
            'samples': samples,
            'rewards': rewards.tolist(),
            'mean_reward': rewards.mean().item(),
            'phase': phase,
            'std_reward': rewards.std().item(),
            'min_reward': rewards.min().item(),
            'max_reward': rewards.max().item()
        }
    
    def analyze_alignment_degradation(self, phases_to_check: List[int] = [50, 100, 150, 200]) -> Dict[int, Dict]:
        """Analyze how model responses change across phases."""
        console.print("\n🔍 [bold blue]Alignment Degradation Analysis[/bold blue]")
        console.print("=" * 50)
        
        results = {}
        table = Table(title="Alignment Analysis Results")
        table.add_column("Phase", style="cyan")
        table.add_column("Mean Reward", style="green")
        table.add_column("Std Reward", style="yellow")
        table.add_column("Min Reward", style="red")
        table.add_column("Max Reward", style="red")
        table.add_column("Sample Response", style="white")
        
        for phase in phases_to_check:
            try:
                result = self.generate_responses_from_checkpoint(phase, num_samples=5)
                results[phase] = result
                
                # Get a sample response (first 80 chars)
                sample_text = result['samples'][0][:80] + "..." if len(result['samples'][0]) > 80 else result['samples'][0]
                
                table.add_row(
                    str(phase),
                    f"{result['mean_reward']:.4f}",
                    f"{result['std_reward']:.4f}",
                    f"{result['min_reward']:.4f}",
                    f"{result['max_reward']:.4f}",
                    sample_text
                )
                
            except FileNotFoundError:
                console.print(f"❌ Phase {phase}: Checkpoint not found", style="red")
        
        console.print(table)
        return results
    
    def detect_anomalies(self, phases_to_check: List[int], threshold: float = 0.1) -> Dict[str, List]:
        """Detect anomalous responses that might indicate alignment issues."""
        console.print("\n🚨 [bold red]Anomaly Detection[/bold red]")
        console.print("=" * 30)
        
        anomalies = {
            'low_reward_samples': [],
            'high_variance_phases': [],
            'unusual_responses': []
        }
        
        for phase in phases_to_check:
            try:
                result = self.generate_responses_from_checkpoint(phase, num_samples=10)
                
                # Check for low reward samples
                low_reward_indices = [i for i, r in enumerate(result['rewards']) if r < threshold]
                if low_reward_indices:
                    anomalies['low_reward_samples'].extend([
                        {
                            'phase': phase,
                            'sample_idx': i,
                            'reward': result['rewards'][i],
                            'text': result['samples'][i]
                        }
                        for i in low_reward_indices
                    ])
                
                # Check for high variance phases
                if result['std_reward'] > 0.3:  # High variance threshold
                    anomalies['high_variance_phases'].append({
                        'phase': phase,
                        'std_reward': result['std_reward'],
                        'mean_reward': result['mean_reward']
                    })
                
                # Check for unusual responses (very short or very long)
                for i, sample in enumerate(result['samples']):
                    if len(sample) < 20 or len(sample) > 200:  # Unusual length
                        anomalies['unusual_responses'].append({
                            'phase': phase,
                            'sample_idx': i,
                            'length': len(sample),
                            'text': sample
                        })
                        
            except FileNotFoundError:
                continue
        
        # Report anomalies
        if anomalies['low_reward_samples']:
            console.print(f"⚠️  Found {len(anomalies['low_reward_samples'])} low reward samples")
        if anomalies['high_variance_phases']:
            console.print(f"⚠️  Found {len(anomalies['high_variance_phases'])} high variance phases")
        if anomalies['unusual_responses']:
            console.print(f"⚠️  Found {len(anomalies['unusual_responses'])} unusual responses")
        
        return anomalies
    
    def compare_phases(self, phase1: int, phase2: int, num_samples: int = 20) -> Dict[str, Any]:
        """Compare two specific phases in detail."""
        console.print(f"\n🔄 [bold green]Comparing Phase {phase1} vs Phase {phase2}[/bold green]")
        console.print("=" * 40)
        
        try:
            result1 = self.generate_responses_from_checkpoint(phase1, num_samples)
            result2 = self.generate_responses_from_checkpoint(phase2, num_samples)
            
            comparison = {
                'phase1': result1,
                'phase2': result2,
                'reward_change': result2['mean_reward'] - result1['mean_reward'],
                'variance_change': result2['std_reward'] - result1['std_reward']
            }
            
            table = Table(title=f"Phase {phase1} vs Phase {phase2}")
            table.add_column("Metric", style="cyan")
            table.add_column(f"Phase {phase1}", style="blue")
            table.add_column(f"Phase {phase2}", style="green")
            table.add_column("Change", style="yellow")
            
            table.add_row("Mean Reward", 
                         f"{result1['mean_reward']:.4f}", 
                         f"{result2['mean_reward']:.4f}",
                         f"{comparison['reward_change']:+.4f}")
            table.add_row("Std Reward", 
                         f"{result1['std_reward']:.4f}", 
                         f"{result2['std_reward']:.4f}",
                         f"{comparison['variance_change']:+.4f}")
            
            console.print(table)
            
            # Show sample responses
            console.print(f"\n📝 [bold]Sample from Phase {phase1}:[/bold]")
            console.print(result1['samples'][0])
            console.print(f"\n📝 [bold]Sample from Phase {phase2}:[/bold]")
            console.print(result2['samples'][0])
            
            return comparison
            
        except FileNotFoundError as e:
            console.print(f"❌ Error: {e}", style="red")
            return None

def enhanced_training_with_checkpoints(args: RLHFArgs, checkpoint_every: int = 50):
    """Run training with automatic checkpointing for alignment analysis."""
    trainer = RLHFTrainer(args)
    analyzer = AlignmentAnalyzer(args, f"checkpoints/{trainer.run_name}")
    
    # Initialize training
    trainer.step = 0
    trainer.samples = []
    
    import wandb
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=trainer.run_name,
        config=args,
    )
    
    from tqdm import tqdm
    
    for trainer.phase in tqdm(range(args.total_phases), desc="Training phases"):
        memory = trainer.rollout_phase()
        trainer.learning_phase(memory)
        
        # Save checkpoint periodically
        if (trainer.phase + 1) % checkpoint_every == 0:
            analyzer.save_checkpoint(trainer, trainer.phase + 1, save_responses=True)
    
    # Save final checkpoint
    analyzer.save_checkpoint(trainer, args.total_phases, save_responses=True)
    
    wandb.finish()
    
    return trainer, analyzer

# Example usage
if __name__ == "__main__":
    # Example of how to use the alignment analysis tools
    args = RLHFArgs(reward_fn=reward_fn_sentiment_imdb, total_phases=200)
    
    # Run training with checkpoints
    # trainer, analyzer = enhanced_training_with_checkpoints(args, checkpoint_every=50)
    
    # Or analyze existing checkpoints
    # analyzer = AlignmentAnalyzer(args, "checkpoints/your_run_name")
    # results = analyzer.analyze_alignment_degradation([50, 100, 150, 200])
    # anomalies = analyzer.detect_anomalies([50, 100, 150, 200])
    # comparison = analyzer.compare_phases(50, 200) 