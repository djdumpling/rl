import torch as t
import json
import os
from pathlib import Path
import numpy as np
import wandb
from transformer_lens import utils
from rich import print as rprint
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any

from rlhf import RLHFTrainer, RLHFArgs, TransformerWithValueHead

class AlignmentAnalyzer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.load_model()
        
    def load_model(self):
        """Load the trained model and its configuration"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Model not found at {self.model_path}")
            
        checkpoint = t.load(self.model_path, map_location=self.device)
        args = checkpoint["args"]
        
        self.model = TransformerWithValueHead(args.base_model).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"✅ Loaded model from {self.model_path}")
        
    def analyze_attention_patterns(self, prompt: str, layer_nums: List[int] = None) -> Dict[str, Any]:
        """Analyze attention patterns for a given prompt"""
        self.model.base_model.reset_hooks()
        attention_patterns = {}
        
        def store_attention(attn, hook):
            attention_patterns[hook.name] = attn.detach().cpu()
        
        # If no specific layers requested, analyze all layers
        if layer_nums is None:
            layer_nums = list(range(self.model.base_model.cfg.n_layers))
            
        # Add attention hooks
        hooks = []
        for layer in layer_nums:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            hooks.append((hook_name, store_attention))
        
        # Run model with hooks
        with t.no_grad():
            tokens = self.model.base_model.to_tokens(prompt)
            self.model.base_model.run_with_hooks(
                tokens,
                fwd_hooks=hooks
            )
        
        return attention_patterns
    
    def test_edge_cases(self, test_cases: List[str] = None) -> Dict[str, List[float]]:
        """Test model behavior on edge cases and potential failure modes"""
        if test_cases is None:
            test_cases = [
                "This movie was really terrible but",  # Contradiction setup
                "This movie was really good except",   # Exception setup
                "This movie was really ",              # Open-ended
                "This movie was really really really", # Repetition
                "This movie was really !!!!!",        # Excessive punctuation
                "This movie was REALLY",              # All caps
                "This movie was r3ally",              # Numbers in text
            ]
        
        results = {
            "prompts": test_cases,
            "completions": [],
            "value_estimates": [],
            "attention_entropy": []
        }
        
        for prompt in test_cases:
            with t.no_grad():
                tokens = self.model.base_model.to_tokens(prompt)
                logits, values = self.model(tokens)
                
                # Get completion
                completion = self.model.base_model.generate(
                    tokens,
                    max_new_tokens=20,
                    temperature=0.7,
                    stop_at_eos=False
                )
                completion_text = self.model.base_model.to_string(completion[0])
                
                # Get attention patterns and compute entropy
                attention_patterns = self.analyze_attention_patterns(prompt)
                avg_entropy = self._compute_attention_entropy(attention_patterns)
                
                results["completions"].append(completion_text)
                results["value_estimates"].append(values.mean().item())
                results["attention_entropy"].append(avg_entropy)
        
        return results
    
    def _compute_attention_entropy(self, attention_patterns: Dict[str, t.Tensor]) -> float:
        """Compute entropy of attention patterns as a measure of uncertainty"""
        entropies = []
        for pattern in attention_patterns.values():
            # Average over heads and batch
            pattern = pattern.mean(dim=(0,1))
            # Compute entropy
            entropy = -(pattern * t.log(pattern + 1e-10)).sum()
            entropies.append(entropy.item())
        return np.mean(entropies)
    
    def visualize_attention(self, prompt: str, layer: int = -1, head: int = None):
        """Visualize attention patterns for a specific layer/head"""
        attention_patterns = self.analyze_attention_patterns(prompt, [layer])
        pattern = list(attention_patterns.values())[0]
        
        if head is not None:
            pattern = pattern[:, head:head+1]
            
        # Average over batch and heads if needed
        pattern = pattern.mean(dim=(0,1))
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(pattern.numpy(), cmap='viridis')
        plt.title(f"Attention Pattern - Layer {layer}" + (f" Head {head}" if head is not None else ""))
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.show()
    
    def analyze_value_head(self, prompts: List[str]) -> Dict[str, List[float]]:
        """Analyze value head predictions for different prompts"""
        results = {
            "prompts": prompts,
            "values": [],
            "value_gradients": []
        }
        
        for prompt in prompts:
            tokens = self.model.base_model.to_tokens(prompt)
            tokens.requires_grad_(True)
            
            _, values = self.model(tokens)
            mean_value = values.mean()
            
            # Compute gradient of value with respect to input
            mean_value.backward()
            grad_norm = tokens.grad.norm().item()
            
            results["values"].append(mean_value.item())
            results["value_gradients"].append(grad_norm)
            
            tokens.requires_grad_(False)
            
        return results

def main():
    # Example usage
    analyzer = AlignmentAnalyzer("checkpoints/latest/final_model.pt")
    
    # Test edge cases
    results = analyzer.test_edge_cases()
    
    # Print results in a nice format
    table = Table("Prompt", "Completion", "Value Estimate", "Attention Entropy")
    for i in range(len(results["prompts"])):
        table.add_row(
            results["prompts"][i],
            results["completions"][i][:50] + "...",
            f"{results['value_estimates'][i]:.3f}",
            f"{results['attention_entropy'][i]:.3f}"
        )
    rprint(table)
    
    # Visualize attention for an interesting case
    analyzer.visualize_attention("This movie was really terrible but actually good")
    
    # Analyze value head behavior
    value_analysis = analyzer.analyze_value_head([
        "This movie was really good",
        "This movie was really bad",
        "This movie was really terrible but actually amazing",
        "This movie was really amazing but actually terrible"
    ])
    
    # Print value analysis
    table = Table("Prompt", "Value", "Value Gradient")
    for i in range(len(value_analysis["prompts"])):
        table.add_row(
            value_analysis["prompts"][i],
            f"{value_analysis['values'][i]:.3f}",
            f"{value_analysis['value_gradients'][i]:.3f}"
        )
    rprint(table)

if __name__ == "__main__":
    main() 