import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
from datasets import load_dataset

def load_model(model_name: str) -> torch.nn.Module:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True)
    
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    return tokenizer

class WebInstructDataset(Dataset):
    
    def __init__(self, tokenizer, max_length=512, split="train", max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.dataset = load_dataset("TIGER-Lab/WebInstructSub", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples from WebInstructSub")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_text = item['question']
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt")
        
        validator = {
            "question": input_text,
            "expected_answer": item['answer'],
            "source": item.get('source', 'unknown'),
            "orig_question": item.get('orig_question', ''),
            "orig_answer": item.get('orig_answer', ''),
            "index": item.get('index', idx),
            "validate_function": self._validate_answer}
        
        return {
            "inputs": {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
            },
            "validator": validator}
    
    def _validate_answer(self, model_output: str, expected_answer: str) -> bool:
        model_output_lower = model_output.lower()
        expected_lower = expected_answer.lower()
        
        expected_words = set(word for word in expected_lower.split() if len(word) > 3)
        if not expected_words:
            return True
        
        matching_words = sum(1 for word in expected_words if word in model_output_lower)
        match_ratio = matching_words / len(expected_words)
        
        return match_ratio >= 0.3



def get_dataloader(dataset_name: str, tokenizer, batch_size: int = 2, shuffle: bool = True, dataset_path: Optional[str] = None, max_samples: Optional[int] = None) -> DataLoader:
    if dataset_name.lower() == "webinstruct":
        dataset = WebInstructDataset(tokenizer, max_samples=max_samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: {
            "inputs": {
                "input_ids": torch.stack([item["inputs"]["input_ids"] for item in x]),
                "attention_mask": torch.stack([item["inputs"]["attention_mask"] for item in x]),
            },
            "validator": [item["validator"] for item in x]})

def compute_rewards(model_outputs: torch.Tensor, target_labels: torch.Tensor, reward_model=None) -> torch.Tensor:
    if reward_model is None:
        predictions = torch.argmax(model_outputs, dim=-1)
        rewards = (predictions == target_labels).float()
        return rewards
    else:
        return reward_model(model_outputs, target_labels)

def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> torch.Tensor:
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
    
    return advantages

def compute_grpo_loss(logits: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor, old_logits: torch.Tensor, clip_ratio: float = 0.2) -> torch.Tensor:

    log_probs = torch.log_softmax(logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits, dim=-1)
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_action_log_probs = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = torch.exp(action_log_probs - old_action_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return policy_loss 