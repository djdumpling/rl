import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
from datasets import load_dataset

import re

FORMAT_REWARD_WEIGHT = 0.15
CORRECTNESS_REWARD_WEIGHT = 0.85

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

def calculate_logits(llm, full_responses, attention_masks):
    logits = llm(input_ids = full_responses, attention_masks = attention_masks).logits
    log_probs = torch.log_softmax(logits, dim = -1)

    selected_log_probs = torch.gather(input = log_probs, dim = 2, index = full_responses.unsqueeze(-1)).squeeze(-1)

    return selected_log_probs

def extract_answer(response):
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer is not None:
        return answer.group(1).strip()
    else:
        return ""

def calculate_format_reward(response):
    if ( "<answer>" not in response and "</answer>" not in response
       and "<think>" not in response and "</think>" not in response
    ):
        return -1
    format_reward = 0

    if "<think>" in response:
        format_reward += 0.15
    if "</think>" in response:
        format_reward += 0.15
    if "<answer>" in response and "</answer>" in response:
        return format_reward + 0.7
    else:
        return format_reward
    
def calculate_correctness_reward(response, validation_object):
    # Use the built-in validation function from the dataset
    # This avoids dependency on reasoning_gym
    extracted_answer = extract_answer(response)
    if extracted_answer:
        # Use the validate_function that's already in the validation_object
        is_correct = validation_object["validate_function"](extracted_answer, validation_object["expected_answer"])
        return 1.0 if is_correct else 0.0
    else:
        return 0.0


def calculate_rewards(batch_responses, validation_objects):
    format_rewards = np.array([calculate_format_reward(response) 
                               for response in batch_responses])
    correctness_rewards = np.array([calculate_correctness_reward(extract_answer(response), val_obj)
                                    for val_obj, response in zip(validation_objects, batch_responses)])
    
    rewards = (FORMAT_REWARD_WEIGHT * format_rewards + 
               CORRECTNESS_REWARD_WEIGHT * correctness_rewards)
    return rewards