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
    # regex to extract answer
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer is not None:
        return answer.group(1).strip()
    
    response_lower = response.lower()
    answer_patterns = [
        r"answer[:\s]+(.*?)(?:\.|$)",
        r"result[:\s]+(.*?)(?:\.|$)", 
        r"solution[:\s]+(.*?)(?:\.|$)",
        r"therefore[,\s]+(.*?)(?:\.|$)",
        r"thus[,\s]+(.*?)(?:\.|$)",
        r"hence[,\s]+(.*?)(?:\.|$)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_lower, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if sentences:
        return sentences[-1]
    

    return response.strip()

def calculate_format_reward(response):
    has_think_tags = "<think>" in response and "</think>" in response
    has_answer_tags = "<answer>" in response and "</answer>" in response
    
    # if specific thinking tags
    if has_think_tags or has_answer_tags:
        format_reward = 0
        if "<think>" in response:
            format_reward += 0.15
        if "</think>" in response:
            format_reward += 0.15
        if has_answer_tags:
            return format_reward + 0.7
        else:
            return format_reward
    
    response_lower = response.lower().strip()
    
    if not response_lower:
        return -1.0
    
    format_reward = 0.0
    
    reasoning_indicators = [
        "let", "assume", "suppose", "consider", "given", "therefore", "thus", "hence",
        "because", "since", "if", "then", "else", "however", "but", "although",
        "first", "second", "third", "finally", "in conclusion", "to solve",
        "we have", "we get", "we obtain", "we find", "we can", "we need",
        "the answer is", "the result is", "the solution is"
    ]
    
    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
    
    reasoning_reward = min(0.5, reasoning_count * 0.1)
    format_reward += reasoning_reward
    
    math_indicators = ["=", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}", "∑", "∫", "∞"]
    math_count = sum(1 for indicator in math_indicators if indicator in response)
    math_reward = min(0.2, math_count * 0.05)
    format_reward += math_reward
    
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if len(sentences) > 1:
        format_reward += 0.2

    answer_indicators = ["answer:", "result:", "solution:", "therefore", "thus", "hence"]
    has_answer_section = any(indicator in response_lower for indicator in answer_indicators)
    if has_answer_section:
        format_reward += 0.1
    
    if format_reward < 0.1 and response_lower:
        format_reward = 0.1
    
    return format_reward
    
def calculate_correctness_reward(extracted_answer, validation_object):
    if extracted_answer:
        is_correct = validation_object["validate_function"](extracted_answer, validation_object["expected_answer"])
        return 1.0 if is_correct else 0.0
    else:
        return 0.0


def calculate_rewards(batch_responses, validation_objects):
    format_rewards = np.array([calculate_format_reward(response) 
                               for response in batch_responses])
    
    extracted_answers = [extract_answer(response) for response in batch_responses]
    correctness_rewards = np.array([calculate_correctness_reward(extracted_answer, val_obj)
                                    for val_obj, extracted_answer in zip(validation_objects, extracted_answers)])
    
    rewards = (FORMAT_REWARD_WEIGHT * format_rewards + 
               CORRECTNESS_REWARD_WEIGHT * correctness_rewards)
    return rewards
