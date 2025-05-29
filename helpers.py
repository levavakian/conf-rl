import re
import json
import random
import math
from datetime import datetime, timezone
from datasets import load_dataset, Dataset
from typing import Tuple, List, Dict, Any

def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_confidence(text: str) -> str:
    if "<confidence>" not in text or "</confidence>" not in text:
        return ""
    confidence = text.split("<confidence>")[-1]
    confidence = confidence.split("</confidence>")[0]
    return confidence.strip()

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def timestamp_to_iso(timestamp_ms: int) -> str:
    """
    Convert Unix timestamp in milliseconds to ISO string format.
    
    Args:
        timestamp_ms: Unix timestamp in milliseconds
    
    Returns:
        ISO formatted datetime string
    """
    if timestamp_ms == 0:
        return ""
    
    # Convert milliseconds to seconds
    timestamp_s = timestamp_ms / 1000
    
    # Create datetime object and convert to ISO format
    dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
    return dt.isoformat()

def load_and_split_markets(filename: str, seed: int = 42, train_ratio: float = 0.9) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load markets data from JSON file, shuffle deterministically, and split into train/test sets.
    
    Args:
        filename: Path to the JSON file containing markets data
        seed: Random seed for deterministic shuffling
        train_ratio: Ratio of data to use for training (default 0.9 for 90/10 split)
    
    Returns:
        Tuple of (train_data, test_data) where each is a list of market dictionaries
    """
    # Load the JSON data
    with open(filename, 'r', encoding='utf-8') as f:
        markets_data = json.load(f)
    
    # Set random seed for deterministic shuffling
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = markets_data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    total_items = len(shuffled_data)
    train_size = int(total_items * train_ratio)
    
    # Split the data
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]
    
    return train_data, test_data

def create_markets_dataset(markets_data: List[Dict[str, Any]], system_prompt: str) -> Dataset:
    """
    Convert markets data into a Dataset format suitable for training.
    
    Args:
        markets_data: List of market dictionaries from load_and_split_markets
        system_prompt: System prompt to use for the conversation format
    
    Returns:
        Dataset with 'prompt' and 'answer' fields formatted for training
    """
    dataset_items = []
    
    for market in markets_data:
        # Create the prompt in conversation format
        prompt_question = (
            f"Provide a YES or NO prediction and confidence percentage for the following question that was opened at {timestamp_to_iso(market.get('openTime', 0))}:\n\n"
            f"{market['question']}"
        )
        prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt_question}
        ]
        
        # Extract the answer (resolution)
        answer = market['resolution']
        
        # Convert timestamps to ISO strings
        dataset_items.append({
            'prompt': prompt,
            'answer': answer,
        })
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(dataset_items)
    
    return dataset

def unpack_reward(extracted, answer):
    if extracted[0] != "YES" and extracted[0] != "NO":
        return 0.0

    if not is_float(extracted[1]):
        return 0.0
    
    confidence = float(extracted[1])
    if confidence < 0.5 or confidence >= 1.0:
        return 0.0
    
    is_correct = extracted[0] == answer
    log_odds = min(1000, math.log(confidence / max(1 - confidence, 1e-10)))

    return log_odds if is_correct else -log_odds

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    print("====PROMPTS====")
    print(prompts)
    print("====COMPLETIONS====")
    print(completions)
    print("====ANSWER====")
    print(answer)
    responses = [completion[0]['content'] for completion in completions]
    extracted_results = [(extract_xml_answer(r).upper(), extract_xml_confidence(r)) for r in responses]
    return [unpack_reward(r, a) for r, a in zip(extracted_results, answer)]

def confidence_format_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_confidence(r) for r in responses]
    return [0.5 if is_float(r) and 0.5 <= float(r) < 1.0 else 0.0 for r in extracted_responses]

def answer_format_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r).upper() for r in responses]
    return [0.5 if r == "YES" or r == "NO" else 0.0 for r in extracted_responses]