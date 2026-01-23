"""Load REALM dataset from HuggingFace"""

import numpy as np
from dataclasses import dataclass
from typing import List
from datasets import load_dataset


@dataclass
class Sample:
    prompt: str
    category: str
    occupation: str = None
    source: str = None


def load_realm(n: int = 50, seed: int = 42, source_filter: str = None) -> List[Sample]:
    """Load REALM dataset - real-world LLM use cases from Reddit/news"""
    ds = load_dataset("kkChimmy/REALM", split="train")
    
    # Filter by source if specified
    if source_filter:
        ds = ds.filter(lambda x: x.get('source', '').lower() == source_filter.lower())
    
    idx = np.random.RandomState(seed).choice(len(ds), min(n, len(ds)), replace=False)
    samples = []
    for i in idx:
        item = ds[int(i)]
        text = item.get('text', item.get('content', item.get('use_case', '')))
        category = item.get('use_category', item.get('category', 'unknown'))
        occupation = item.get('occupation', None)
        source = item.get('source', 'unknown')
        samples.append(Sample(text, category, occupation, source))
    return samples


def load_data(name: str = "realm", n: int = 50, seed: int = 42) -> List[Sample]:
    if name == "realm":
        return load_realm(n, seed)
    if name == "reddit":
        return load_realm(n, seed, source_filter="reddit")
    if name == "news":
        return load_realm(n, seed, source_filter="news")
    raise ValueError(f"Unknown dataset: {name}. Use 'realm', 'reddit', or 'news'")