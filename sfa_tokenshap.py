"""
SFA-TokenSHAP: Two-Stage Shapley Refinement
Learned suppression via POS/IDF/dependency - no hardcoded stopwords
Qwen tokenizer aligned with Qwen LLM
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from math import factorial
import spacy
from transformers import AutoTokenizer
import ollama

CLOSED_CLASS_POS = {'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ'}
FUNCTION_DEP = {'det', 'aux', 'auxpass', 'mark', 'case', 'cc'}


@dataclass
class ShapleyResult:
    token_idx: int
    token_text: str
    shapley_value: float


class LinguisticExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_weights(self, tokens: List[str]) -> np.ndarray:
        doc = self.nlp(" ".join(tokens))
        weights = np.ones(len(tokens))
        sp_idx = 0
        for i, tok in enumerate(tokens):
            if tok.strip() and all(c in ".,!?;:'-\"()[]{}/" for c in tok.strip()):
                weights[i] = 0.1
                continue
            while sp_idx < len(doc):
                if doc[sp_idx].text.strip() == tok.strip():
                    sp = doc[sp_idx]
                    if sp.pos_ in CLOSED_CLASS_POS:
                        weights[i] *= 0.3
                    if sp.dep_ in FUNCTION_DEP:
                        weights[i] *= 0.4
                    if sum(1 for _ in sp.ancestors) <= 1 and sp.dep_ != "ROOT":
                        weights[i] *= 0.6
                    if sp.dep_ == "ROOT":
                        weights[i] = max(weights[i], 0.8)
                    if len(list(sp.children)) >= 2:
                        weights[i] = max(weights[i], 0.7)
                    sp_idx += 1
                    break
                sp_idx += 1
        return weights


class OllamaBackend:
    def __init__(self, model: str = "qwen3:0.6b", embed_model: str = "nomic-embed-text", seed: int = 42):
        self.model = model
        self.embed_model = embed_model
        self.seed = seed
        self.client = ollama.Client()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    
    def generate(self, prompt: str) -> str:
        return self.client.generate(model=self.model, prompt=prompt,
                                    options={"temperature": 0.0, "seed": self.seed})["response"]
    
    def embed(self, text: str) -> np.ndarray:
        return np.array(self.client.embeddings(model=self.embed_model, prompt=text)["embedding"])
    
    def tokenize(self, text: str) -> List[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [t for t in [self.tokenizer.decode([i]).strip() for i in ids] if t]


class SFATokenSHAP:
    def __init__(self, backend: OllamaBackend, budget: int = 100, seed: int = 42):
        self.backend = backend
        self.budget = budget
        self.rng = np.random.RandomState(seed)
        self.ling = LinguisticExtractor()
    
    def _weight(self, n: int, s: int) -> float:
        return factorial(s) * factorial(n - s - 1) / factorial(n) if 0 <= s < n else 0.0
    
    def _coalitions(self, n: int) -> List[frozenset]:
        full = frozenset(range(n))
        cs = {full, frozenset()}
        for i in range(n):
            cs.add(full - {i})
            cs.add(frozenset({i}))
        for sz in range(1, n // 2 + 1):
            for _ in range(max(1, (self.budget - len(cs)) // n)):
                if len(cs) >= self.budget:
                    break
                m = frozenset(self.rng.choice(n, size=sz, replace=False))
                cs.add(m)
                cs.add(full - m)
        return list(cs)
    
    def explain(self, prompt: str) -> Dict:
        tokens = self.backend.tokenize(prompt)
        n = len(tokens)
        
        ref_out = self.backend.generate(prompt)
        ref_emb = self.backend.embed(ref_out)
        
        def value(c: frozenset) -> float:
            if not c:
                return 0.0
            p = " ".join(tokens[i] for i in sorted(c))
            out = self.backend.generate(p)
            if not out or not out.strip():
                return 0.0
            emb = self.backend.embed(out)
            if emb.size == 0:
                return 0.0
            return float(np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-10))
        
        coalitions = self._coalitions(n)
        payoffs = {c: value(c) for c in coalitions}
        
        full = frozenset(range(n))
        shapley = np.zeros(n)
        for i in range(n):
            for c in coalitions:
                if i not in c and (c | {i}) in payoffs:
                    shapley[i] += self._weight(n, len(c)) * (payoffs[c | {i}] - payoffs[c])
        
        weights = self.ling.get_weights(tokens)
        refined = shapley * weights
        
        s1 = shapley / (np.abs(shapley).max() + 1e-10)
        s2 = refined / (np.abs(refined).max() + 1e-10)
        
        return {"tokens": tokens, "stage1": s1, "stage2": s2, "weights": weights}