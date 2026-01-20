"""
SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution
Concise implementation using Qwen models via Ollama/HuggingFace
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from math import factorial


class CoalitionStrategy(Enum):
    DELETE = "delete"
    MASK = "mask"


@dataclass
class ShapleyResult:
    token_idx: int
    token_text: str
    shapley_value: float
    variance: float = 0.0
    leave_one_out_delta: float = 0.0


@dataclass
class CoalitionResult:
    coalition: frozenset
    payoff: float
    output_text: str


@dataclass
class SFAFeature:
    """Feature vector for Stage 2 refinement"""
    token_idx: int
    stage1_value: float
    position_norm: float
    token_length: int
    is_punct: bool
    is_stopword: bool
    pos_category: int
    idf_score: float
    variance: float
    loo_delta: float
    neighbor_synergy: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.stage1_value,
            self.position_norm,
            self.token_length / 20.0,
            float(self.is_punct),
            float(self.is_stopword),
            self.pos_category / 2.0,
            self.idf_score,
            self.variance,
            self.loo_delta,
            self.neighbor_synergy
        ])


# ============ LLM Backends ============

class OllamaBackend:
    """Ollama backend using Qwen model"""
    
    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        
        import ollama
        self.client = ollama.Client(host=base_url)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True
        )
    
    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": self.temperature, "seed": self.seed}
        )
        return response["response"]
    
    def get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings(model=self.embedding_model, prompt=text)
        return np.array(response["embedding"])
    
    def tokenize(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]).strip() for tid in token_ids]
        return [t for t in tokens if t]


class HuggingFaceBackend:
    """HuggingFace backend using Qwen model"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        temperature: float = 0.0,
        seed: int = 42
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        
        torch.manual_seed(seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.01, self.temperature),
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def tokenize(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.decode([tid]).strip() for tid in token_ids if self.tokenizer.decode([tid]).strip()]


# ============ Linguistic Features ============

class LinguisticExtractor:
    """Extract POS, IDF features using spaCy (optional)"""
    
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'we', 'they', 'as', 'so', 'than', 'just', 'only'
    }
    
    IDF_DEFAULTS = {
        'the': 0.1, 'a': 0.15, 'an': 0.2, 'is': 0.12, 'are': 0.15,
        'and': 0.12, 'or': 0.2, 'but': 0.22, 'to': 0.1, 'of': 0.1,
        'in': 0.12, 'for': 0.15, 'with': 0.18, 'that': 0.15
    }
    
    POS_CONTENT = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}
    POS_FUNCTION = {'DET', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PRON', 'PART'}
    
    def __init__(self):
        self.nlp = None
        self._use_spacy = False
        # Try loading spaCy
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self._use_spacy = True
        except:
            pass  # Fall back to heuristics
    
    def extract(self, tokens: List[str]) -> List[Dict]:
        if not self._use_spacy:
            return self._extract_fallback(tokens)
        
        text = " ".join(tokens)
        doc = self.nlp(text)
        
        features = []
        spacy_idx = 0
        
        for token in tokens:
            token_lower = token.lower().strip()
            pos_tag, is_stop = "X", token_lower in self.STOPWORDS
            
            while spacy_idx < len(doc):
                if doc[spacy_idx].text.strip() == token.strip():
                    pos_tag = doc[spacy_idx].pos_
                    is_stop = doc[spacy_idx].is_stop
                    spacy_idx += 1
                    break
                spacy_idx += 1
            
            if pos_tag in self.POS_CONTENT:
                pos_cat = 0
            elif pos_tag in self.POS_FUNCTION:
                pos_cat = 1
            else:
                pos_cat = 2
            
            idf = self.IDF_DEFAULTS.get(token_lower, 0.7 if len(token_lower) > 2 else 0.3)
            
            features.append({
                'pos_tag': pos_tag, 'pos_category': pos_cat,
                'idf_score': idf, 'is_stopword': is_stop
            })
        
        return features
    
    def _extract_fallback(self, tokens: List[str]) -> List[Dict]:
        """Heuristic-based extraction without spaCy"""
        features = []
        for token in tokens:
            token_lower = token.lower().strip()
            is_punct = all(c in ".,!?;:'-\"()[]{}/" for c in token.strip()) if token.strip() else False
            is_stop = token_lower in self.STOPWORDS
            
            if is_punct:
                pos_cat = 2
            elif is_stop:
                pos_cat = 1
            else:
                pos_cat = 0
            
            idf = self.IDF_DEFAULTS.get(token_lower, 0.7 if len(token_lower) > 2 else 0.3)
            
            features.append({
                'pos_tag': 'X', 'pos_category': pos_cat,
                'idf_score': idf, 'is_stopword': is_stop
            })
        return features


# ============ Coalition Handling ============

class CoalitionCache:
    """Cache for coalition evaluations"""
    
    def __init__(self):
        self._cache: Dict[frozenset, CoalitionResult] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, coalition: frozenset) -> Optional[CoalitionResult]:
        result = self._cache.get(coalition)
        if result: self.hits += 1
        else: self.misses += 1
        return result
    
    def set(self, coalition: frozenset, result: CoalitionResult):
        self._cache[coalition] = result
    
    @property
    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {"size": len(self._cache), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(1, total)}


class CoalitionGenerator:
    """Generate coalitions with stratified paired sampling"""
    
    def __init__(self, n_tokens: int, budget: int, seed: int = 42):
        self.n_tokens = n_tokens
        self.budget = budget
        self.rng = np.random.RandomState(seed)
        self.all_tokens = frozenset(range(n_tokens))
        self.coalitions = self._generate()
    
    def _generate(self) -> List[frozenset]:
        coalitions = {self.all_tokens, frozenset()}
        
        # Leave-one-out coalitions
        for i in range(self.n_tokens):
            coalitions.add(self.all_tokens - {i})
            coalitions.add(frozenset({i}))
        
        # Stratified paired sampling
        remaining = self.budget - len(coalitions)
        if remaining > 0:
            max_size = self.n_tokens // 2
            sizes = list(range(1, max_size + 1)) or [1]
            samples_per_size = max(1, (remaining // 2) // len(sizes))
            
            for size in sizes:
                for _ in range(samples_per_size):
                    if len(coalitions) >= self.budget:
                        break
                    members = self.rng.choice(self.n_tokens, size=size, replace=False)
                    coalition = frozenset(members)
                    coalitions.add(coalition)
                    coalitions.add(self.all_tokens - coalition)
        
        return list(coalitions)


class PromptRenderer:
    """Render prompts from coalitions"""
    
    def __init__(self, tokens: List[str], strategy: CoalitionStrategy = CoalitionStrategy.DELETE):
        self.tokens = tokens
        self.strategy = strategy
    
    def render(self, coalition: frozenset) -> str:
        if self.strategy == CoalitionStrategy.DELETE:
            return " ".join(self.tokens[i] for i in sorted(coalition))
        return " ".join(self.tokens[i] if i in coalition else "[MASK]" for i in range(len(self.tokens)))


# ============ Value Function ============

class ValueFunction:
    """Embedding similarity value function"""
    
    def __init__(self, backend):
        self.backend = backend
        self.ref_output = None
        self.ref_embedding = None
    
    def set_reference(self, prompt: str):
        self.ref_output = self.backend.generate(prompt)
        self.ref_embedding = self.backend.get_embedding(self.ref_output)
    
    def __call__(self, output: str) -> float:
        if not output.strip():
            return 0.0
        out_emb = self.backend.get_embedding(output)
        dot = np.dot(self.ref_embedding, out_emb)
        norm = np.linalg.norm(self.ref_embedding) * np.linalg.norm(out_emb)
        return float(dot / (norm + 1e-10))


# ============ Stage 1: TokenSHAP ============

class Stage1TokenSHAP:
    """Monte Carlo Shapley estimation with proper weighting"""
    
    def __init__(self, backend, value_fn: ValueFunction, budget: int = 100, 
                 cache: CoalitionCache = None, seed: int = 42):
        self.backend = backend
        self.value_fn = value_fn
        self.budget = budget
        self.cache = cache or CoalitionCache()
        self.seed = seed
    
    def _shapley_weight(self, n: int, s: int) -> float:
        if s < 0 or s >= n:
            return 0.0
        return factorial(s) * factorial(n - s - 1) / factorial(n)
    
    def compute(self, prompt: str) -> Tuple[List[ShapleyResult], Dict[frozenset, CoalitionResult]]:
        tokens = self.backend.tokenize(prompt)
        n = len(tokens)
        if n == 0:
            return [], {}
        
        self.value_fn.set_reference(prompt)
        
        generator = CoalitionGenerator(n, self.budget, self.seed)
        renderer = PromptRenderer(tokens)
        
        results = {}
        for coalition in generator.coalitions:
            cached = self.cache.get(coalition)
            if cached:
                results[coalition] = cached
                continue
            
            coalition_prompt = renderer.render(coalition)
            output = self.backend.generate(coalition_prompt) if coalition else ""
            payoff = self.value_fn(output) if output else 0.0
            
            result = CoalitionResult(coalition, payoff, output)
            results[coalition] = result
            self.cache.set(coalition, result)
        
        # Compute Shapley values
        full_coalition = frozenset(range(n))
        full_payoff = results[full_coalition].payoff
        
        shapley_results = []
        for i in range(n):
            shapley_value = 0.0
            total_weight = 0.0
            
            without_i = [c for c in results if i not in c]
            
            for coalition in without_i:
                s = len(coalition)
                coalition_with_i = coalition | {i}
                
                if coalition_with_i in results:
                    weight = self._shapley_weight(n, s)
                    v_with = results[coalition_with_i].payoff
                    v_without = results[coalition].payoff
                    shapley_value += weight * (v_with - v_without)
                    total_weight += weight
            
            if total_weight > 0 and total_weight < 1.0:
                shapley_value /= total_weight
            
            with_i = [results[c].payoff for c in results if i in c]
            variance = np.var(with_i) if len(with_i) > 1 else 0.0
            
            loo = full_coalition - {i}
            loo_delta = full_payoff - results.get(loo, CoalitionResult(loo, 0.0, "")).payoff
            
            shapley_results.append(ShapleyResult(
                token_idx=i, token_text=tokens[i],
                shapley_value=shapley_value, variance=variance,
                leave_one_out_delta=loo_delta
            ))
        
        return shapley_results, results


# ============ Stage 2: SFA Refinement ============

class Stage2SFARefinement:
    """SFA-style refinement using learned features"""
    
    def __init__(self, backend, value_fn: ValueFunction, renderer: PromptRenderer):
        self.backend = backend
        self.value_fn = value_fn
        self.renderer = renderer
        self.ling_extractor = LinguisticExtractor()
        self._refiner = None
    
    def build_features(self, tokens: List[str], stage1: List[ShapleyResult],
                       coalition_results: Dict) -> List[SFAFeature]:
        n = len(tokens)
        ling_features = self.ling_extractor.extract(tokens)
        full_coalition = frozenset(range(n))
        full_payoff = coalition_results[full_coalition].payoff if full_coalition in coalition_results else 0.0
        
        features = []
        for i, (token, shap) in enumerate(zip(tokens, stage1)):
            ling = ling_features[i] if i < len(ling_features) else {
                'pos_category': 0, 'idf_score': 0.5, 'is_stopword': False
            }
            
            synergy = 0.0
            if i < n - 1:
                pair_removed = full_coalition - {i, i + 1}
                i_removed = full_coalition - {i}
                j_removed = full_coalition - {i + 1}
                
                if all(c in coalition_results for c in [pair_removed, i_removed, j_removed]):
                    v_pair = coalition_results[pair_removed].payoff
                    v_i = coalition_results[i_removed].payoff
                    v_j = coalition_results[j_removed].payoff
                    synergy = (full_payoff - v_pair) - (full_payoff - v_i) - (full_payoff - v_j)
            
            is_punct = all(c in ".,!?;:'-\"()[]{}/" for c in token.strip()) if token.strip() else False
            
            features.append(SFAFeature(
                token_idx=i,
                stage1_value=shap.shapley_value,
                position_norm=i / n,
                token_length=len(token),
                is_punct=is_punct,
                is_stopword=ling['is_stopword'],
                pos_category=ling['pos_category'],
                idf_score=ling['idf_score'],
                variance=shap.variance,
                loo_delta=shap.leave_one_out_delta,
                neighbor_synergy=synergy
            ))
        
        return features
    
    def compute_target(self, tokens: List[str], stage1: List[ShapleyResult], k: int = 3) -> np.ndarray:
        n = len(tokens)
        k = min(k, max(1, n // 2))
        
        ranked = sorted(enumerate(stage1), key=lambda x: x[1].shapley_value, reverse=True)
        top_idx = [idx for idx, _ in ranked[:k]]
        bottom_idx = [idx for idx, _ in ranked[-k:]]
        
        top_removed = frozenset(range(n)) - frozenset(top_idx)
        top_prompt = self.renderer.render(top_removed)
        top_output = self.backend.generate(top_prompt) if top_removed else ""
        delta_top = self.value_fn(self.value_fn.ref_output) - self.value_fn(top_output)
        
        bottom_removed = frozenset(range(n)) - frozenset(bottom_idx)
        bottom_prompt = self.renderer.render(bottom_removed)
        bottom_output = self.backend.generate(bottom_prompt) if bottom_removed else ""
        delta_bottom = self.value_fn(self.value_fn.ref_output) - self.value_fn(bottom_output)
        
        gap = delta_top - delta_bottom
        
        targets = np.zeros(n)
        for idx in top_idx:
            targets[idx] = max(0, gap)
        for idx in bottom_idx:
            targets[idx] = min(0, -gap)
        
        return targets
    
    def fit(self, features: List[SFAFeature], targets: np.ndarray):
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = np.array([f.to_array() for f in features])
        self._refiner = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self._refiner.fit(X, targets)
    
    def refine(self, features: List[SFAFeature], stage1_values: np.ndarray) -> np.ndarray:
        if self._refiner is None:
            return stage1_values.copy()
        
        X = np.array([f.to_array() for f in features])
        adjustments = self._refiner.predict(X)
        return stage1_values + adjustments


# ============ Main Explainer ============

class SFATokenSHAP:
    """Complete SFA-TokenSHAP pipeline"""
    
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'it', 'its', 'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they'
    }
    
    def __init__(self, backend, budget: int = 100, seed: int = 42, suppress_stopwords: bool = True):
        self.backend = backend
        self.budget = budget
        self.seed = seed
        self.suppress_stopwords = suppress_stopwords
        self.cache = CoalitionCache()
    
    def explain(self, prompt: str, train_refiner: bool = True, normalize: bool = True) -> Dict:
        value_fn = ValueFunction(self.backend)
        
        # Stage 1
        stage1 = Stage1TokenSHAP(self.backend, value_fn, self.budget, self.cache, self.seed)
        stage1_results, coalition_results = stage1.compute(prompt)
        
        tokens = [r.token_text for r in stage1_results]
        stage1_values = np.array([r.shapley_value for r in stage1_results])
        
        # Stage 2
        renderer = PromptRenderer(tokens)
        stage2 = Stage2SFARefinement(self.backend, value_fn, renderer)
        features = stage2.build_features(tokens, stage1_results, coalition_results)
        
        if train_refiner and len(tokens) > 3:
            targets = stage2.compute_target(tokens, stage1_results)
            stage2.fit(features, targets)
        
        stage2_values = stage2.refine(features, stage1_values)
        
        # Stopword suppression
        if self.suppress_stopwords:
            for i, token in enumerate(tokens):
                token_lower = token.lower().strip()
                is_punct = all(c in ".,!?;:'-\"()[]{}/" for c in token.strip())
                is_stop = token_lower in self.STOPWORDS
                
                if is_stop or is_punct or not token.strip():
                    stage2_values[i] *= 0.3
                    if abs(stage2_values[i]) < 0.01:
                        stage2_values[i] = 0.0
        
        # Normalize
        if normalize:
            stage1_norm = stage1_values / (np.abs(stage1_values).max() + 1e-10)
            stage2_norm = stage2_values / (np.abs(stage2_values).max() + 1e-10)
        else:
            stage1_norm = stage1_values
            stage2_norm = stage2_values
        
        return {
            "prompt": prompt,
            "tokens": tokens,
            "stage1_values": stage1_norm,
            "stage2_values": stage2_norm,
            "stage1_raw": stage1_values,
            "stage2_raw": stage2_values,
            "features": features,
            "coalition_results": coalition_results,
            "cache_stats": self.cache.stats
        }
    
    def visualize(self, result: Dict, use_stage2: bool = True) -> str:
        tokens = result["tokens"]
        values = result["stage2_values"] if use_stage2 else result["stage1_values"]
        
        max_abs = np.abs(values).max()
        normalized = values / max_abs if max_abs > 0 else values
        
        html = ['<div style="font-family: Arial; line-height: 2;">']
        
        for token, value in zip(tokens, normalized):
            if value > 0:
                intensity = int(value * 200)
                bg = f"rgba(255, {200-intensity}, {200-intensity}, 0.8)"
            else:
                intensity = int(-value * 200)
                bg = f"rgba({200-intensity}, {200-intensity}, 255, 0.8)"
            
            html.append(f'<span style="background:{bg}; padding:2px 4px; margin:1px; '
                       f'border-radius:3px;" title="SHAP:{value:.4f}">{token}</span>')
        
        html.append('</div>')
        return " ".join(html)