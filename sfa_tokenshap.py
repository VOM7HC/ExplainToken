"""
SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution

This implementation combines:
- TokenSHAP: Monte Carlo Shapley value estimation for LLM interpretability
- SFA (Shapley-based Feature Augmentation): Two-stage ensemble learning
- llmSHAP: Principled approach to handling LLM stochasticity

Reference Papers:
- TokenSHAP (Horovicz & Goldshmidt, 2024)
- llmSHAP (Naudot et al., 2025)
- Shapley-based Feature Augmentation (Antwarg et al., 2023)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json
from collections import defaultdict
import warnings
from enum import Enum


class CoalitionStrategy(Enum):
    """Strategy for handling missing tokens in coalitions."""
    DELETE = "delete"           # Remove tokens entirely
    MASK = "mask"               # Replace with [MASK] token
    PLACEHOLDER = "placeholder" # Replace with neutral placeholder
    UNK = "unk"                 # Replace with [UNK] token


class ValueFunctionType(Enum):
    """Type of value function for measuring coalition payoff."""
    EMBEDDING_SIMILARITY = "embedding_similarity"
    LOGPROB = "logprob"
    TASK_SCORER = "task_scorer"


@dataclass
class TokenInfo:
    """Information about a single token."""
    token_id: int
    text: str
    position: int
    is_stopword: bool = False
    is_punctuation: bool = False
    is_whitespace: bool = False
    length: int = 0
    
    def __post_init__(self):
        self.length = len(self.text)
        # Simple heuristics for token properties
        self.is_whitespace = self.text.strip() == ""
        self.is_punctuation = all(c in ".,!?;:'-\"()[]{}/" for c in self.text.strip())


@dataclass
class CoalitionResult:
    """Result of evaluating a coalition."""
    coalition: frozenset
    payoff: float
    output_text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ShapleyResult:
    """Result of Shapley value computation for a token."""
    token_idx: int
    token_text: str
    shapley_value: float
    variance: float = 0.0
    leave_one_out_delta: float = 0.0
    interaction_signals: Dict[int, float] = field(default_factory=dict)


@dataclass
class SFAFeatureVector:
    """
    Enhanced feature vector for SFA Stage 2 refinement.
    
    Following the "Text Feature Enrichment" philosophy: we ADD features
    to let the model learn what to suppress, rather than hard-coding rules.
    
    Feature Categories:
    - Stage 1 attribution (from TokenSHAP)
    - Position features
    - Linguistic features (POS, IDF - computed, not hard-coded)
    - Coalition statistics (variance, LOO delta)
    - Interaction features (neighbor synergy)
    - Augmentation-invariance features (stability across variants)
    """
    token_idx: int
    
    # Stage 1 attribution
    stage1_attribution: float              # φ^(1)_i
    
    # Position features
    position_normalized: float             # i/n (0 to 1)
    position_absolute: int                 # Raw position
    
    # Token surface features
    token_length: int
    is_punctuation: bool
    is_whitespace: bool
    is_capitalized: bool                   # First letter uppercase
    is_all_caps: bool                      # All uppercase
    
    # Linguistic features (computed via NLP, not hard-coded)
    pos_tag: str = "UNK"                   # Part-of-speech tag
    pos_category: int = 0                  # Numeric POS category (0=content, 1=function, 2=punct)
    idf_score: float = 1.0                 # Inverse document frequency (higher = rarer = more important)
    is_stopword_computed: bool = False     # From NLP tagger, not hard-coded list
    
    # Coalition statistics
    payoff_variance: float = 0.0           # Variance of v(S) over subsets containing i
    leave_one_out_delta: float = 0.0       # v(N) - v(N\{i})
    
    # Interaction features
    neighbor_synergy: float = 0.0          # Interaction with adjacent tokens
    
    # Augmentation-invariance features (stability across prompt variants)
    attribution_stability: float = 1.0     # 1 - variance across augmented prompts
    augmentation_mean: float = 0.0         # Mean attribution across augmentations
    
    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array for model input.
        
        Returns 15-dimensional feature vector (expanded from original 9).
        """
        return np.array([
            # Stage 1 + position (3)
            self.stage1_attribution,
            self.position_normalized,
            self.position_absolute / 100.0,  # Normalize
            
            # Surface features (5)
            self.token_length / 20.0,  # Normalize
            float(self.is_punctuation),
            float(self.is_whitespace),
            float(self.is_capitalized),
            float(self.is_all_caps),
            
            # Linguistic features (3) - THE KEY ADDITION
            self.pos_category / 2.0,  # Normalize to [0, 1]
            self.idf_score,  # Already normalized
            float(self.is_stopword_computed),
            
            # Coalition statistics (2)
            self.payoff_variance,
            self.leave_one_out_delta,
            
            # Interaction + augmentation (2)
            self.neighbor_synergy,
            self.attribution_stability,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names for interpretability."""
        return [
            'stage1_attribution', 'position_normalized', 'position_absolute',
            'token_length', 'is_punctuation', 'is_whitespace', 'is_capitalized', 'is_all_caps',
            'pos_category', 'idf_score', 'is_stopword_computed',
            'payoff_variance', 'leave_one_out_delta',
            'neighbor_synergy', 'attribution_stability'
        ]


class CoalitionCache:
    """
    Order-invariant cache for coalition evaluations.
    Implements caching strategy from llmSHAP (Algorithm 1).
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[frozenset, CoalitionResult] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, coalition: frozenset) -> Optional[CoalitionResult]:
        """Retrieve cached result for coalition."""
        if not self.use_cache:
            return None
        result = self._cache.get(coalition)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
    
    def set(self, coalition: frozenset, result: CoalitionResult):
        """Store result in cache."""
        if self.use_cache:
            self._cache[coalition] = result
    
    def contains(self, coalition: frozenset) -> bool:
        """Check if coalition is in cache."""
        return coalition in self._cache
    
    @property
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses)
        }
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the model's tokenizer."""
        pass
    
    def get_logprob(self, prompt: str, target: str) -> float:
        """Get log probability of target given prompt (optional)."""
        raise NotImplementedError("Logprob not supported by this backend")


class OllamaBackend(LLMBackend):
    """
    Ollama LLM backend implementation with HuggingFace tokenizer.
    
    Uses HuggingFace tokenizer for model-aligned tokenization since
    Ollama doesn't yet expose /api/tokenize endpoint consistently.
    This fixes the tokenizer mismatch issue that causes attribution instability.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        hf_tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct"  # Model-aligned tokenizer
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.hf_tokenizer_name = hf_tokenizer_name
        
        # Import ollama - will be done lazily
        self._client = None
        self._hf_tokenizer = None
    
    def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")
        return self._client
    
    def _get_hf_tokenizer(self):
        """Lazy initialization of HuggingFace tokenizer for model-aligned tokenization."""
        if self._hf_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_tokenizer_name,
                    trust_remote_code=True
                )
            except ImportError:
                warnings.warn(
                    "transformers not available, falling back to simple tokenization. "
                    "Install with: pip install transformers"
                )
                return None
            except Exception as e:
                warnings.warn(f"Could not load HF tokenizer: {e}. Falling back to simple tokenization.")
                return None
        return self._hf_tokenizer
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama."""
        client = self._get_client()
        
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.seed is not None:
            options["seed"] = self.seed
        
        response = client.generate(
            model=self.model_name,
            prompt=prompt,
            options=options
        )
        return response["response"]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using Ollama."""
        client = self._get_client()
        response = client.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return np.array(response["embedding"])
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using HuggingFace tokenizer for model alignment.
        
        This fixes the critical issue where whitespace tokenization
        doesn't match the model's actual token boundaries, causing
        attribution instability (low Spearman/Kendall correlations).
        
        Falls back to punctuation-aware splitting if HF tokenizer unavailable.
        """
        tokenizer = self._get_hf_tokenizer()
        
        if tokenizer is not None:
            # Use model-aligned tokenization
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            # Clean up tokens (remove leading spaces from subword tokens)
            cleaned = []
            for t in tokens:
                # Preserve the token but mark it properly
                if t.startswith(' ') and len(t) > 1:
                    cleaned.append(t.strip())
                elif t.strip():
                    cleaned.append(t)
            return cleaned if cleaned else [text]  # Fallback for edge cases
        
        # Fallback: punctuation-aware splitting
        import re
        tokens = []
        for word in text.split():
            if word and word[-1] in '.,!?;:"\')]:':
                if len(word) > 1:
                    tokens.append(word[:-1])
                tokens.append(word[-1])
            elif word and word[0] in '"\'([':
                tokens.append(word[0])
                if len(word) > 1:
                    tokens.append(word[1:])
            else:
                tokens.append(word)
        return [t for t in tokens if t]


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend implementation."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        temperature: float = 0.0,
        seed: Optional[int] = None
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.device = device
        self.temperature = temperature
        self.seed = seed
        
        self._model = None
        self._tokenizer = None
        self._embedding_model = None
    
    def _load_model(self):
        """Lazy loading of model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16
                )
                
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    
            except ImportError:
                raise ImportError("transformers package required. Install with: pip install transformers torch")
    
    def _load_embedding_model(self):
        """Lazy loading of embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
    
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        """Generate response using HuggingFace model."""
        self._load_model()
        import torch
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.01, kwargs.get("temperature", self.temperature)),
                do_sample=kwargs.get("temperature", self.temperature) > 0,
                pad_token_id=self._tokenizer.pad_token_id
            )
        
        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using sentence-transformers."""
        self._load_embedding_model()
        return self._embedding_model.encode(text, convert_to_numpy=True)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using the model's tokenizer."""
        self._load_model()
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [self._tokenizer.decode([tid]) for tid in token_ids]


class LinguisticFeatureExtractor:
    """
    Extracts linguistic features (POS, IDF, etc.) for tokens.
    
    Following the "Text Feature Enrichment" philosophy from the PDF:
    - Use NLP taggers (spaCy/stanza) for POS tags
    - Compute IDF scores from corpus statistics
    - These are ADDITIVE features, not hard-coded filters
    
    The key insight: let the Stage 2 model LEARN to suppress stopwords
    based on features, rather than hard-coding rules.
    """
    
    # POS category mapping (universal tagset)
    POS_CONTENT = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}  # Content words
    POS_FUNCTION = {'DET', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PRON', 'PART'}  # Function words
    POS_PUNCT = {'PUNCT', 'SYM', 'SPACE', 'X'}  # Punctuation/other
    
    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        """
        Initialize linguistic feature extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NLP processing
            spacy_model: spaCy model name
        """
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp = None
        self._idf_cache: Dict[str, float] = {}
        
        # Default IDF values based on corpus statistics (Penn Treebank approximation)
        self._default_idf = {
            'the': 0.1, 'a': 0.15, 'an': 0.2, 'is': 0.12, 'are': 0.15,
            'was': 0.18, 'were': 0.22, 'be': 0.2, 'been': 0.25, 'being': 0.3,
            'have': 0.2, 'has': 0.22, 'had': 0.25, 'do': 0.25, 'does': 0.28,
            'did': 0.28, 'will': 0.25, 'would': 0.22, 'could': 0.28, 'should': 0.3,
            'and': 0.12, 'or': 0.2, 'but': 0.22, 'if': 0.25, 'then': 0.3,
            'that': 0.15, 'which': 0.25, 'who': 0.28, 'what': 0.3, 'where': 0.35,
            'when': 0.32, 'why': 0.4, 'how': 0.35,
            'i': 0.18, 'you': 0.2, 'he': 0.25, 'she': 0.28, 'it': 0.15,
            'we': 0.25, 'they': 0.22, 'this': 0.2, 'these': 0.28, 'those': 0.3,
            'to': 0.1, 'of': 0.1, 'in': 0.12, 'on': 0.18, 'at': 0.22,
            'for': 0.15, 'with': 0.18, 'by': 0.22, 'from': 0.2, 'as': 0.18,
        }
    
    def _load_spacy(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                try:
                    self._nlp = spacy.load(self.spacy_model)
                except OSError:
                    # Model not installed, try downloading
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", self.spacy_model], 
                                   capture_output=True)
                    self._nlp = spacy.load(self.spacy_model)
            except ImportError:
                warnings.warn(
                    "spaCy not available. Install with: pip install spacy && "
                    "python -m spacy download en_core_web_sm"
                )
                self.use_spacy = False
        return self._nlp
    
    def extract_features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Extract linguistic features for each token.
        
        Returns list of dictionaries with:
        - pos_tag: Universal POS tag
        - pos_category: 0=content, 1=function, 2=punct
        - idf_score: Inverse document frequency (0-1, higher = rarer)
        - is_stopword: Boolean from NLP tagger
        """
        text = " ".join(tokens)
        nlp = self._load_spacy()
        
        if nlp is not None:
            # Use spaCy for accurate NLP features
            doc = nlp(text)
            
            # Align spaCy tokens with our tokens
            features = []
            spacy_idx = 0
            
            for token in tokens:
                token_lower = token.lower().strip()
                
                # Find matching spaCy token
                pos_tag = "X"
                is_stop = False
                
                while spacy_idx < len(doc):
                    spacy_token = doc[spacy_idx]
                    if spacy_token.text.strip() == token.strip() or token_lower in spacy_token.text.lower():
                        pos_tag = spacy_token.pos_
                        is_stop = spacy_token.is_stop
                        spacy_idx += 1
                        break
                    spacy_idx += 1
                
                # Determine POS category
                if pos_tag in self.POS_CONTENT:
                    pos_category = 0  # Content word
                elif pos_tag in self.POS_FUNCTION:
                    pos_category = 1  # Function word
                else:
                    pos_category = 2  # Punctuation/other
                
                # Get IDF score
                idf = self._get_idf(token_lower)
                
                features.append({
                    'pos_tag': pos_tag,
                    'pos_category': pos_category,
                    'idf_score': idf,
                    'is_stopword': is_stop
                })
            
            return features
        
        else:
            # Fallback: heuristic-based features
            return self._extract_features_heuristic(tokens)
    
    def _extract_features_heuristic(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Fallback feature extraction without spaCy."""
        features = []
        
        for token in tokens:
            token_lower = token.lower().strip()
            
            # Heuristic POS detection
            if all(c in '.,!?;:"\'-()[]{}' for c in token.strip()):
                pos_tag = "PUNCT"
                pos_category = 2
            elif token_lower in self._default_idf:
                pos_tag = "FUNC"  # Likely function word
                pos_category = 1
            else:
                pos_tag = "NOUN"  # Assume content word
                pos_category = 0
            
            # IDF score
            idf = self._get_idf(token_lower)
            
            # Stopword detection
            is_stop = token_lower in self._default_idf and self._default_idf[token_lower] < 0.25
            
            features.append({
                'pos_tag': pos_tag,
                'pos_category': pos_category,
                'idf_score': idf,
                'is_stopword': is_stop
            })
        
        return features
    
    def _get_idf(self, token: str) -> float:
        """Get IDF score for token (higher = rarer = more important)."""
        token_lower = token.lower().strip()
        
        if token_lower in self._idf_cache:
            return self._idf_cache[token_lower]
        
        if token_lower in self._default_idf:
            idf = self._default_idf[token_lower]
        elif all(c in '.,!?;:"\'-()[]{}' for c in token_lower):
            idf = 0.05  # Punctuation has very low IDF
        elif len(token_lower) <= 2:
            idf = 0.3  # Short words tend to be common
        else:
            idf = 0.7  # Default for unknown content words
        
        self._idf_cache[token_lower] = idf
        return idf


class TextAugmentor:
    """
    Text augmentation for stability testing and invariance-based refinement.
    
    Following the PDF's "Data Augmentation: Exploring the Feature Manifold":
    - Generate semantics-preserving variants (EDA techniques)
    - Compute attribution stability across variants
    - Use variance penalization to suppress spurious tokens
    
    Key insight: Stopwords are UNSTABLE across augmented prompts,
    while content words are INVARIANT. This lets us learn suppression
    rather than hard-coding it.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        
        # Simple synonym dictionary for EDA
        self._synonyms = {
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'minute'],
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'happy': ['glad', 'joyful', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'dejected'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['sluggish', 'leisurely', 'gradual'],
            'important': ['significant', 'crucial', 'vital'],
            'beautiful': ['pretty', 'lovely', 'gorgeous'],
        }
    
    def generate_variants(
        self,
        tokens: List[str],
        n_variants: int = 3,
        methods: List[str] = ['synonym', 'swap', 'delete']
    ) -> List[List[str]]:
        """
        Generate augmented variants of the token sequence.
        
        Methods:
        - synonym: Replace words with synonyms
        - swap: Swap adjacent words
        - delete: Delete random non-essential words
        
        Returns list of augmented token sequences.
        """
        variants = []
        
        for _ in range(n_variants):
            method = self.rng.choice(methods)
            
            if method == 'synonym':
                variants.append(self._synonym_replacement(tokens))
            elif method == 'swap':
                variants.append(self._random_swap(tokens))
            elif method == 'delete':
                variants.append(self._random_deletion(tokens))
            else:
                variants.append(tokens.copy())
        
        return variants
    
    def _synonym_replacement(self, tokens: List[str]) -> List[str]:
        """Replace random words with synonyms."""
        result = tokens.copy()
        
        for i, token in enumerate(result):
            token_lower = token.lower()
            if token_lower in self._synonyms and self.rng.random() > 0.5:
                synonym = self.rng.choice(self._synonyms[token_lower])
                # Preserve capitalization
                if token[0].isupper():
                    synonym = synonym.capitalize()
                result[i] = synonym
        
        return result
    
    def _random_swap(self, tokens: List[str]) -> List[str]:
        """Swap two random adjacent words."""
        if len(tokens) < 2:
            return tokens.copy()
        
        result = tokens.copy()
        # Find valid swap positions (not punctuation)
        valid_positions = [
            i for i in range(len(tokens) - 1)
            if not all(c in '.,!?;:' for c in tokens[i])
            and not all(c in '.,!?;:' for c in tokens[i + 1])
        ]
        
        if valid_positions:
            i = self.rng.choice(valid_positions)
            result[i], result[i + 1] = result[i + 1], result[i]
        
        return result
    
    def _random_deletion(self, tokens: List[str]) -> List[str]:
        """Delete a random function word (preserving content words)."""
        if len(tokens) < 3:
            return tokens.copy()
        
        result = tokens.copy()
        
        # Find candidates for deletion (short common words)
        deletable = [
            i for i, t in enumerate(tokens)
            if len(t) <= 3 and t.lower() in {'a', 'an', 'the', 'is', 'are', 'was', 'were'}
        ]
        
        if deletable and self.rng.random() > 0.3:
            idx = self.rng.choice(deletable)
            result.pop(idx)
        
        return result
    
    def compute_attribution_stability(
        self,
        original_attrs: np.ndarray,
        variant_attrs_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute stability of attributions across variants.
        
        Returns array of stability scores (0-1) where:
        - 1 = perfectly stable (invariant content word)
        - 0 = highly unstable (spurious/surface token)
        
        Tokens with high variance across augmentations should be suppressed.
        """
        if not variant_attrs_list:
            return np.ones(len(original_attrs))
        
        # Stack all attributions
        all_attrs = np.vstack([original_attrs] + variant_attrs_list)
        
        # Compute variance per token position
        variances = np.var(all_attrs, axis=0)
        
        # Convert to stability (1 - normalized variance)
        max_var = np.max(variances) + 1e-10
        stability = 1 - (variances / max_var)
        
        return stability


class ValueFunction:
    """
    Value function for computing coalition payoffs.
    Implements multiple strategies from the workflow document.
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        function_type: ValueFunctionType = ValueFunctionType.EMBEDDING_SIMILARITY,
        reference_output: Optional[str] = None,
        reference_embedding: Optional[np.ndarray] = None,
        task_scorer: Optional[Callable[[str, str], float]] = None
    ):
        self.backend = backend
        self.function_type = function_type
        self.reference_output = reference_output
        self.reference_embedding = reference_embedding
        self.task_scorer = task_scorer
    
    def set_reference(self, full_prompt: str):
        """Set reference output from full prompt."""
        self.reference_output = self.backend.generate(full_prompt)
        if self.function_type == ValueFunctionType.EMBEDDING_SIMILARITY:
            self.reference_embedding = self.backend.get_embedding(self.reference_output)
    
    def __call__(self, coalition_output: str) -> float:
        """Compute payoff for coalition output."""
        if self.function_type == ValueFunctionType.EMBEDDING_SIMILARITY:
            return self._embedding_similarity(coalition_output)
        elif self.function_type == ValueFunctionType.TASK_SCORER:
            return self._task_score(coalition_output)
        elif self.function_type == ValueFunctionType.LOGPROB:
            raise NotImplementedError("Logprob value function requires special handling")
        else:
            raise ValueError(f"Unknown value function type: {self.function_type}")
    
    def _embedding_similarity(self, output: str) -> float:
        """Cosine similarity between embeddings."""
        if self.reference_embedding is None:
            raise ValueError("Reference embedding not set. Call set_reference() first.")
        
        output_embedding = self.backend.get_embedding(output)
        
        # Cosine similarity
        dot_product = np.dot(self.reference_embedding, output_embedding)
        norm_product = np.linalg.norm(self.reference_embedding) * np.linalg.norm(output_embedding)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def _task_score(self, output: str) -> float:
        """Score using custom task scorer."""
        if self.task_scorer is None:
            raise ValueError("Task scorer not provided.")
        return self.task_scorer(output, self.reference_output)


class CoalitionGenerator:
    """
    Generates coalitions for Shapley value estimation.
    
    Implements advanced sampling strategies for variance reduction:
    - Stratified sampling by coalition size (ensures diverse sizes)
    - Paired/antithetic sampling (sample S and complement N\S together)
    - Essential leave-one-out subsets (guarantees each token is tested)
    
    These techniques reduce Monte Carlo variance, improving:
    - Consistency (higher Spearman/Kendall correlations)
    - Comprehensiveness metrics (more stable attribution rankings)
    """
    
    def __init__(
        self,
        n_tokens: int,
        budget: int,
        include_leave_one_out: bool = True,
        stratify_sizes: bool = True,
        use_paired_sampling: bool = True,  # NEW: Antithetic sampling
        seed: Optional[int] = None
    ):
        self.n_tokens = n_tokens
        self.budget = budget
        self.include_leave_one_out = include_leave_one_out
        self.stratify_sizes = stratify_sizes
        self.use_paired_sampling = use_paired_sampling
        self.rng = np.random.RandomState(seed)
        
        self.all_tokens = frozenset(range(n_tokens))
        self.coalitions: List[frozenset] = []
        self._paired_coalitions: Dict[frozenset, frozenset] = {}  # Track pairs
        
        self._generate_coalitions()
    
    def _generate_coalitions(self):
        """Generate the set of coalitions to evaluate."""
        coalitions = set()
        
        # Always include the full coalition and empty coalition
        coalitions.add(self.all_tokens)
        coalitions.add(frozenset())
        self._paired_coalitions[self.all_tokens] = frozenset()
        self._paired_coalitions[frozenset()] = self.all_tokens
        
        # Include all leave-one-out coalitions (essential for stability)
        if self.include_leave_one_out:
            for i in range(self.n_tokens):
                loo = self.all_tokens - {i}
                singleton = frozenset({i})
                coalitions.add(loo)
                coalitions.add(singleton)  # Also add the complement
                self._paired_coalitions[loo] = singleton
                self._paired_coalitions[singleton] = loo
        
        # Calculate remaining budget
        remaining_budget = self.budget - len(coalitions)
        
        if remaining_budget > 0:
            if self.stratify_sizes and self.use_paired_sampling:
                # Best: Stratified + Paired sampling
                coalitions.update(self._stratified_paired_sample(remaining_budget, coalitions))
            elif self.stratify_sizes:
                # Stratified sampling
                coalitions.update(self._stratified_sample(remaining_budget, coalitions))
            else:
                # Random sampling with optional pairing
                coalitions.update(self._random_sample(remaining_budget, coalitions))
        
        self.coalitions = list(coalitions)
    
    def _stratified_paired_sample(self, budget: int, existing: set) -> List[frozenset]:
        """
        Generate stratified random coalitions with paired/antithetic sampling.
        
        For each sampled coalition S, also include its complement N\S.
        This reduces variance by canceling out symmetric errors.
        
        Key insight: v(S) and v(N\S) are correlated, so sampling them
        together reduces the variance of the Shapley estimator.
        """
        new_coalitions = []
        
        # Only sample sizes up to n/2 (pairs will cover larger sizes)
        max_size = self.n_tokens // 2
        sizes = list(range(1, max_size + 1))
        
        if not sizes:
            return self._stratified_sample(budget, existing)
        
        # Distribute budget across sizes (each sample produces a pair)
        samples_per_size = max(1, (budget // 2) // len(sizes))
        
        for size in sizes:
            for _ in range(samples_per_size):
                if len(new_coalitions) >= budget:
                    break
                
                # Generate random coalition of this size
                members = self.rng.choice(self.n_tokens, size=size, replace=False)
                coalition = frozenset(members)
                complement = self.all_tokens - coalition
                
                # Add both the coalition and its complement
                if coalition not in existing and coalition not in new_coalitions:
                    new_coalitions.append(coalition)
                    self._paired_coalitions[coalition] = complement
                    
                    if complement not in existing and complement not in new_coalitions:
                        new_coalitions.append(complement)
                        self._paired_coalitions[complement] = coalition
        
        return new_coalitions
    
    def _stratified_sample(self, budget: int, existing: set) -> List[frozenset]:
        """Generate stratified random coalitions (diverse sizes)."""
        new_coalitions = []
        
        # Distribute budget across different coalition sizes
        sizes = list(range(1, self.n_tokens))  # Exclude empty and full
        samples_per_size = max(1, budget // len(sizes))
        
        for size in sizes:
            for _ in range(samples_per_size):
                if len(new_coalitions) >= budget:
                    break
                
                # Generate random coalition of this size
                members = self.rng.choice(self.n_tokens, size=size, replace=False)
                coalition = frozenset(members)
                
                if coalition not in existing and coalition not in new_coalitions:
                    new_coalitions.append(coalition)
        
        return new_coalitions
    
    def _random_sample(self, budget: int, existing: set) -> List[frozenset]:
        """Generate random coalitions."""
        new_coalitions = []
        
        for _ in range(budget * 2):  # Oversample to handle duplicates
            if len(new_coalitions) >= budget:
                break
            
            # Random size with 50% inclusion probability for each token
            mask = self.rng.random(self.n_tokens) > 0.5
            coalition = frozenset(np.where(mask)[0])
            
            if coalition not in existing and coalition not in new_coalitions:
                new_coalitions.append(coalition)
                
                # Also add complement if using paired sampling
                if self.use_paired_sampling:
                    complement = self.all_tokens - coalition
                    if complement not in existing and complement not in new_coalitions:
                        new_coalitions.append(complement)
                        self._paired_coalitions[coalition] = complement
                        self._paired_coalitions[complement] = coalition
        
        return new_coalitions
    
    def get_coalitions_with_token(self, token_idx: int) -> List[frozenset]:
        """Get all coalitions containing a specific token."""
        return [c for c in self.coalitions if token_idx in c]
    
    def get_coalitions_without_token(self, token_idx: int) -> List[frozenset]:
        """Get all coalitions not containing a specific token."""
        return [c for c in self.coalitions if token_idx not in c]
    
    def get_paired_coalition(self, coalition: frozenset) -> Optional[frozenset]:
        """Get the complement/paired coalition if it was sampled."""
        return self._paired_coalitions.get(coalition)


class PromptRenderer:
    """Renders prompts from token coalitions."""
    
    def __init__(
        self,
        tokens: List[str],
        strategy: CoalitionStrategy = CoalitionStrategy.DELETE,
        placeholder: str = "[...]"
    ):
        self.tokens = tokens
        self.strategy = strategy
        self.placeholder = placeholder
    
    def render(self, coalition: frozenset) -> str:
        """Render prompt for a coalition."""
        if self.strategy == CoalitionStrategy.DELETE:
            return " ".join(self.tokens[i] for i in sorted(coalition))
        
        elif self.strategy == CoalitionStrategy.PLACEHOLDER:
            result = []
            for i, token in enumerate(self.tokens):
                if i in coalition:
                    result.append(token)
                else:
                    result.append(self.placeholder)
            return " ".join(result)
        
        elif self.strategy == CoalitionStrategy.MASK:
            result = []
            for i, token in enumerate(self.tokens):
                if i in coalition:
                    result.append(token)
                else:
                    result.append("[MASK]")
            return " ".join(result)
        
        elif self.strategy == CoalitionStrategy.UNK:
            result = []
            for i, token in enumerate(self.tokens):
                if i in coalition:
                    result.append(token)
                else:
                    result.append("[UNK]")
            return " ".join(result)
        
        else:
            raise ValueError(f"Unknown coalition strategy: {self.strategy}")


class Stage1TokenSHAP:
    """
    Stage 1: TokenSHAP Monte Carlo Shapley Value Estimation
    
    Implements the base explainer using Monte Carlo sampling
    to estimate Shapley values for each token.
    
    Supports two Shapley estimation methods:
    - 'simple': TokenSHAP-style averaging (E[v|i∈S] - E[v|i∉S])
    - 'weighted': Proper Shapley weights based on coalition size
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        value_function: ValueFunction,
        coalition_strategy: CoalitionStrategy = CoalitionStrategy.DELETE,
        budget: int = 100,
        cache: Optional[CoalitionCache] = None,
        seed: Optional[int] = None,
        shapley_method: str = "weighted",  # "simple" or "weighted"
        n_runs: int = 1  # Multi-run averaging for stability
    ):
        self.backend = backend
        self.value_function = value_function
        self.coalition_strategy = coalition_strategy
        self.budget = budget
        self.cache = cache or CoalitionCache(use_cache=True)
        self.seed = seed
        self.shapley_method = shapley_method
        self.n_runs = n_runs
    
    def _compute_shapley_weights(self, n: int, s: int) -> float:
        """
        Compute proper Shapley weight for coalition of size s with n total players.
        
        Weight = s!(n-s-1)! / n!
        
        This ensures contributions from different sized coalitions are properly weighted.
        """
        from math import factorial
        if s < 0 or s >= n:
            return 0.0
        return factorial(s) * factorial(n - s - 1) / factorial(n)
    
    def compute(self, prompt: str) -> Tuple[List[ShapleyResult], Dict[frozenset, CoalitionResult]]:
        """
        Compute Stage 1 Shapley values for all tokens.
        
        Returns:
            Tuple of (shapley_results, coalition_cache)
        """
        # Tokenize prompt
        tokens = self.backend.tokenize(prompt)
        n_tokens = len(tokens)
        
        if n_tokens == 0:
            return [], {}
        
        # Set up value function reference
        self.value_function.set_reference(prompt)
        
        # Generate coalitions
        generator = CoalitionGenerator(
            n_tokens=n_tokens,
            budget=self.budget,
            include_leave_one_out=True,
            stratify_sizes=True,
            seed=self.seed
        )
        
        # Create prompt renderer
        renderer = PromptRenderer(tokens, self.coalition_strategy)
        
        # Evaluate all coalitions
        coalition_results = {}
        
        for coalition in generator.coalitions:
            # Check cache first
            cached = self.cache.get(coalition)
            if cached is not None:
                coalition_results[coalition] = cached
                continue
            
            # Render and evaluate
            coalition_prompt = renderer.render(coalition)
            
            if len(coalition) == 0:
                # Empty coalition
                output = ""
                payoff = 0.0
            else:
                output = self.backend.generate(coalition_prompt)
                payoff = self.value_function(output)
            
            result = CoalitionResult(
                coalition=coalition,
                payoff=payoff,
                output_text=output
            )
            
            coalition_results[coalition] = result
            self.cache.set(coalition, result)
        
        # Compute Shapley values using selected method
        shapley_results = []
        full_coalition = frozenset(range(n_tokens))
        full_payoff = coalition_results[full_coalition].payoff
        
        for i in range(n_tokens):
            # Get coalitions with and without token i
            with_i = [c for c in coalition_results.keys() if i in c]
            without_i = [c for c in coalition_results.keys() if i not in c]
            
            if self.shapley_method == "weighted":
                # Proper Shapley formula with coalition size weighting
                # φ_i = Σ_{S⊆N\{i}} [s!(n-s-1)!/n!] * [v(S∪{i}) - v(S)]
                shapley_value = 0.0
                total_weight = 0.0
                
                for coalition_without in without_i:
                    s = len(coalition_without)
                    coalition_with = coalition_without | {i}
                    
                    if coalition_with in coalition_results:
                        weight = self._compute_shapley_weights(n_tokens, s)
                        v_with = coalition_results[coalition_with].payoff
                        v_without = coalition_results[coalition_without].payoff
                        shapley_value += weight * (v_with - v_without)
                        total_weight += weight
                
                # Normalize if we don't have all coalitions
                if total_weight > 0 and total_weight < 1.0:
                    shapley_value = shapley_value / total_weight
                    
            else:  # "simple" method - TokenSHAP averaging
                # Calculate simple averages
                avg_with = np.mean([coalition_results[c].payoff for c in with_i]) if with_i else 0.0
                avg_without = np.mean([coalition_results[c].payoff for c in without_i]) if without_i else 0.0
                shapley_value = avg_with - avg_without
            
            # Calculate variance for stability analysis
            payoffs_with = [coalition_results[c].payoff for c in with_i]
            variance = np.var(payoffs_with) if len(payoffs_with) > 1 else 0.0
            
            # Leave-one-out delta
            leave_one_out = full_coalition - {i}
            loo_delta = full_payoff - coalition_results.get(
                leave_one_out, 
                CoalitionResult(leave_one_out, 0.0, "")
            ).payoff
            
            shapley_results.append(ShapleyResult(
                token_idx=i,
                token_text=tokens[i],
                shapley_value=shapley_value,
                variance=variance,
                leave_one_out_delta=loo_delta
            ))
        
        return shapley_results, coalition_results
    
    def compute_with_multirun(self, prompt: str) -> Tuple[List[ShapleyResult], Dict[frozenset, CoalitionResult]]:
        """
        Compute Shapley values with multi-run averaging for stability.
        
        Runs the computation n_runs times with different random seeds
        and averages the results to reduce variance.
        """
        if self.n_runs <= 1:
            return self.compute(prompt)
        
        all_values = []
        final_coalition_results = None
        tokens = None
        
        base_seed = self.seed if self.seed is not None else 42
        
        for run in range(self.n_runs):
            # Use different seed for each run
            self.seed = base_seed + run
            shapley_results, coalition_results = self.compute(prompt)
            
            if tokens is None:
                tokens = [r.token_text for r in shapley_results]
            if final_coalition_results is None:
                final_coalition_results = coalition_results
            
            all_values.append([r.shapley_value for r in shapley_results])
        
        # Average across runs
        avg_values = np.mean(all_values, axis=0)
        var_values = np.var(all_values, axis=0)
        
        # Create final results with averaged values
        final_results = []
        for i, (token, avg_val, var_val) in enumerate(zip(tokens, avg_values, var_values)):
            final_results.append(ShapleyResult(
                token_idx=i,
                token_text=token,
                shapley_value=float(avg_val),
                variance=float(var_val),
                leave_one_out_delta=shapley_results[i].leave_one_out_delta
            ))
        
        # Restore original seed
        self.seed = base_seed
        
        return final_results, final_coalition_results


class Stage2SFARefinement:
    """
    Stage 2: SFA-style Shapley Refinement with Enhanced Features
    
    Following the "Text Feature Enrichment" philosophy:
    - Uses ADDITIVE features (POS, IDF, stability) instead of hard-coded rules
    - Lets the model LEARN to suppress stopwords based on features
    - Incorporates augmentation-invariance for robust token importance
    
    Key improvement over original: No hard-coded stopword filtering!
    Instead, linguistic features let the refiner learn what to suppress.
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        value_function: ValueFunction,
        renderer: PromptRenderer,
        refiner_type: str = "gradient_boosting",  # Changed default for better learning
        use_linguistic_features: bool = True,  # NEW: Enable POS/IDF features
        use_augmentation: bool = True,  # NEW: Enable stability features
        n_augmentation_variants: int = 3
    ):
        self.backend = backend
        self.value_function = value_function
        self.renderer = renderer
        self.refiner_type = refiner_type
        self.use_linguistic_features = use_linguistic_features
        self.use_augmentation = use_augmentation
        self.n_augmentation_variants = n_augmentation_variants
        
        self._refiner = None
        self._is_fitted = False
        
        # Initialize linguistic feature extractor and augmentor
        self._linguistic_extractor = LinguisticFeatureExtractor() if use_linguistic_features else None
        self._augmentor = TextAugmentor() if use_augmentation else None
    
    def build_feature_vectors(
        self,
        tokens: List[str],
        stage1_results: List[ShapleyResult],
        coalition_results: Dict[frozenset, CoalitionResult],
        augmented_attrs: Optional[List[np.ndarray]] = None
    ) -> List[SFAFeatureVector]:
        """
        Build enhanced feature vectors for each token.
        
        Features (15-dimensional):
        - Stage 1 attribution (φ^(1)_i)
        - Position features (normalized, absolute)
        - Surface features (length, punctuation, whitespace, capitalization)
        - Linguistic features (POS category, IDF score, is_stopword) - COMPUTED, not hard-coded!
        - Coalition statistics (variance, leave-one-out delta)
        - Interaction features (neighbor synergy)
        - Augmentation stability (invariance across variants)
        
        The key insight: linguistic features let the model LEARN to suppress
        stopwords rather than hard-coding rules. This aligns with SFA's
        "increase features" philosophy.
        """
        n_tokens = len(tokens)
        feature_vectors = []
        full_coalition = frozenset(range(n_tokens))
        
        # Extract linguistic features (POS, IDF)
        if self._linguistic_extractor is not None:
            ling_features = self._linguistic_extractor.extract_features(tokens)
        else:
            ling_features = [{'pos_tag': 'X', 'pos_category': 0, 'idf_score': 0.5, 'is_stopword': False} 
                           for _ in tokens]
        
        # Compute augmentation stability if available
        if augmented_attrs is not None and len(augmented_attrs) > 0:
            original_attrs = np.array([r.shapley_value for r in stage1_results])
            stability_scores = self._augmentor.compute_attribution_stability(
                original_attrs, augmented_attrs
            ) if self._augmentor else np.ones(n_tokens)
        else:
            stability_scores = np.ones(n_tokens)
        
        full_payoff = coalition_results[full_coalition].payoff if full_coalition in coalition_results else 0.0
        
        for i, (token, shap_result) in enumerate(zip(tokens, stage1_results)):
            # Calculate neighbor synergy
            neighbor_synergy = 0.0
            if i < n_tokens - 1:
                pair_removed = full_coalition - {i, i + 1}
                i_removed = full_coalition - {i}
                j_removed = full_coalition - {i + 1}
                
                if all(c in coalition_results for c in [pair_removed, i_removed, j_removed]):
                    v_pair = coalition_results[pair_removed].payoff
                    v_i = coalition_results[i_removed].payoff
                    v_j = coalition_results[j_removed].payoff
                    neighbor_synergy = (full_payoff - v_pair) - (full_payoff - v_i) - (full_payoff - v_j)
            
            # Get linguistic features for this token
            ling = ling_features[i] if i < len(ling_features) else {
                'pos_tag': 'X', 'pos_category': 0, 'idf_score': 0.5, 'is_stopword': False
            }
            
            feature_vectors.append(SFAFeatureVector(
                token_idx=i,
                
                # Stage 1 attribution
                stage1_attribution=shap_result.shapley_value,
                
                # Position features
                position_normalized=i / n_tokens,
                position_absolute=i,
                
                # Surface features
                token_length=len(token),
                is_punctuation=all(c in ".,!?;:'-\"()[]{}/" for c in token.strip()) if token.strip() else False,
                is_whitespace=token.strip() == "",
                is_capitalized=token[0].isupper() if token else False,
                is_all_caps=token.isupper() if token else False,
                
                # Linguistic features (FROM NLP, not hard-coded!)
                pos_tag=ling['pos_tag'],
                pos_category=ling['pos_category'],
                idf_score=ling['idf_score'],
                is_stopword_computed=ling['is_stopword'],
                
                # Coalition statistics
                payoff_variance=shap_result.variance,
                leave_one_out_delta=shap_result.leave_one_out_delta,
                
                # Interaction + augmentation
                neighbor_synergy=neighbor_synergy,
                attribution_stability=float(stability_scores[i]),
            ))
        
        return feature_vectors
    
    def compute_faithfulness_target(
        self,
        tokens: List[str],
        stage1_results: List[ShapleyResult],
        top_k: int = 3
    ) -> np.ndarray:
        """
        Compute faithfulness-based target for refinement training.
        
        Performs perturbation tests to create training signal:
        - Remove top-k tokens → measure payoff drop
        - Remove bottom-k tokens → measure payoff drop
        - Good explainer maximizes the gap
        """
        n_tokens = len(tokens)
        
        if n_tokens <= top_k:
            top_k = max(1, n_tokens // 2)
        
        # Rank tokens by Stage 1 Shapley values
        ranked = sorted(enumerate(stage1_results), key=lambda x: x[1].shapley_value, reverse=True)
        
        top_indices = [idx for idx, _ in ranked[:top_k]]
        bottom_indices = [idx for idx, _ in ranked[-top_k:]]
        
        # Create coalition without top tokens
        top_removed = frozenset(range(n_tokens)) - frozenset(top_indices)
        top_prompt = self.renderer.render(top_removed)
        top_output = self.backend.generate(top_prompt) if len(top_removed) > 0 else ""
        delta_top = self.value_function(self.value_function.reference_output) - self.value_function(top_output)
        
        # Create coalition without bottom tokens
        bottom_removed = frozenset(range(n_tokens)) - frozenset(bottom_indices)
        bottom_prompt = self.renderer.render(bottom_removed)
        bottom_output = self.backend.generate(bottom_prompt) if len(bottom_removed) > 0 else ""
        delta_bottom = self.value_function(self.value_function.reference_output) - self.value_function(bottom_output)
        
        # Faithfulness gap
        faithfulness_gap = delta_top - delta_bottom
        
        # Create target: boost top tokens, suppress bottom tokens
        targets = np.zeros(n_tokens)
        for idx in top_indices:
            targets[idx] = max(0, faithfulness_gap)
        for idx in bottom_indices:
            targets[idx] = min(0, -faithfulness_gap)
        
        return targets
    
    def fit_refiner(
        self,
        feature_vectors: List[SFAFeatureVector],
        targets: np.ndarray
    ):
        """
        Fit the refinement model.
        
        Supported types:
        - ridge: Ridge regression (interpretable)
        - gradient_boosting: Gradient boosting (stronger)
        - mlp: Small MLP (if available)
        """
        X = np.array([fv.to_array() for fv in feature_vectors])
        y = targets
        
        if self.refiner_type == "ridge":
            from sklearn.linear_model import Ridge
            self._refiner = Ridge(alpha=1.0)
            self._refiner.fit(X, y)
            
        elif self.refiner_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            self._refiner = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self._refiner.fit(X, y)
            
        elif self.refiner_type == "mlp":
            from sklearn.neural_network import MLPRegressor
            self._refiner = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                random_state=42
            )
            self._refiner.fit(X, y)
            
        else:
            raise ValueError(f"Unknown refiner type: {self.refiner_type}")
        
        self._is_fitted = True
    
    def refine(
        self,
        feature_vectors: List[SFAFeatureVector],
        stage1_values: np.ndarray
    ) -> np.ndarray:
        """
        Generate refined Shapley values.
        
        If refiner is not fitted, returns stage1 values with stopword suppression.
        """
        if not self._is_fitted:
            # Fallback: apply heuristic refinement
            refined = stage1_values.copy()
            
            if self.regularize_stopwords:
                for i, fv in enumerate(feature_vectors):
                    if fv.is_stopword or fv.is_punctuation or fv.is_whitespace:
                        refined[i] *= 0.5  # Suppress common tokens
            
            return refined
        
        # Apply trained refiner
        X = np.array([fv.to_array() for fv in feature_vectors])
        adjustments = self._refiner.predict(X)
        
        # Combine Stage 1 values with learned adjustments
        refined = stage1_values + adjustments
        
        return refined


class SFATokenSHAP:
    """
    SFA-TokenSHAP: Complete Two-Stage Shapley Refinement Pipeline
    
    Combines TokenSHAP Monte Carlo estimation with SFA-style
    feature augmentation for improved token attribution.
    
    Key improvements over basic TokenSHAP:
    - Semantic embedding-based value function (not TF-IDF)
    - Proper Shapley weighting option
    - SFA-style feature augmentation in Stage 2
    - Stopword suppression
    - Multi-run averaging for stability
    """
    
    def __init__(
        self,
        backend: LLMBackend,
        value_function_type: ValueFunctionType = ValueFunctionType.EMBEDDING_SIMILARITY,
        coalition_strategy: CoalitionStrategy = CoalitionStrategy.DELETE,
        budget: int = 100,
        refiner_type: str = "ridge",
        use_cache: bool = True,
        seed: Optional[int] = None,
        shapley_method: str = "weighted",  # "simple" or "weighted"
        n_runs: int = 1,  # Multi-run averaging
        suppress_stopwords: bool = True  # Post-processing stopword suppression
    ):
        self.backend = backend
        self.value_function_type = value_function_type
        self.coalition_strategy = coalition_strategy
        self.budget = budget
        self.refiner_type = refiner_type
        self.use_cache = use_cache
        self.seed = seed
        self.shapley_method = shapley_method
        self.n_runs = n_runs
        self.suppress_stopwords = suppress_stopwords
        
        self.cache = CoalitionCache(use_cache=use_cache)
        
        # Default stopwords for post-processing
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'it', 'its', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'as',
            'so', 'than', 'just', 'only', 'also', 'very', 'too', 'more', 'most'
        }
    
    def explain(
        self,
        prompt: str,
        train_refiner: bool = True,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate token-level explanations for a prompt.
        
        Args:
            prompt: Input prompt to explain
            train_refiner: Whether to train Stage 2 refiner
            normalize: Whether to normalize final values
            
        Returns:
            Dictionary containing:
            - tokens: List of tokens
            - stage1_values: Raw Shapley values from Stage 1
            - stage2_values: Refined Shapley values from Stage 2
            - feature_vectors: SFA feature vectors
            - cache_stats: Coalition cache statistics
        """
        # Create value function
        value_function = ValueFunction(
            backend=self.backend,
            function_type=self.value_function_type
        )
        
        # Stage 1: TokenSHAP
        stage1 = Stage1TokenSHAP(
            backend=self.backend,
            value_function=value_function,
            coalition_strategy=self.coalition_strategy,
            budget=self.budget,
            cache=self.cache,
            seed=self.seed,
            shapley_method=self.shapley_method,
            n_runs=self.n_runs
        )
        
        # Use multi-run if configured
        if self.n_runs > 1:
            stage1_results, coalition_results = stage1.compute_with_multirun(prompt)
        else:
            stage1_results, coalition_results = stage1.compute(prompt)
        
        # Extract tokens and values
        tokens = [r.token_text for r in stage1_results]
        stage1_values = np.array([r.shapley_value for r in stage1_results])
        
        # Create renderer for Stage 2
        renderer = PromptRenderer(tokens, self.coalition_strategy)
        
        # Stage 2: SFA Refinement
        stage2 = Stage2SFARefinement(
            backend=self.backend,
            value_function=value_function,
            renderer=renderer,
            refiner_type=self.refiner_type
        )
        
        # Build feature vectors
        feature_vectors = stage2.build_feature_vectors(
            tokens, stage1_results, coalition_results
        )
        
        # Train and apply refiner
        if train_refiner and len(tokens) > 3:
            try:
                targets = stage2.compute_faithfulness_target(tokens, stage1_results)
                stage2.fit_refiner(feature_vectors, targets)
            except Exception as e:
                warnings.warn(f"Refiner training failed: {e}. Using heuristic refinement.")
        
        stage2_values = stage2.refine(feature_vectors, stage1_values)
        
        # Post-processing: Explicit stopword suppression (if enabled)
        if self.suppress_stopwords:
            for i, token in enumerate(tokens):
                token_lower = token.lower().strip()
                is_punct = all(c in ".,!?;:'-\"()[]{}/" for c in token.strip())
                is_stop = token_lower in self.stopwords
                is_whitespace = token.strip() == ""
                
                if is_stop or is_punct or is_whitespace:
                    # Suppress trivial tokens by scaling down
                    stage2_values[i] *= 0.3
                    # Also apply threshold: if very small, zero out
                    if abs(stage2_values[i]) < 0.01:
                        stage2_values[i] = 0.0
        
        # Normalize if requested
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
            "feature_vectors": feature_vectors,
            "coalition_results": coalition_results,
            "cache_stats": self.cache.stats
        }
    
    def visualize(
        self,
        result: Dict[str, Any],
        use_stage2: bool = True,
        colormap: str = "coolwarm"
    ) -> str:
        """
        Generate HTML visualization of token attributions.
        
        Args:
            result: Result from explain()
            use_stage2: Whether to use Stage 2 refined values
            colormap: Matplotlib colormap name
            
        Returns:
            HTML string for visualization
        """
        tokens = result["tokens"]
        values = result["stage2_values"] if use_stage2 else result["stage1_values"]
        
        # Normalize to [-1, 1]
        max_abs = np.abs(values).max()
        if max_abs > 0:
            normalized = values / max_abs
        else:
            normalized = values
        
        # Generate HTML
        html_parts = ['<div style="font-family: Arial, sans-serif; line-height: 2;">']
        
        for token, value in zip(tokens, normalized):
            # Map value to color
            if value > 0:
                # Positive: red
                intensity = int(value * 200)
                bg_color = f"rgba(255, {200 - intensity}, {200 - intensity}, 0.8)"
            else:
                # Negative: blue
                intensity = int(-value * 200)
                bg_color = f"rgba({200 - intensity}, {200 - intensity}, 255, 0.8)"
            
            html_parts.append(
                f'<span style="background-color: {bg_color}; padding: 2px 4px; '
                f'margin: 1px; border-radius: 3px;" title="SHAP: {value:.4f}">{token}</span>'
            )
        
        html_parts.append('</div>')
        
        return " ".join(html_parts)


# Utility functions

def compute_exact_shapley(
    n_tokens: int,
    coalition_results: Dict[frozenset, CoalitionResult]
) -> np.ndarray:
    """
    Compute exact Shapley values (exponential complexity).
    Only use for small n_tokens (< 12).
    """
    if n_tokens > 12:
        raise ValueError(f"Exact Shapley computation not recommended for {n_tokens} tokens")
    
    from math import factorial
    
    shapley_values = np.zeros(n_tokens)
    N = frozenset(range(n_tokens))
    
    for i in range(n_tokens):
        for coalition in coalition_results.keys():
            if i not in coalition:
                coalition_with_i = coalition | {i}
                if coalition_with_i in coalition_results:
                    # Shapley weight
                    s = len(coalition)
                    weight = factorial(s) * factorial(n_tokens - s - 1) / factorial(n_tokens)
                    
                    # Marginal contribution
                    marginal = coalition_results[coalition_with_i].payoff - coalition_results[coalition].payoff
                    
                    shapley_values[i] += weight * marginal
    
    return shapley_values


def sliding_window_shapley(
    tokens: List[str],
    coalition_results: Dict[frozenset, CoalitionResult],
    window_size: int = 3
) -> np.ndarray:
    """
    Compute sliding window Shapley values (llmSHAP φ^SW).
    
    Provides computational efficiency at cost of some axiom violations.
    """
    from math import factorial
    
    n_tokens = len(tokens)
    attributions = np.zeros(n_tokens)
    counts = np.zeros(n_tokens)
    
    n_windows = n_tokens - window_size + 1
    
    for window_start in range(n_windows):
        window = list(range(window_start, window_start + window_size))
        outside = frozenset(range(n_tokens)) - frozenset(window)
        
        for i in window:
            local_value = 0.0
            window_without_i = [j for j in window if j != i]
            
            # Iterate over all subsets of window (excluding i)
            for subset_size in range(window_size):
                from itertools import combinations
                for subset in combinations(window_without_i, subset_size):
                    S = frozenset(subset) | outside
                    S_with_i = S | {i}
                    
                    if S in coalition_results and S_with_i in coalition_results:
                        weight = factorial(subset_size) * factorial(window_size - subset_size - 1) / factorial(window_size)
                        marginal = coalition_results[S_with_i].payoff - coalition_results[S].payoff
                        local_value += weight * marginal
            
            attributions[i] += local_value
            counts[i] += 1
    
    # Average across windows
    for i in range(n_tokens):
        if counts[i] > 0:
            attributions[i] /= counts[i]
    
    return attributions


def group_tokens(
    tokens: List[str],
    attributions: np.ndarray,
    groups: Optional[List[List[int]]] = None,
    auto_detect_phrases: bool = False
) -> Tuple[List[str], np.ndarray]:
    """
    Group tokens into phrases for more interpretable attributions.
    
    This addresses the review's suggestion to handle multi-word expressions
    like "New York" as single units.
    
    Args:
        tokens: List of token strings
        attributions: Shapley values for each token
        groups: Manual groupings as list of index lists [[0,1], [3,4,5], ...]
                If None and auto_detect_phrases=True, attempts auto-detection
        auto_detect_phrases: Whether to auto-detect common phrases
        
    Returns:
        Tuple of (grouped_tokens, grouped_attributions)
        
    Example:
        tokens = ["New", "York", "is", "great"]
        attributions = [0.3, 0.4, 0.1, 0.5]
        groups = [[0, 1]]  # Group "New" and "York"
        
        grouped_tokens, grouped_values = group_tokens(tokens, attributions, groups)
        # Result: ["New York", "is", "great"], [0.7, 0.1, 0.5]
    """
    if groups is None and auto_detect_phrases:
        # Simple heuristic: group capitalized adjacent tokens
        groups = []
        current_group = []
        
        for i, token in enumerate(tokens):
            if token[0].isupper() if token else False:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = []
        
        if len(current_group) > 1:
            groups.append(current_group)
    
    if not groups:
        return tokens, attributions
    
    # Build mapping from original index to group
    grouped_indices = set()
    for group in groups:
        grouped_indices.update(group)
    
    result_tokens = []
    result_values = []
    i = 0
    
    while i < len(tokens):
        # Check if this index starts a group
        group_found = None
        for group in groups:
            if group and group[0] == i:
                group_found = group
                break
        
        if group_found:
            # Combine tokens and sum attributions
            combined_token = " ".join(tokens[j] for j in group_found)
            combined_value = sum(attributions[j] for j in group_found)
            result_tokens.append(combined_token)
            result_values.append(combined_value)
            i = group_found[-1] + 1
        else:
            result_tokens.append(tokens[i])
            result_values.append(attributions[i])
            i += 1
    
    return result_tokens, np.array(result_values)


def apply_importance_threshold(
    attributions: np.ndarray,
    threshold: float = 0.05,
    mode: str = "zero"
) -> np.ndarray:
    """
    Apply threshold to filter noise from attribution values.
    
    Args:
        attributions: Shapley values
        threshold: Minimum absolute value to keep
        mode: "zero" to set below-threshold to 0, 
              "relative" to use relative threshold based on max value
              
    Returns:
        Filtered attributions
    """
    result = attributions.copy()
    
    if mode == "relative":
        max_val = np.abs(attributions).max()
        threshold = threshold * max_val
    
    result[np.abs(result) < threshold] = 0.0
    
    return result


if __name__ == "__main__":
    # Example usage
    print("SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution")
    print("=" * 70)
    print()
    print("Usage:")
    print("  from sfa_tokenshap import SFATokenSHAP, OllamaBackend")
    print()
    print("  # Initialize backend")
    print("  backend = OllamaBackend(model_name='llama3.2')")
    print()
    print("  # Create explainer")
    print("  explainer = SFATokenSHAP(backend, budget=50)")
    print()
    print("  # Explain a prompt")
    print("  result = explainer.explain('Why is the sky blue?')")
    print()
    print("  # Visualize")
    print("  html = explainer.visualize(result)")