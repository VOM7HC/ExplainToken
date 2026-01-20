"""
Evaluation Metrics for SFA-TokenSHAP
Faithfulness, Consistency, Bias Detection, Plausibility
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr, kendalltau


@dataclass
class FaithfulnessResult:
    comprehensiveness: float
    sufficiency: float
    deletion_auc: float
    insertion_auc: float
    
    def interpret(self) -> str:
        lines = []
        if self.comprehensiveness > 0.3:
            lines.append(f"✓ Comprehensiveness: {self.comprehensiveness:.4f} (Good)")
        elif self.comprehensiveness > 0.1:
            lines.append(f"⚠ Comprehensiveness: {self.comprehensiveness:.4f} (Moderate)")
        else:
            lines.append(f"✗ Comprehensiveness: {self.comprehensiveness:.4f} (Low)")
        
        if self.sufficiency > 0.8:
            lines.append(f"✓ Sufficiency: {self.sufficiency:.4f} (Excellent)")
        elif self.sufficiency > 0.5:
            lines.append(f"⚠ Sufficiency: {self.sufficiency:.4f} (Moderate)")
        else:
            lines.append(f"✗ Sufficiency: {self.sufficiency:.4f} (Low)")
        
        return "\n".join(lines)


@dataclass
class ConsistencyResult:
    spearman_correlation: float
    kendall_tau: float
    top_k_overlap: float


@dataclass
class BiasResult:
    ctf_gap: float
    pair_score: float
    demographic_parity: float
    detected_bias_type: str
    
    def interpret(self) -> str:
        lines = []
        
        if self.ctf_gap < 0.1:
            lines.append("✓ CTF Gap: Low - similar attributions across groups")
        elif self.ctf_gap < 0.3:
            lines.append("⚠ CTF Gap: Moderate - some attribution difference")
        else:
            lines.append("✗ CTF Gap: High - significant attribution imbalance")
        
        pair_deviation = abs(self.pair_score - 0.5)
        if pair_deviation < 0.1:
            lines.append("✓ PAIR Score: Neutral - no implicit preference")
        elif pair_deviation < 0.3:
            lines.append("⚠ PAIR Score: Moderate implicit bias")
        else:
            direction = "group1" if self.pair_score > 0.5 else "group2"
            lines.append(f"✗ PAIR Score: Strong implicit bias toward {direction}")
        
        if self.detected_bias_type == "none":
            lines.append("\n✓ No significant bias detected")
        else:
            lines.append(f"\n✗ Detected Bias Type: {self.detected_bias_type}")
        
        return "\n".join(lines)


class FaithfulnessEvaluator:
    """Evaluate faithfulness of token attributions"""
    
    def __init__(self, model_fn: Callable[[str], float], tokens: List[str], baseline_output: float):
        self.model_fn = model_fn
        self.tokens = tokens
        self.baseline_output = baseline_output
        self.n_tokens = len(tokens)
    
    def _render_subset(self, indices: set) -> str:
        return " ".join(self.tokens[i] for i in sorted(indices))
    
    def comprehensiveness(self, attributions: np.ndarray, k: int = 5) -> float:
        k = min(k, self.n_tokens)
        top_k_indices = set(np.argsort(attributions)[-k:])
        remaining = set(range(self.n_tokens)) - top_k_indices
        
        if len(remaining) == 0:
            return 1.0
        
        subset_prompt = self._render_subset(remaining)
        subset_output = self.model_fn(subset_prompt)
        return self.baseline_output - subset_output
    
    def sufficiency(self, attributions: np.ndarray, k: int = 5) -> float:
        k = min(k, self.n_tokens)
        top_k_indices = set(np.argsort(attributions)[-k:])
        
        subset_prompt = self._render_subset(top_k_indices)
        return self.model_fn(subset_prompt)
    
    def deletion_curve(self, attributions: np.ndarray, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(attributions)[::-1]
        fractions = np.linspace(0, 1, steps)
        outputs = []
        
        for frac in fractions:
            n_remove = int(frac * self.n_tokens)
            removed = set(sorted_indices[:n_remove])
            remaining = set(range(self.n_tokens)) - removed
            
            if len(remaining) == 0:
                outputs.append(0.0)
            else:
                prompt = self._render_subset(remaining)
                outputs.append(self.model_fn(prompt))
        
        return fractions, np.array(outputs)
    
    def insertion_curve(self, attributions: np.ndarray, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(attributions)[::-1]
        fractions = np.linspace(0, 1, steps)
        outputs = []
        
        for frac in fractions:
            n_keep = max(1, int(frac * self.n_tokens))
            kept = set(sorted_indices[:n_keep])
            prompt = self._render_subset(kept)
            outputs.append(self.model_fn(prompt))
        
        return fractions, np.array(outputs)
    
    def evaluate(self, attributions: np.ndarray, k: int = 5, steps: int = 10) -> FaithfulnessResult:
        comp = self.comprehensiveness(attributions, k)
        suff = self.sufficiency(attributions, k)
        
        del_frac, del_out = self.deletion_curve(attributions, steps)
        ins_frac, ins_out = self.insertion_curve(attributions, steps)
        
        del_auc = np.trapz(del_out, del_frac)
        ins_auc = np.trapz(ins_out, ins_frac)
        
        return FaithfulnessResult(comp, suff, del_auc, ins_auc)
    
    def aopc(self, attributions: np.ndarray, fractions: List[float] = [0.05, 0.10, 0.20, 0.50]) -> Dict:
        scores = []
        fraction_scores = {}
        
        for frac in fractions:
            k = max(1, int(len(attributions) * frac))
            comp = self.comprehensiveness(attributions, k)
            scores.append(comp)
            fraction_scores[f"comp_{int(frac*100)}%"] = comp
        
        return {'aopc': np.mean(scores), 'aopc_scores': fraction_scores, 'fractions': fractions}
    
    def random_baseline(self, n_runs: int = 10, k: int = 5, seed: int = 42) -> Dict:
        rng = np.random.RandomState(seed)
        comp_scores, suff_scores = [], []
        
        for _ in range(n_runs):
            random_attr = rng.randn(self.n_tokens)
            comp_scores.append(self.comprehensiveness(random_attr, k))
            suff_scores.append(self.sufficiency(random_attr, k))
        
        return {
            'random_comp_mean': np.mean(comp_scores),
            'random_comp_std': np.std(comp_scores),
            'random_suff_mean': np.mean(suff_scores),
            'random_suff_std': np.std(suff_scores)
        }


class ConsistencyEvaluator:
    """Evaluate consistency across multiple runs"""
    
    @staticmethod
    def spearman_correlation(attr1: np.ndarray, attr2: np.ndarray) -> float:
        corr, _ = spearmanr(attr1, attr2)
        return corr
    
    @staticmethod
    def kendall_tau_correlation(attr1: np.ndarray, attr2: np.ndarray) -> float:
        tau, _ = kendalltau(attr1, attr2)
        return tau
    
    @staticmethod
    def top_k_overlap(attr1: np.ndarray, attr2: np.ndarray, k: int = 5) -> float:
        k = min(k, len(attr1), len(attr2))
        top_k_1 = set(np.argsort(attr1)[-k:])
        top_k_2 = set(np.argsort(attr2)[-k:])
        return len(top_k_1 & top_k_2) / k
    
    @staticmethod
    def evaluate(attributions_list: List[np.ndarray], k: int = 5) -> ConsistencyResult:
        spearman_scores, kendall_scores, overlap_scores = [], [], []
        
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                spearman_scores.append(ConsistencyEvaluator.spearman_correlation(
                    attributions_list[i], attributions_list[j]))
                kendall_scores.append(ConsistencyEvaluator.kendall_tau_correlation(
                    attributions_list[i], attributions_list[j]))
                overlap_scores.append(ConsistencyEvaluator.top_k_overlap(
                    attributions_list[i], attributions_list[j], k))
        
        return ConsistencyResult(
            np.mean(spearman_scores),
            np.mean(kendall_scores),
            np.mean(overlap_scores)
        )


class BiasDetectionEvaluator:
    """Evaluate bias detection using token attributions"""
    
    GENDER_WORDS = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'male', 'father', 'husband', 'brother', 'son'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female', 'mother', 'wife', 'sister', 'daughter']
    }
    
    PROFESSION_STEREOTYPES = {
        'male_stereotyped': ['engineer', 'developer', 'programmer', 'ceo', 'doctor', 'lawyer', 'executive'],
        'female_stereotyped': ['nurse', 'secretary', 'teacher', 'receptionist', 'assistant', 'caregiver']
    }
    
    def __init__(self, tokens: List[str], attributions: np.ndarray):
        self.tokens = [t.lower() for t in tokens]
        self.attributions = attributions
    
    def identify_demographic_tokens(self) -> Dict[str, List[int]]:
        demographic_indices = {}
        
        for group, words in self.GENDER_WORDS.items():
            indices = [i for i, t in enumerate(self.tokens) if t in words]
            if indices:
                demographic_indices[f'gender_{group}'] = indices
        
        for group, words in self.PROFESSION_STEREOTYPES.items():
            indices = [i for i, t in enumerate(self.tokens) if t in words]
            if indices:
                demographic_indices[f'profession_{group}'] = indices
        
        return demographic_indices
    
    def ctf_gap(self, group1_indices: List[int], group2_indices: List[int]) -> float:
        if not group1_indices or not group2_indices:
            return 0.0
        avg1 = np.mean(self.attributions[group1_indices])
        avg2 = np.mean(self.attributions[group2_indices])
        return abs(avg1 - avg2)
    
    def pair_score(self, group1_indices: List[int], group2_indices: List[int]) -> float:
        if not group1_indices or not group2_indices:
            return 0.5
        
        norm_attr = self.attributions / (np.abs(self.attributions).max() + 1e-10)
        avg1 = np.mean(norm_attr[group1_indices])
        avg2 = np.mean(norm_attr[group2_indices])
        
        return np.clip(0.5 + (avg1 - avg2) / 2, 0, 1)
    
    def demographic_parity(self, threshold: float = 0.0) -> Dict[str, float]:
        demographic_indices = self.identify_demographic_tokens()
        parity_scores = {}
        
        for group, indices in demographic_indices.items():
            positive_count = sum(1 for i in indices if self.attributions[i] > threshold)
            parity_scores[group] = positive_count / len(indices) if indices else 0.0
        
        return parity_scores
    
    def detect_bias_type(self, ctf_threshold: float = 0.3, pair_threshold: float = 0.2) -> Tuple[str, float]:
        demographic_indices = self.identify_demographic_tokens()
        
        max_gap = 0.0
        bias_type = "none"
        
        # Check gender bias
        if 'gender_male' in demographic_indices and 'gender_female' in demographic_indices:
            gap = self.ctf_gap(demographic_indices['gender_male'], demographic_indices['gender_female'])
            pair = self.pair_score(demographic_indices['gender_male'], demographic_indices['gender_female'])
            
            if gap > max_gap:
                max_gap = gap
                bias_type = "gender"
            
            if abs(pair - 0.5) > pair_threshold and gap < ctf_threshold:
                bias_type = "implicit_gender"
        
        # Check profession stereotype bias
        if 'profession_male_stereotyped' in demographic_indices and 'profession_female_stereotyped' in demographic_indices:
            gap = self.ctf_gap(
                demographic_indices['profession_male_stereotyped'],
                demographic_indices['profession_female_stereotyped']
            )
            
            if gap > max_gap:
                max_gap = gap
                bias_type = "profession_stereotype"
        
        return bias_type, max_gap
    
    def evaluate(self) -> BiasResult:
        demographic_indices = self.identify_demographic_tokens()
        
        gaps, pair_scores = [], []
        groups = list(demographic_indices.keys())
        
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                gaps.append(self.ctf_gap(demographic_indices[g1], demographic_indices[g2]))
                pair_scores.append(self.pair_score(demographic_indices[g1], demographic_indices[g2]))
        
        parity = self.demographic_parity()
        parity_variance = np.var(list(parity.values())) if parity else 0.0
        
        bias_type, _ = self.detect_bias_type()
        
        return BiasResult(
            ctf_gap=np.mean(gaps) if gaps else 0.0,
            pair_score=np.mean(pair_scores) if pair_scores else 0.5,
            demographic_parity=1.0 - parity_variance,
            detected_bias_type=bias_type
        )


class PlausibilityEvaluator:
    """Evaluate plausibility of attributions"""
    
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'it', 'its'
    }
    
    QUESTION_WORDS = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
    
    def __init__(self, tokens: List[str], attributions: np.ndarray):
        self.tokens = [t.lower() for t in tokens]
        self.attributions = attributions
    
    def stopword_suppression(self) -> float:
        stopword_attrs = [self.attributions[i] for i, t in enumerate(self.tokens) if t in self.STOPWORDS]
        content_attrs = [self.attributions[i] for i, t in enumerate(self.tokens) 
                        if t not in self.STOPWORDS and t.isalpha()]
        
        if not stopword_attrs or not content_attrs:
            return 1.0
        
        avg_stopword = np.mean(np.abs(stopword_attrs))
        avg_content = np.mean(np.abs(content_attrs))
        
        return avg_content / (avg_stopword + 1e-10)
    
    def question_word_importance(self) -> float:
        question_indices = [i for i, t in enumerate(self.tokens) if t in self.QUESTION_WORDS]
        
        if not question_indices:
            return 0.0
        
        n = len(self.tokens)
        top_half = set(np.argsort(self.attributions)[-n//2:])
        
        overlap = len(set(question_indices) & top_half)
        return overlap / len(question_indices)


def evaluate_explanation_quality(tokens: List[str], attributions: np.ndarray,
                                  model_fn: Callable[[str], float], k: int = 5) -> Dict:
    """Comprehensive evaluation of explanation quality"""
    
    full_prompt = " ".join(tokens)
    baseline = model_fn(full_prompt)
    
    # Faithfulness
    faith_eval = FaithfulnessEvaluator(model_fn, tokens, baseline)
    faith_result = faith_eval.evaluate(attributions, k=k)
    
    # Plausibility
    plaus_eval = PlausibilityEvaluator(tokens, attributions)
    stopword_ratio = plaus_eval.stopword_suppression()
    question_importance = plaus_eval.question_word_importance()
    
    # Bias detection
    bias_eval = BiasDetectionEvaluator(tokens, attributions)
    bias_result = bias_eval.evaluate()
    
    return {
        'faithfulness': {
            'comprehensiveness': faith_result.comprehensiveness,
            'sufficiency': faith_result.sufficiency,
            'deletion_auc': faith_result.deletion_auc,
            'insertion_auc': faith_result.insertion_auc
        },
        'plausibility': {
            'stopword_suppression_ratio': stopword_ratio,
            'question_word_importance': question_importance
        },
        'bias_detection': {
            'ctf_gap': bias_result.ctf_gap,
            'pair_score': bias_result.pair_score,
            'demographic_parity': bias_result.demographic_parity,
            'detected_bias': bias_result.detected_bias_type
        }
    }