"""Evaluation metrics for SFA-TokenSHAP"""

import numpy as np
from typing import List, Dict, Callable
from scipy.stats import spearmanr, kendalltau


class FaithfulnessEvaluator:
    def __init__(self, model_fn: Callable[[str], float], tokens: List[str], baseline: float):
        self.model_fn = model_fn
        self.tokens = tokens
        self.baseline = baseline
        self.n = len(tokens)
    
    def _render(self, idx: set) -> str:
        return " ".join(self.tokens[i] for i in sorted(idx))
    
    def evaluate(self, attr: np.ndarray, k: int = 3) -> Dict:
        k = min(k, self.n)
        top_k = set(np.argsort(attr)[-k:])
        
        comp = self.baseline - self.model_fn(self._render(set(range(self.n)) - top_k))
        suff = self.model_fn(self._render(top_k))
        
        order = np.argsort(attr)[::-1]
        del_scores, ins_scores = [], []
        for f in np.linspace(0, 1, 10):
            nr = int(f * self.n)
            rem = set(range(self.n)) - set(order[:nr])
            del_scores.append(self.model_fn(self._render(rem)) if rem else 0.0)
            kept = set(order[:max(1, int(f * self.n))])
            ins_scores.append(self.model_fn(self._render(kept)))
        
        return {"comprehensiveness": comp, "sufficiency": suff,
                "deletion_auc": np.trapz(del_scores), "insertion_auc": np.trapz(ins_scores)}


class ConsistencyEvaluator:
    @staticmethod
    def evaluate(attrs: List[np.ndarray], k: int = 3) -> Dict:
        sp, kt, ov = [], [], []
        for i in range(len(attrs)):
            for j in range(i + 1, len(attrs)):
                c, _ = spearmanr(attrs[i], attrs[j])
                sp.append(c if not np.isnan(c) else 0.0)
                t, _ = kendalltau(attrs[i], attrs[j])
                kt.append(t if not np.isnan(t) else 0.0)
                k_ = min(k, len(attrs[i]))
                t1, t2 = set(np.argsort(attrs[i])[-k_:]), set(np.argsort(attrs[j])[-k_:])
                ov.append(len(t1 & t2) / k_)
        return {"spearman": np.mean(sp), "kendall": np.mean(kt), "top_k_overlap": np.mean(ov)}


class BiasEvaluator:
    GENDER = {'male': {'he', 'him', 'his', 'man', 'men', 'boy', 'father', 'husband', 'brother', 'son'},
              'female': {'she', 'her', 'hers', 'woman', 'women', 'girl', 'mother', 'wife', 'sister', 'daughter'}}
    
    @staticmethod
    def evaluate(tokens: List[str], attr: np.ndarray) -> Dict:
        toks = [t.lower() for t in tokens]
        m_idx = [i for i, t in enumerate(toks) if t in BiasEvaluator.GENDER['male']]
        f_idx = [i for i, t in enumerate(toks) if t in BiasEvaluator.GENDER['female']]
        
        if not m_idx or not f_idx:
            return {"ctf_gap": 0.0, "pair_score": 0.5, "bias_type": "none"}
        
        gap = abs(np.mean(attr[m_idx]) - np.mean(attr[f_idx]))
        norm = attr / (np.abs(attr).max() + 1e-10)
        pair = np.clip(0.5 + (np.mean(norm[m_idx]) - np.mean(norm[f_idx])) / 2, 0, 1)
        
        return {"ctf_gap": gap, "pair_score": pair,
                "bias_type": "gender" if gap > 0.3 or abs(pair - 0.5) > 0.2 else "none"}


class PlausibilityEvaluator:
    QUESTION = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
    
    @staticmethod
    def evaluate(tokens: List[str], attr: np.ndarray, weights: np.ndarray) -> Dict:
        content_mask = weights > 0.5
        c_attr = np.abs(attr[content_mask]).mean() if content_mask.any() else 0
        f_attr = np.abs(attr[~content_mask]).mean() if (~content_mask).any() else 1e-10
        
        toks = [t.lower() for t in tokens]
        q_idx = [i for i, t in enumerate(toks) if t in PlausibilityEvaluator.QUESTION]
        top_half = set(np.argsort(attr)[-len(tokens)//2:])
        q_imp = len(set(q_idx) & top_half) / len(q_idx) if q_idx else 0.0
        
        return {"suppression_ratio": c_attr / f_attr, "question_importance": q_imp}
