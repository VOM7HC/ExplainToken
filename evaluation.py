"""
Evaluation Metrics for SFA-TokenSHAP

Implements evaluation metrics for:
- Explanation faithfulness (comprehensiveness, sufficiency)
- Bias detection (CTF Gap, PAIR scores, demographic parity)
- Explanation quality (plausibility, consistency)

Reference: llmSHAP paper evaluation methodology
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import warnings


@dataclass
class FaithfulnessResult:
    """
    Result from faithfulness evaluation.
    
    Metrics Explanation:
    -------------------
    
    Comprehensiveness (higher is better):
        - Measures performance DROP when removing top-k important tokens
        - Formula: baseline_output - output_without_top_k
        - High value = removing "important" tokens hurts output significantly
        - Range: typically [0, 1] for normalized outputs
        
    Sufficiency (higher is better):
        - Measures performance RETAINED when keeping only top-k tokens
        - Formula: output_with_only_top_k (NOT the drop!)
        - High value = top tokens alone can recreate the output
        - Range: typically [0, 1] for normalized outputs
        
        NOTE: This differs from ERASER's definition where sufficiency is the DROP
        (lower is better). We report RETAINED performance for intuitive interpretation.
        To convert to ERASER-style: eraser_sufficiency = baseline - our_sufficiency
    
    Deletion AUC (lower is better):
        - Area under curve when removing tokens from most to least important
        - Low value = output degrades quickly as important tokens are removed
        - Good explanations have low deletion AUC
        
    Insertion AUC (higher is better):
        - Area under curve when adding tokens from most to least important  
        - High value = output recovers quickly as important tokens are added
        - Good explanations have high insertion AUC
        
    Interpretation:
        Good faithfulness = High comprehensiveness + High sufficiency +
                           Low deletion AUC + High insertion AUC
    """
    comprehensiveness: float      # Drop in performance when removing important tokens
    sufficiency: float            # Performance when keeping only important tokens (RETAINED, not drop)
    deletion_curve_auc: float     # AUC of deletion curve (lower = better)
    insertion_curve_auc: float    # AUC of insertion curve (higher = better)
    
    def interpret(self) -> str:
        """Generate human-readable interpretation."""
        lines = []
        
        # Comprehensiveness
        if self.comprehensiveness > 0.3:
            lines.append(f"✓ Comprehensiveness: {self.comprehensiveness:.4f} (Good - important tokens matter)")
        elif self.comprehensiveness > 0.1:
            lines.append(f"⚠ Comprehensiveness: {self.comprehensiveness:.4f} (Moderate)")
        else:
            lines.append(f"✗ Comprehensiveness: {self.comprehensiveness:.4f} (Low - tokens may not be truly important)")
        
        # Sufficiency
        if self.sufficiency > 0.8:
            lines.append(f"✓ Sufficiency: {self.sufficiency:.4f} (Excellent - top tokens capture most info)")
        elif self.sufficiency > 0.5:
            lines.append(f"⚠ Sufficiency: {self.sufficiency:.4f} (Moderate)")
        else:
            lines.append(f"✗ Sufficiency: {self.sufficiency:.4f} (Low - top tokens miss important info)")
        
        # Deletion AUC
        if self.deletion_curve_auc < 0.5:
            lines.append(f"✓ Deletion AUC: {self.deletion_curve_auc:.4f} (Good - output degrades with removal)")
        else:
            lines.append(f"⚠ Deletion AUC: {self.deletion_curve_auc:.4f} (Higher than ideal)")
        
        # Insertion AUC
        if self.insertion_curve_auc > 0.7:
            lines.append(f"✓ Insertion AUC: {self.insertion_curve_auc:.4f} (Good - output recovers with insertion)")
        else:
            lines.append(f"⚠ Insertion AUC: {self.insertion_curve_auc:.4f} (Lower than ideal)")
        
        return "\n".join(lines)


@dataclass
class ConsistencyResult:
    """Result from consistency evaluation."""
    spearman_correlation: float   # Rank correlation between runs
    kendall_tau: float            # Kendall's tau correlation
    top_k_overlap: float          # Overlap of top-k important tokens


@dataclass
class BiasResult:
    """
    Result from bias detection evaluation.
    
    Metrics Explanation:
    -------------------
    
    CTF Gap (Counterfactual Token Fairness):
        - Measures absolute difference in average attributions between demographic groups
        - Range: [0, 1] where 0 = no difference, 1 = maximum difference
        - Higher values indicate potential bias
        - Example: If "she" has attribution 0.8 and "he" would have 0.3, CTF gap = 0.5
    
    PAIR Score (Pairwise Assessment of Implicit Reasoning):
        - Measures relative preference between demographic groups
        - Range: [0, 1] where 0.5 = unbiased (no preference)
        - Score < 0.3: Strong bias toward group2 (e.g., male-stereotyped view)
        - Score > 0.7: Strong bias toward group1 (e.g., female-stereotyped view)
        - Example: PAIR=0.02 means strong implicit bias toward stereotypical assumption
    
    Demographic Parity:
        - Measures fairness of positive attribution distribution across groups
        - Range: [0, 1] where 1 = perfect parity (all groups treated equally)
        - Lower values indicate unequal treatment across demographic groups
    
    Detected Bias Type:
        - "none": No significant bias detected
        - "gender": Gender-based attribution imbalance (CTF gap > threshold)
        - "race": Race-based attribution imbalance
        - "profession_stereotype": Profession stereotype bias
        - "implicit_gender": Subtle gender bias detected via PAIR score deviation
        - "implicit_profession": Subtle profession stereotype via PAIR deviation
        - "implicit_stereotype_conflict": Counter-stereotypical case with implicit bias
    
    Interpretation Guide:
    --------------------
    - If PAIR << 0.5 (e.g., 0.02) but bias_type="none", this indicates the CTF 
      threshold wasn't crossed but there IS implicit bias. Check PAIR score directly.
    - High CTF gap + PAIR near 0.5: Explicit attribution difference, no implicit bias
    - Low CTF gap + PAIR far from 0.5: Subtle implicit bias without explicit difference
    """
    ctf_gap: float                # Counterfactual Token Fairness gap
    pair_score: float             # Pairwise Assessment of Implicit Reasoning
    demographic_parity: float     # Demographic parity measure
    detected_bias_type: str       # Type of detected bias
    
    def interpret(self) -> str:
        """Generate human-readable interpretation of bias results."""
        lines = []
        
        # CTF Gap interpretation
        if self.ctf_gap < 0.1:
            lines.append("✓ CTF Gap: Low - attributions are similar across groups")
        elif self.ctf_gap < 0.3:
            lines.append("⚠ CTF Gap: Moderate - some attribution difference detected")
        else:
            lines.append("✗ CTF Gap: High - significant attribution imbalance")
        
        # PAIR Score interpretation  
        pair_deviation = abs(self.pair_score - 0.5)
        if pair_deviation < 0.1:
            lines.append("✓ PAIR Score: Neutral - no implicit preference detected")
        elif pair_deviation < 0.2:
            lines.append("⚠ PAIR Score: Slight preference detected")
        elif pair_deviation < 0.3:
            lines.append("⚠ PAIR Score: Moderate implicit bias")
        else:
            direction = "group1" if self.pair_score > 0.5 else "group2 (stereotypical)"
            lines.append(f"✗ PAIR Score: Strong implicit bias toward {direction}")
        
        # Demographic Parity interpretation
        if self.demographic_parity > 0.9:
            lines.append("✓ Demographic Parity: High - fair treatment across groups")
        elif self.demographic_parity > 0.7:
            lines.append("⚠ Demographic Parity: Moderate fairness")
        else:
            lines.append("✗ Demographic Parity: Low - unequal treatment detected")
        
        # Overall bias type
        if self.detected_bias_type == "none":
            if pair_deviation > 0.3:
                lines.append(f"\n⚠ Note: While no explicit bias flagged, PAIR score ({self.pair_score:.3f}) "
                           f"suggests implicit bias. Consider manual review.")
            else:
                lines.append("\n✓ Overall: No significant bias detected")
        else:
            lines.append(f"\n✗ Detected Bias Type: {self.detected_bias_type}")
        
        return "\n".join(lines)


class FaithfulnessEvaluator:
    """
    Evaluates faithfulness of token attributions.
    
    Faithfulness measures how well the attributions explain
    the actual model behavior (not just human intuition).
    """
    
    def __init__(
        self,
        model_fn: Callable[[str], float],
        tokens: List[str],
        baseline_output: float
    ):
        """
        Args:
            model_fn: Function that takes prompt string and returns score
            tokens: List of tokens in the prompt
            baseline_output: Model output for full prompt
        """
        self.model_fn = model_fn
        self.tokens = tokens
        self.baseline_output = baseline_output
        self.n_tokens = len(tokens)
    
    def _render_subset(self, indices: set) -> str:
        """Render prompt from subset of token indices."""
        return " ".join(self.tokens[i] for i in sorted(indices))
    
    def comprehensiveness(
        self,
        attributions: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Measure comprehensiveness: drop when removing top-k important tokens.
        
        Higher is better (important tokens actually matter).
        """
        if k > self.n_tokens:
            k = self.n_tokens
        
        # Get top-k token indices by attribution
        top_k_indices = set(np.argsort(attributions)[-k:])
        remaining_indices = set(range(self.n_tokens)) - top_k_indices
        
        if len(remaining_indices) == 0:
            return 1.0  # Maximum drop
        
        # Evaluate with top tokens removed
        subset_prompt = self._render_subset(remaining_indices)
        subset_output = self.model_fn(subset_prompt)
        
        # Comprehensiveness = original - perturbed (positive means drop)
        return self.baseline_output - subset_output
    
    def sufficiency(
        self,
        attributions: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Measure sufficiency: performance when keeping only top-k tokens.
        
        Higher is better (top tokens are sufficient for output).
        """
        if k > self.n_tokens:
            k = self.n_tokens
        
        # Get top-k token indices
        top_k_indices = set(np.argsort(attributions)[-k:])
        
        # Evaluate with only top tokens
        subset_prompt = self._render_subset(top_k_indices)
        subset_output = self.model_fn(subset_prompt)
        
        return subset_output
    
    def deletion_curve(
        self,
        attributions: np.ndarray,
        steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deletion curve: remove tokens in order of importance.
        
        Returns fractions and corresponding outputs.
        """
        # Sort indices by attribution (descending)
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
    
    def insertion_curve(
        self,
        attributions: np.ndarray,
        steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute insertion curve: add tokens in order of importance.
        
        Returns fractions and corresponding outputs.
        """
        # Sort indices by attribution (descending)
        sorted_indices = np.argsort(attributions)[::-1]
        
        fractions = np.linspace(0, 1, steps)
        outputs = []
        
        for frac in fractions:
            n_keep = max(1, int(frac * self.n_tokens))
            kept = set(sorted_indices[:n_keep])
            
            prompt = self._render_subset(kept)
            outputs.append(self.model_fn(prompt))
        
        return fractions, np.array(outputs)
    
    def evaluate(
        self,
        attributions: np.ndarray,
        k: int = 5,
        steps: int = 10
    ) -> FaithfulnessResult:
        """Run full faithfulness evaluation."""
        comp = self.comprehensiveness(attributions, k)
        suff = self.sufficiency(attributions, k)
        
        del_frac, del_out = self.deletion_curve(attributions, steps)
        ins_frac, ins_out = self.insertion_curve(attributions, steps)
        
        # AUC using trapezoidal rule
        del_auc = np.trapz(del_out, del_frac)
        ins_auc = np.trapz(ins_out, ins_frac)
        
        return FaithfulnessResult(
            comprehensiveness=comp,
            sufficiency=suff,
            deletion_curve_auc=del_auc,
            insertion_curve_auc=ins_auc
        )
    
    def aopc(
        self,
        attributions: np.ndarray,
        fractions: List[float] = [0.05, 0.10, 0.20, 0.50]
    ) -> Dict[str, float]:
        """
        Compute Area Over Perturbation Curve (AOPC) - standard ERASER metric.
        
        Measures average comprehensiveness across multiple masking fractions.
        This is the recommended evaluation approach from ERASER benchmark.
        
        Args:
            attributions: Token attribution values
            fractions: List of masking fractions to evaluate (default: 5%, 10%, 20%, 50%)
            
        Returns:
            Dictionary containing:
            - 'aopc': Average comprehensiveness across all fractions
            - 'aopc_scores': Individual scores for each fraction
            - 'fractions': The fractions used
            
        Reference: 
            ERASER: A Benchmark to Evaluate Rationalized NLP Models
            https://aclanthology.org/2020.acl-main.408.pdf
        """
        scores = []
        fraction_scores = {}
        
        for frac in fractions:
            k = max(1, int(len(attributions) * frac))
            comp = self.comprehensiveness(attributions, k)
            scores.append(comp)
            fraction_scores[f"comp_{int(frac*100)}%"] = comp
        
        return {
            'aopc': np.mean(scores),
            'aopc_scores': fraction_scores,
            'fractions': fractions
        }
    
    def random_baseline(
        self,
        n_runs: int = 10,
        k: int = 5,
        seed: Optional[int] = 42
    ) -> Dict[str, float]:
        """
        Compute random attribution baseline for comparison.
        
        A good explanation method should significantly outperform
        random token selection on faithfulness metrics.
        
        Args:
            n_runs: Number of random attribution runs to average
            k: Number of top tokens for comprehensiveness/sufficiency
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
            - 'random_comp_mean': Mean comprehensiveness with random attributions
            - 'random_comp_std': Std of comprehensiveness
            - 'random_suff_mean': Mean sufficiency with random attributions
            - 'random_suff_std': Std of sufficiency
        """
        rng = np.random.RandomState(seed)
        
        comp_scores = []
        suff_scores = []
        
        for _ in range(n_runs):
            # Generate random attributions
            random_attr = rng.randn(self.n_tokens)
            
            comp = self.comprehensiveness(random_attr, k)
            suff = self.sufficiency(random_attr, k)
            
            comp_scores.append(comp)
            suff_scores.append(suff)
        
        return {
            'random_comp_mean': np.mean(comp_scores),
            'random_comp_std': np.std(comp_scores),
            'random_suff_mean': np.mean(suff_scores),
            'random_suff_std': np.std(suff_scores)
        }
    
    def evaluate_with_baseline(
        self,
        attributions: np.ndarray,
        k: int = 5,
        n_random_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation including random baseline comparison.
        
        Returns faithfulness metrics along with random baseline for context.
        This helps determine if the explanation is significantly better than random.
        """
        # Standard evaluation
        result = self.evaluate(attributions, k)
        
        # Random baseline
        baseline = self.random_baseline(n_random_runs, k)
        
        # AOPC
        aopc_result = self.aopc(attributions)
        
        # Compute improvement over random
        comp_improvement = result.comprehensiveness - baseline['random_comp_mean']
        suff_improvement = result.sufficiency - baseline['random_suff_mean']
        
        return {
            'faithfulness': result,
            'random_baseline': baseline,
            'aopc': aopc_result,
            'improvement_over_random': {
                'comprehensiveness': comp_improvement,
                'sufficiency': suff_improvement,
                'comp_significant': comp_improvement > 2 * baseline['random_comp_std'],
                'suff_significant': suff_improvement > 2 * baseline['random_suff_std']
            }
        }


class ConsistencyEvaluator:
    """
    Evaluates consistency of attributions across multiple runs.
    
    Important for stochastic LLMs where repeated inference
    may yield different outputs.
    """
    
    @staticmethod
    def spearman_correlation(
        attributions1: np.ndarray,
        attributions2: np.ndarray
    ) -> float:
        """Compute Spearman rank correlation between two attribution vectors."""
        from scipy.stats import spearmanr
        
        if len(attributions1) != len(attributions2):
            raise ValueError("Attribution vectors must have same length")
        
        corr, _ = spearmanr(attributions1, attributions2)
        return corr
    
    @staticmethod
    def kendall_tau(
        attributions1: np.ndarray,
        attributions2: np.ndarray
    ) -> float:
        """Compute Kendall's tau correlation between two attribution vectors."""
        from scipy.stats import kendalltau
        
        if len(attributions1) != len(attributions2):
            raise ValueError("Attribution vectors must have same length")
        
        tau, _ = kendalltau(attributions1, attributions2)
        return tau
    
    @staticmethod
    def top_k_overlap(
        attributions1: np.ndarray,
        attributions2: np.ndarray,
        k: int = 5
    ) -> float:
        """Compute overlap of top-k important tokens."""
        k = min(k, len(attributions1), len(attributions2))
        
        top_k_1 = set(np.argsort(attributions1)[-k:])
        top_k_2 = set(np.argsort(attributions2)[-k:])
        
        overlap = len(top_k_1 & top_k_2)
        return overlap / k
    
    @staticmethod
    def evaluate(
        attributions_list: List[np.ndarray],
        k: int = 5
    ) -> ConsistencyResult:
        """
        Evaluate consistency across multiple attribution runs.
        """
        if len(attributions_list) < 2:
            raise ValueError("Need at least 2 attribution vectors for consistency")
        
        spearman_scores = []
        kendall_scores = []
        overlap_scores = []
        
        # Pairwise comparisons
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                spearman_scores.append(
                    ConsistencyEvaluator.spearman_correlation(
                        attributions_list[i], attributions_list[j]
                    )
                )
                kendall_scores.append(
                    ConsistencyEvaluator.kendall_tau(
                        attributions_list[i], attributions_list[j]
                    )
                )
                overlap_scores.append(
                    ConsistencyEvaluator.top_k_overlap(
                        attributions_list[i], attributions_list[j], k
                    )
                )
        
        return ConsistencyResult(
            spearman_correlation=np.mean(spearman_scores),
            kendall_tau=np.mean(kendall_scores),
            top_k_overlap=np.mean(overlap_scores)
        )


class BiasDetectionEvaluator:
    """
    Evaluates bias detection capabilities using token attributions.
    
    Implements metrics from bias detection literature:
    - CTF (Counterfactual Token Fairness) Gap
    - PAIR (Pairwise Assessment of Implicit Reasoning) scores
    - Demographic parity measures
    """
    
    # Common bias-related word lists
    GENDER_WORDS = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'male', 'father', 'husband', 'brother', 'son'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female', 'mother', 'wife', 'sister', 'daughter']
    }
    
    RACE_WORDS = {
        'white': ['white', 'caucasian', 'european'],
        'black': ['black', 'african', 'african-american'],
        'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian'],
        'hispanic': ['hispanic', 'latino', 'latina', 'mexican', 'spanish']
    }
    
    PROFESSION_STEREOTYPES = {
        'male_stereotyped': ['engineer', 'developer', 'programmer', 'ceo', 'doctor', 'lawyer', 'executive'],
        'female_stereotyped': ['nurse', 'secretary', 'teacher', 'receptionist', 'assistant', 'caregiver']
    }
    
    def __init__(
        self,
        tokens: List[str],
        attributions: np.ndarray
    ):
        self.tokens = [t.lower() for t in tokens]
        self.attributions = attributions
        self.n_tokens = len(tokens)
    
    def identify_demographic_tokens(self) -> Dict[str, List[int]]:
        """Identify tokens belonging to different demographic groups."""
        demographic_indices = {}
        
        # Gender
        for group, words in self.GENDER_WORDS.items():
            indices = [i for i, t in enumerate(self.tokens) if t in words]
            if indices:
                demographic_indices[f'gender_{group}'] = indices
        
        # Race
        for group, words in self.RACE_WORDS.items():
            indices = [i for i, t in enumerate(self.tokens) if t in words]
            if indices:
                demographic_indices[f'race_{group}'] = indices
        
        # Profession stereotypes
        for group, words in self.PROFESSION_STEREOTYPES.items():
            indices = [i for i, t in enumerate(self.tokens) if t in words]
            if indices:
                demographic_indices[f'profession_{group}'] = indices
        
        return demographic_indices
    
    def ctf_gap(
        self,
        group1_indices: List[int],
        group2_indices: List[int]
    ) -> float:
        """
        Compute Counterfactual Token Fairness gap.
        
        Measures difference in average attributions between
        demographically different but semantically similar tokens.
        """
        if not group1_indices or not group2_indices:
            return 0.0
        
        avg1 = np.mean(self.attributions[group1_indices])
        avg2 = np.mean(self.attributions[group2_indices])
        
        return abs(avg1 - avg2)
    
    def pair_score(
        self,
        group1_indices: List[int],
        group2_indices: List[int],
        context_indices: Optional[List[int]] = None
    ) -> float:
        """
        Compute PAIR (Pairwise Assessment of Implicit Reasoning) score.
        
        Measures whether attributions show preference for one group
        over another in similar contexts.
        """
        if not group1_indices or not group2_indices:
            return 0.5  # No bias detectable
        
        # Normalize attributions
        norm_attr = self.attributions / (np.abs(self.attributions).max() + 1e-10)
        
        # Average attribution for each group
        avg1 = np.mean(norm_attr[group1_indices])
        avg2 = np.mean(norm_attr[group2_indices])
        
        # PAIR score: 0.5 is unbiased, deviation indicates bias
        # Positive indicates preference for group1, negative for group2
        pair = 0.5 + (avg1 - avg2) / 2
        
        return np.clip(pair, 0, 1)
    
    def demographic_parity(
        self,
        positive_threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute demographic parity: proportion of positive attributions per group.
        
        Args:
            positive_threshold: Threshold above which attribution is "positive"
        
        Returns:
            Dictionary of group -> proportion positive
        """
        demographic_indices = self.identify_demographic_tokens()
        parity_scores = {}
        
        for group, indices in demographic_indices.items():
            positive_count = sum(
                1 for i in indices if self.attributions[i] > positive_threshold
            )
            parity_scores[group] = positive_count / len(indices) if indices else 0.0
        
        return parity_scores
    
    def detect_bias_type(
        self,
        ctf_threshold: float = 0.3,
        pair_threshold: float = 0.2
    ) -> Tuple[str, float]:
        """
        Automatically detect the most prominent type of bias.
        
        Uses BOTH CTF gap AND PAIR score for more sensitive detection.
        
        Args:
            ctf_threshold: CTF gap above this indicates potential bias (default 0.3)
            pair_threshold: PAIR deviation from 0.5 above this indicates bias (default 0.2)
                           e.g., PAIR < 0.3 or PAIR > 0.7 triggers detection
        
        Returns:
            Tuple of (bias_type, severity_score)
            
        Bias Types:
            - "none": No significant bias detected
            - "gender": Gender-based attribution imbalance
            - "race": Race-based attribution imbalance  
            - "profession_stereotype": Profession stereotype bias
            - "implicit_gender": Subtle gender bias (PAIR deviation)
            - "implicit_profession": Subtle profession stereotype (PAIR deviation)
        """
        demographic_indices = self.identify_demographic_tokens()
        
        max_gap = 0.0
        max_pair_deviation = 0.0
        bias_type = "none"
        
        # Check gender bias (both CTF and PAIR)
        if 'gender_male' in demographic_indices and 'gender_female' in demographic_indices:
            gap = self.ctf_gap(
                demographic_indices['gender_male'],
                demographic_indices['gender_female']
            )
            pair = self.pair_score(
                demographic_indices['gender_male'],
                demographic_indices['gender_female']
            )
            pair_deviation = abs(pair - 0.5)
            
            if gap > max_gap:
                max_gap = gap
                bias_type = "gender"
            
            # Check for implicit bias via PAIR score
            if pair_deviation > max_pair_deviation:
                max_pair_deviation = pair_deviation
                if pair_deviation > pair_threshold and gap < ctf_threshold:
                    bias_type = "implicit_gender"
        
        # Check race bias
        race_groups = [k for k in demographic_indices.keys() if k.startswith('race_')]
        for i, g1 in enumerate(race_groups):
            for g2 in race_groups[i+1:]:
                gap = self.ctf_gap(
                    demographic_indices[g1],
                    demographic_indices[g2]
                )
                if gap > max_gap:
                    max_gap = gap
                    bias_type = "race"
        
        # Check profession stereotype bias (both CTF and PAIR)
        if 'profession_male_stereotyped' in demographic_indices and 'profession_female_stereotyped' in demographic_indices:
            gap = self.ctf_gap(
                demographic_indices['profession_male_stereotyped'],
                demographic_indices['profession_female_stereotyped']
            )
            pair = self.pair_score(
                demographic_indices['profession_male_stereotyped'],
                demographic_indices['profession_female_stereotyped']
            )
            pair_deviation = abs(pair - 0.5)
            
            if gap > max_gap:
                max_gap = gap
                bias_type = "profession_stereotype"
            
            # Check for implicit profession stereotype via PAIR
            if pair_deviation > max_pair_deviation:
                max_pair_deviation = pair_deviation
                if pair_deviation > pair_threshold and gap < ctf_threshold:
                    bias_type = "implicit_profession"
        
        # Also check for implicit bias if only one demographic group is present
        # (e.g., only "she" with "doctor" - can detect implicit association)
        gender_groups = [k for k in demographic_indices.keys() if k.startswith('gender_')]
        prof_groups = [k for k in demographic_indices.keys() if k.startswith('profession_')]
        
        if len(gender_groups) == 1 and len(prof_groups) == 1:
            # Single gender + single profession: check attribution correlation
            gender_idx = demographic_indices[gender_groups[0]]
            prof_idx = demographic_indices[prof_groups[0]]
            
            gender_attr = np.mean(self.attributions[gender_idx]) if gender_idx else 0
            prof_attr = np.mean(self.attributions[prof_idx]) if prof_idx else 0
            
            # Large difference in attribution might indicate bias
            attr_gap = abs(gender_attr - prof_attr)
            if attr_gap > ctf_threshold:
                # Check if it's counter-stereotypical
                is_female = 'female' in gender_groups[0]
                is_male_prof = 'male_stereotyped' in prof_groups[0]
                is_female_prof = 'female_stereotyped' in prof_groups[0]
                
                if (is_female and is_male_prof) or (not is_female and is_female_prof):
                    # Counter-stereotypical case
                    if max_pair_deviation > pair_threshold:
                        bias_type = "implicit_stereotype_conflict"
                        max_gap = max(max_gap, attr_gap)
        
        # Final severity is max of CTF gap and PAIR deviation
        severity = max(max_gap, max_pair_deviation * 2)  # Scale PAIR to be comparable
        
        return bias_type, severity
    
    def evaluate(self) -> BiasResult:
        """Run full bias detection evaluation."""
        demographic_indices = self.identify_demographic_tokens()
        
        # Compute CTF gap across all detected groups
        gaps = []
        pair_scores = []
        
        groups = list(demographic_indices.keys())
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                gap = self.ctf_gap(
                    demographic_indices[g1],
                    demographic_indices[g2]
                )
                gaps.append(gap)
                
                pair = self.pair_score(
                    demographic_indices[g1],
                    demographic_indices[g2]
                )
                pair_scores.append(pair)
        
        # Demographic parity
        parity = self.demographic_parity()
        parity_variance = np.var(list(parity.values())) if parity else 0.0
        
        # Detect primary bias type
        bias_type, severity = self.detect_bias_type()
        
        return BiasResult(
            ctf_gap=np.mean(gaps) if gaps else 0.0,
            pair_score=np.mean(pair_scores) if pair_scores else 0.5,
            demographic_parity=1.0 - parity_variance,  # Higher is more fair
            detected_bias_type=bias_type
        )


class PlausibilityEvaluator:
    """
    Evaluates plausibility: alignment with human intuition.
    
    Note: Unlike faithfulness, plausibility measures whether
    explanations make sense to humans, not whether they
    accurately reflect model behavior.
    """
    
    def __init__(
        self,
        tokens: List[str],
        attributions: np.ndarray
    ):
        self.tokens = [t.lower() for t in tokens]
        self.attributions = attributions
    
    def stopword_suppression(self) -> float:
        """
        Check if stopwords have lower attributions than content words.
        
        Returns ratio of content word attribution to stopword attribution.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'it', 'its'
        }
        
        stopword_attrs = [
            self.attributions[i] for i, t in enumerate(self.tokens) 
            if t in stopwords
        ]
        content_attrs = [
            self.attributions[i] for i, t in enumerate(self.tokens)
            if t not in stopwords and t.isalpha()
        ]
        
        if not stopword_attrs or not content_attrs:
            return 1.0
        
        avg_stopword = np.mean(np.abs(stopword_attrs))
        avg_content = np.mean(np.abs(content_attrs))
        
        if avg_stopword == 0:
            return float('inf')
        
        return avg_content / avg_stopword
    
    def question_word_importance(self) -> float:
        """
        For questions, check if question words are highly attributed.
        """
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
        
        question_indices = [
            i for i, t in enumerate(self.tokens)
            if t in question_words
        ]
        
        if not question_indices:
            return 0.0  # No question words to evaluate
        
        # Check if question words are in top half of attributions
        n = len(self.tokens)
        top_half = set(np.argsort(self.attributions)[-n//2:])
        
        overlap = len(set(question_indices) & top_half)
        return overlap / len(question_indices)


class ExplanationQualityMetrics:
    """
    Comprehensive quality metrics for explanations.
    
    Combines faithfulness, consistency, and plausibility measures.
    """
    
    @staticmethod
    def normalized_comprehensiveness(
        attributions: np.ndarray,
        random_baseline: float,
        measured_value: float
    ) -> float:
        """
        Normalize comprehensiveness relative to random baseline.
        """
        if random_baseline == 0:
            return measured_value
        return measured_value / random_baseline
    
    @staticmethod
    def area_between_curves(
        deletion_curve: np.ndarray,
        insertion_curve: np.ndarray
    ) -> float:
        """
        Compute area between deletion and insertion curves.
        
        Larger area indicates better explanation quality.
        """
        if len(deletion_curve) != len(insertion_curve):
            raise ValueError("Curves must have same length")
        
        return np.trapz(insertion_curve - deletion_curve)
    
    @staticmethod
    def log_odds_difference(
        original_prob: float,
        perturbed_prob: float
    ) -> float:
        """
        Compute log-odds difference between original and perturbed.
        
        Useful for classification tasks.
        """
        eps = 1e-10
        
        original_odds = original_prob / (1 - original_prob + eps)
        perturbed_odds = perturbed_prob / (1 - perturbed_prob + eps)
        
        return np.log(original_odds + eps) - np.log(perturbed_odds + eps)


def evaluate_explanation_quality(
    tokens: List[str],
    attributions: np.ndarray,
    model_fn: Callable[[str], float],
    n_runs: int = 3,
    faithfulness_k: int = 5
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of explanation quality.
    
    Args:
        tokens: List of tokens
        attributions: Attribution values for each token
        model_fn: Function that scores prompts
        n_runs: Number of runs for consistency evaluation
        faithfulness_k: Number of top tokens for faithfulness
        
    Returns:
        Dictionary of all evaluation metrics
    """
    # Full prompt baseline
    full_prompt = " ".join(tokens)
    baseline = model_fn(full_prompt)
    
    # Faithfulness
    faith_eval = FaithfulnessEvaluator(model_fn, tokens, baseline)
    faith_result = faith_eval.evaluate(attributions, k=faithfulness_k)
    
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
            'deletion_auc': faith_result.deletion_curve_auc,
            'insertion_auc': faith_result.insertion_curve_auc
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


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics for SFA-TokenSHAP")
    print("=" * 50)
    
    # Example: evaluate some attributions
    tokens = ["why", "is", "the", "sky", "blue", "?"]
    attributions = np.array([0.15, 0.05, 0.02, 0.35, 0.40, 0.03])
    
    print(f"\nTokens: {tokens}")
    print(f"Attributions: {attributions}")
    
    # Plausibility check
    plaus = PlausibilityEvaluator(tokens, attributions)
    print(f"\nStopword suppression ratio: {plaus.stopword_suppression():.2f}")
    print(f"Question word importance: {plaus.question_word_importance():.2f}")
    
    print("\n✓ High-quality explanation: content words ('sky', 'blue') have higher")
    print("  attributions than stopwords ('the', 'is')")