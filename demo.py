#!/usr/bin/env python3
"""
SFA-TokenSHAP Demonstration

This script demonstrates the complete SFA-TokenSHAP workflow:
1. Stage 0: Setup and preparation
2. Stage 1: TokenSHAP Monte Carlo Shapley estimation
3. Stage 2: SFA-style refinement
4. Evaluation and visualization

Usage:
    python demo.py --backend ollama --model llama3.2
    python demo.py --backend huggingface --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import numpy as np
import warnings
from typing import Dict, Any

# Import local modules
try:
    from sfa_tokenshap import (
        SFATokenSHAP,
        OllamaBackend,
        HuggingFaceBackend,
        CoalitionStrategy,
        ValueFunctionType
    )
    from evaluation import (
        evaluate_explanation_quality,
        FaithfulnessEvaluator,
        PlausibilityEvaluator,
        ConsistencyEvaluator,
        BiasDetectionEvaluator
    )
except ImportError:
    # If running as script, add parent to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from sfa_tokenshap import (
        SFATokenSHAP,
        OllamaBackend,
        HuggingFaceBackend,
        CoalitionStrategy,
        ValueFunctionType
    )
    from evaluation import (
        evaluate_explanation_quality,
        FaithfulnessEvaluator,
        PlausibilityEvaluator,
        ConsistencyEvaluator,
        BiasDetectionEvaluator
    )


def create_backend(backend_type: str, model_name: str, seed: int = 42):
    """Create the appropriate LLM backend."""
    if backend_type.lower() == "ollama":
        return OllamaBackend(
            model_name=model_name,
            temperature=0.0,
            seed=seed
        )
    elif backend_type.lower() == "huggingface":
        return HuggingFaceBackend(
            model_name=model_name,
            temperature=0.0,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def demo_basic_explanation(backend, prompt: str) -> Dict[str, Any]:
    """
    Demonstrate basic SFA-TokenSHAP explanation.
    """
    print("\n" + "=" * 60)
    print("BASIC EXPLANATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    # Create explainer
    explainer = SFATokenSHAP(
        backend=backend,
        value_function_type=ValueFunctionType.EMBEDDING_SIMILARITY,
        coalition_strategy=CoalitionStrategy.DELETE,
        budget=50,  # Number of coalition evaluations
        refiner_type="ridge",
        use_cache=True,
        seed=42
    )
    
    print("\nRunning SFA-TokenSHAP...")
    print("  - Stage 1: Monte Carlo Shapley estimation")
    print("  - Stage 2: SFA refinement")
    
    # Generate explanation
    result = explainer.explain(prompt, train_refiner=True, normalize=True)
    
    # Display results
    print("\n--- Results ---")
    print(f"Tokens: {result['tokens']}")
    print(f"\nStage 1 (Raw TokenSHAP) attributions:")
    for token, value in zip(result['tokens'], result['stage1_values']):
        bar = "█" * int(abs(value) * 20)
        sign = "+" if value > 0 else "-"
        print(f"  {token:15s} {sign}{abs(value):.4f} {bar}")
    
    print(f"\nStage 2 (SFA Refined) attributions:")
    for token, value in zip(result['tokens'], result['stage2_values']):
        bar = "█" * int(abs(value) * 20)
        sign = "+" if value > 0 else "-"
        print(f"  {token:15s} {sign}{abs(value):.4f} {bar}")
    
    print(f"\nCache statistics: {result['cache_stats']}")
    
    # Generate HTML visualization
    html = explainer.visualize(result, use_stage2=True)
    print("\n--- HTML Visualization (Stage 2) ---")
    print("(Open in browser to see color-coded tokens)")
    print(html[:200] + "..." if len(html) > 200 else html)
    
    return result


def demo_bias_detection(backend, prompt: str) -> Dict[str, Any]:
    """
    Demonstrate bias detection capabilities.
    """
    print("\n" + "=" * 60)
    print("BIAS DETECTION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    # Create explainer
    explainer = SFATokenSHAP(
        backend=backend,
        budget=30,
        seed=42
    )
    
    print("\nGenerating explanation...")
    result = explainer.explain(prompt, train_refiner=True)
    
    # Run bias detection
    print("\n--- Bias Analysis ---")
    bias_eval = BiasDetectionEvaluator(
        result['tokens'],
        result['stage2_values']
    )
    
    # Identify demographic tokens
    demo_tokens = bias_eval.identify_demographic_tokens()
    print("\nIdentified demographic tokens:")
    for group, indices in demo_tokens.items():
        tokens_in_group = [result['tokens'][i] for i in indices]
        print(f"  {group}: {tokens_in_group}")
    
    # Evaluate bias
    bias_result = bias_eval.evaluate()
    print(f"\nBias Detection Results:")
    print(f"  CTF Gap: {bias_result.ctf_gap:.4f}")
    print(f"  PAIR Score: {bias_result.pair_score:.4f} (0.5 = unbiased)")
    print(f"  Demographic Parity: {bias_result.demographic_parity:.4f}")
    print(f"  Detected Bias Type: {bias_result.detected_bias_type}")
    
    return result


def demo_consistency(backend, prompt: str, n_runs: int = 3) -> None:
    """
    Demonstrate consistency evaluation across multiple runs.
    """
    print("\n" + "=" * 60)
    print("CONSISTENCY EVALUATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    print(f"Running {n_runs} explanation passes...")
    
    attributions_list = []
    
    for i in range(n_runs):
        # Use different seeds for each run to simulate stochasticity
        explainer = SFATokenSHAP(
            backend=backend,
            budget=30,
            seed=42 + i  # Vary seed
        )
        
        result = explainer.explain(prompt, train_refiner=False)
        attributions_list.append(result['stage1_values'])
        print(f"  Run {i+1} complete")
    
    # Evaluate consistency
    print("\n--- Consistency Analysis ---")
    consistency = ConsistencyEvaluator.evaluate(attributions_list, k=3)
    
    print(f"  Spearman Correlation: {consistency.spearman_correlation:.4f}")
    print(f"  Kendall's Tau: {consistency.kendall_tau:.4f}")
    print(f"  Top-3 Overlap: {consistency.top_k_overlap:.4f}")
    
    if consistency.spearman_correlation > 0.8:
        print("\n✓ High consistency: Explanations are stable across runs")
    elif consistency.spearman_correlation > 0.5:
        print("\n⚠ Moderate consistency: Some variation across runs")
    else:
        print("\n✗ Low consistency: Explanations vary significantly")


def demo_faithfulness(backend, prompt: str) -> None:
    """
    Demonstrate faithfulness evaluation.
    """
    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    # Create explainer and get explanation
    explainer = SFATokenSHAP(
        backend=backend,
        budget=50,
        seed=42
    )
    
    print("\nGenerating explanation...")
    result = explainer.explain(prompt, train_refiner=True)
    
    # Create a simple model function for evaluation
    # (Returns similarity to reference output)
    def model_fn(subset_prompt: str) -> float:
        if not subset_prompt.strip():
            return 0.0
        output = backend.generate(subset_prompt)
        ref_emb = backend.get_embedding(result['coalition_results'][frozenset(range(len(result['tokens'])))].output_text)
        out_emb = backend.get_embedding(output)
        similarity = np.dot(ref_emb, out_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(out_emb) + 1e-10)
        return similarity
    
    # Evaluate faithfulness
    print("\n--- Faithfulness Analysis ---")
    faith_eval = FaithfulnessEvaluator(
        model_fn=model_fn,
        tokens=result['tokens'],
        baseline_output=1.0  # Full prompt has similarity 1.0
    )
    
    # Compare Stage 1 vs Stage 2
    print("\nStage 1 (Raw TokenSHAP):")
    faith1 = faith_eval.evaluate(result['stage1_raw'], k=3)
    print(f"  Comprehensiveness: {faith1.comprehensiveness:.4f}")
    print(f"  Sufficiency: {faith1.sufficiency:.4f}")
    print(f"  Deletion AUC: {faith1.deletion_curve_auc:.4f}")
    print(f"  Insertion AUC: {faith1.insertion_curve_auc:.4f}")
    
    print("\nStage 2 (SFA Refined):")
    faith2 = faith_eval.evaluate(result['stage2_raw'], k=3)
    print(f"  Comprehensiveness: {faith2.comprehensiveness:.4f}")
    print(f"  Sufficiency: {faith2.sufficiency:.4f}")
    print(f"  Deletion AUC: {faith2.deletion_curve_auc:.4f}")
    print(f"  Insertion AUC: {faith2.insertion_curve_auc:.4f}")
    
    # Compare improvement
    if faith2.comprehensiveness > faith1.comprehensiveness:
        print("\n✓ SFA refinement improved comprehensiveness")
    if faith2.sufficiency > faith1.sufficiency:
        print("✓ SFA refinement improved sufficiency")


def demo_plausibility(backend, prompt: str) -> None:
    """
    Demonstrate plausibility checks.
    """
    print("\n" + "=" * 60)
    print("PLAUSIBILITY ANALYSIS DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    # Create explainer and get explanation
    explainer = SFATokenSHAP(
        backend=backend,
        budget=30,
        seed=42
    )
    
    result = explainer.explain(prompt, train_refiner=True)
    
    print("\n--- Plausibility Analysis ---")
    plaus = PlausibilityEvaluator(result['tokens'], result['stage2_values'])
    
    stopword_ratio = plaus.stopword_suppression()
    question_importance = plaus.question_word_importance()
    
    print(f"\nStopword Suppression Ratio: {stopword_ratio:.2f}")
    if stopword_ratio > 1.5:
        print("  ✓ Content words have higher attributions than stopwords")
    else:
        print("  ⚠ Stopwords may have unexpectedly high attributions")
    
    print(f"\nQuestion Word Importance: {question_importance:.2f}")
    if question_importance > 0.5:
        print("  ✓ Question words are highly attributed (plausible)")
    elif question_importance > 0:
        print("  ~ Question words have moderate attribution")
    else:
        print("  - No question words in prompt")


def run_all_demos(backend_type: str, model_name: str):
    """Run all demonstration functions."""
    print("\n" + "=" * 70)
    print("SFA-TokenSHAP COMPLETE DEMONSTRATION")
    print("=" * 70)
    print(f"\nBackend: {backend_type}")
    print(f"Model: {model_name}")
    
    try:
        backend = create_backend(backend_type, model_name)
    except Exception as e:
        print(f"\n❌ Failed to create backend: {e}")
        print("\nPlease ensure your LLM backend is properly configured:")
        print("  - For Ollama: Run 'ollama serve' and 'ollama pull llama3.2'")
        print("  - For HuggingFace: Ensure transformers and torch are installed")
        return
    
    # Test prompts
    prompts = [
        "Why is the sky blue?",
        "The doctor said she would be back soon.",
        "What is machine learning and how does it work?",
    ]
    
    # Run basic explanation
    try:
        demo_basic_explanation(backend, prompts[0])
    except Exception as e:
        print(f"\n❌ Basic explanation failed: {e}")
        return
    
    # Run bias detection
    try:
        demo_bias_detection(backend, prompts[1])
    except Exception as e:
        print(f"\n⚠ Bias detection demo skipped: {e}")
    
    # Run consistency evaluation
    try:
        demo_consistency(backend, prompts[0], n_runs=2)
    except Exception as e:
        print(f"\n⚠ Consistency demo skipped: {e}")
    
    # Run faithfulness evaluation
    try:
        demo_faithfulness(backend, prompts[2])
    except Exception as e:
        print(f"\n⚠ Faithfulness demo skipped: {e}")
    
    # Run plausibility analysis
    try:
        demo_plausibility(backend, prompts[0])
    except Exception as e:
        print(f"\n⚠ Plausibility demo skipped: {e}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def run_offline_demo():
    """
    Run a demonstration without requiring an actual LLM.
    Uses mock data to show the evaluation components.
    """
    print("\n" + "=" * 70)
    print("SFA-TokenSHAP OFFLINE DEMONSTRATION")
    print("(No LLM required - using simulated data)")
    print("=" * 70)
    
    # Simulated data
    prompt = "Why is the sky blue?"
    tokens = ["Why", "is", "the", "sky", "blue", "?"]
    
    # Simulated attributions (Stage 1)
    stage1_values = np.array([0.10, 0.05, 0.02, 0.35, 0.45, 0.03])
    
    # Simulated refined attributions (Stage 2)
    # Note: SFA refinement typically suppresses stopwords
    stage2_values = np.array([0.12, 0.03, 0.01, 0.38, 0.44, 0.02])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    
    print("\n--- Stage 1 Attributions (TokenSHAP) ---")
    for token, value in zip(tokens, stage1_values):
        bar = "█" * int(value * 40)
        print(f"  {token:10s} {value:.4f} {bar}")
    
    print("\n--- Stage 2 Attributions (SFA Refined) ---")
    for token, value in zip(tokens, stage2_values):
        bar = "█" * int(value * 40)
        print(f"  {token:10s} {value:.4f} {bar}")
    
    # Plausibility analysis
    print("\n--- Plausibility Analysis ---")
    plaus = PlausibilityEvaluator(tokens, stage2_values)
    print(f"  Stopword Suppression Ratio: {plaus.stopword_suppression():.2f}")
    print(f"  Question Word Importance: {plaus.question_word_importance():.2f}")
    
    # Consistency analysis (simulated multiple runs)
    print("\n--- Consistency Analysis (simulated) ---")
    runs = [
        stage1_values,
        stage1_values + np.random.normal(0, 0.02, len(tokens)),
        stage1_values + np.random.normal(0, 0.02, len(tokens))
    ]
    consistency = ConsistencyEvaluator.evaluate(runs, k=2)
    print(f"  Spearman Correlation: {consistency.spearman_correlation:.4f}")
    print(f"  Top-2 Overlap: {consistency.top_k_overlap:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. Content words ('sky', 'blue') have highest attributions")
    print("2. Stopwords ('the', 'is') are appropriately suppressed")
    print("3. SFA refinement further reduces noise from common tokens")
    print("4. High consistency across simulated runs")


def main():
    parser = argparse.ArgumentParser(
        description="SFA-TokenSHAP Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Ollama backend
  python demo.py --backend ollama --model llama3.2
  
  # Run with HuggingFace backend
  python demo.py --backend huggingface --model meta-llama/Llama-3.2-1B-Instruct
  
  # Run offline demo (no LLM required)
  python demo.py --offline
        """
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "huggingface"],
        help="LLM backend to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Model name/path for the backend"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline demonstration without LLM"
    )
    
    args = parser.parse_args()
    
    if args.offline:
        run_offline_demo()
    else:
        run_all_demos(args.backend, args.model)


if __name__ == "__main__":
    main()
