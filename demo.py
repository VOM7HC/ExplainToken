#!/usr/bin/env python3
"""
SFA-TokenSHAP Demonstration
Usage: python demo.py --backend ollama --model qwen3:0.6b
"""

import argparse
import numpy as np
from sfa_tokenshap import SFATokenSHAP, OllamaBackend, HuggingFaceBackend
from evaluation import (
    FaithfulnessEvaluator,
    ConsistencyEvaluator,
    BiasDetectionEvaluator,
    PlausibilityEvaluator
)


def create_backend(backend_type: str, model_name: str, seed: int = 42):
    if backend_type == "ollama":
        return OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
    elif backend_type == "huggingface":
        return HuggingFaceBackend(model_name=model_name, temperature=0.0, seed=seed)
    raise ValueError(f"Unknown backend: {backend_type}")


def demo_basic_explanation(backend, prompt: str):
    print("\n" + "=" * 60)
    print("BASIC EXPLANATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    explainer = SFATokenSHAP(backend, budget=50, seed=42)
    
    print("\nRunning SFA-TokenSHAP...")
    result = explainer.explain(prompt, train_refiner=True, normalize=True)
    
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
    
    html = explainer.visualize(result, use_stage2=True)
    print("\n--- HTML Visualization ---")
    print(html[:300] + "..." if len(html) > 300 else html)
    
    return result


def demo_bias_detection(backend, prompt: str):
    print("\n" + "=" * 60)
    print("BIAS DETECTION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    explainer = SFATokenSHAP(backend, budget=30, seed=42)
    result = explainer.explain(prompt, train_refiner=True)
    
    bias_eval = BiasDetectionEvaluator(result['tokens'], result['stage2_values'])
    
    demo_tokens = bias_eval.identify_demographic_tokens()
    print("\nIdentified demographic tokens:")
    for group, indices in demo_tokens.items():
        tokens_in_group = [result['tokens'][i] for i in indices]
        print(f"  {group}: {tokens_in_group}")
    
    bias_result = bias_eval.evaluate()
    print(f"\nBias Detection Results:")
    print(f"  CTF Gap: {bias_result.ctf_gap:.4f}")
    print(f"  PAIR Score: {bias_result.pair_score:.4f}")
    print(f"  Detected Bias Type: {bias_result.detected_bias_type}")
    
    print("\n--- Interpretation ---")
    print(bias_result.interpret())
    
    return result


def demo_consistency(backend, prompt: str, n_runs: int = 3):
    print("\n" + "=" * 60)
    print("CONSISTENCY EVALUATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    print(f"Running {n_runs} passes...")
    
    attributions_list = []
    
    for i in range(n_runs):
        explainer = SFATokenSHAP(backend, budget=30, seed=42 + i)
        result = explainer.explain(prompt, train_refiner=False)
        attributions_list.append(result['stage1_values'])
        print(f"  Run {i+1} complete")
    
    consistency = ConsistencyEvaluator.evaluate(attributions_list, k=3)
    
    print(f"\n--- Consistency Analysis ---")
    print(f"  Spearman Correlation: {consistency.spearman_correlation:.4f}")
    print(f"  Kendall's Tau: {consistency.kendall_tau:.4f}")
    print(f"  Top-3 Overlap: {consistency.top_k_overlap:.4f}")
    
    if consistency.spearman_correlation > 0.8:
        print("\n✓ High consistency")
    elif consistency.spearman_correlation > 0.5:
        print("\n⚠ Moderate consistency")
    else:
        print("\n✗ Low consistency")


def demo_faithfulness(backend, prompt: str):
    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    explainer = SFATokenSHAP(backend, budget=50, seed=42)
    result = explainer.explain(prompt, train_refiner=True)
    
    # Model function for evaluation
    def model_fn(subset_prompt: str) -> float:
        if not subset_prompt.strip():
            return 0.0
        output = backend.generate(subset_prompt)
        ref_emb = backend.get_embedding(result['coalition_results'][frozenset(range(len(result['tokens'])))].output_text)
        out_emb = backend.get_embedding(output)
        return float(np.dot(ref_emb, out_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(out_emb) + 1e-10))
    
    faith_eval = FaithfulnessEvaluator(model_fn, result['tokens'], 1.0)
    
    print("\nStage 1 (Raw TokenSHAP):")
    faith1 = faith_eval.evaluate(result['stage1_raw'], k=3)
    print(f"  Comprehensiveness: {faith1.comprehensiveness:.4f}")
    print(f"  Sufficiency: {faith1.sufficiency:.4f}")
    
    print("\nStage 2 (SFA Refined):")
    faith2 = faith_eval.evaluate(result['stage2_raw'], k=3)
    print(f"  Comprehensiveness: {faith2.comprehensiveness:.4f}")
    print(f"  Sufficiency: {faith2.sufficiency:.4f}")
    
    if faith2.comprehensiveness > faith1.comprehensiveness:
        print("\n✓ SFA improved comprehensiveness")
    if faith2.sufficiency > faith1.sufficiency:
        print("✓ SFA improved sufficiency")
    
    # AOPC
    print("\n--- AOPC Analysis ---")
    aopc = faith_eval.aopc(result['stage2_raw'])
    print(f"  AOPC: {aopc['aopc']:.4f}")
    
    # Random baseline
    print("\n--- Random Baseline ---")
    baseline = faith_eval.random_baseline(n_runs=5, k=3)
    print(f"  Random Comprehensiveness: {baseline['random_comp_mean']:.4f} ± {baseline['random_comp_std']:.4f}")
    
    improvement = faith2.comprehensiveness - baseline['random_comp_mean']
    print(f"\n  Stage 2 improvement over random: +{improvement:.4f}")


def demo_plausibility(backend, prompt: str):
    print("\n" + "=" * 60)
    print("PLAUSIBILITY ANALYSIS DEMO")
    print("=" * 60)
    print(f"\nPrompt: '{prompt}'")
    
    explainer = SFATokenSHAP(backend, budget=30, seed=42)
    result = explainer.explain(prompt, train_refiner=True)
    
    plaus = PlausibilityEvaluator(result['tokens'], result['stage2_values'])
    
    stopword_ratio = plaus.stopword_suppression()
    question_importance = plaus.question_word_importance()
    
    print(f"\nStopword Suppression Ratio: {stopword_ratio:.2f}")
    if stopword_ratio > 1.5:
        print("  ✓ Content words have higher attributions")
    else:
        print("  ⚠ Stopwords may have high attributions")
    
    print(f"\nQuestion Word Importance: {question_importance:.2f}")
    if question_importance > 0.5:
        print("  ✓ Question words are highly attributed")


def run_all_demos(backend_type: str, model_name: str):
    print("\n" + "=" * 70)
    print("SFA-TokenSHAP DEMONSTRATION")
    print("=" * 70)
    print(f"\nBackend: {backend_type}")
    print(f"Model: {model_name}")
    
    backend = create_backend(backend_type, model_name)
    
    prompts = [
        "Why is the sky blue?",
        "The doctor said she would be back soon.",
        "What is machine learning?",
    ]
    
    demo_basic_explanation(backend, prompts[0])
    demo_bias_detection(backend, prompts[1])
    demo_consistency(backend, prompts[0], n_runs=2)
    demo_faithfulness(backend, prompts[2])
    demo_plausibility(backend, prompts[0])
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def run_offline_demo():
    """Demo without LLM using simulated data"""
    print("\n" + "=" * 70)
    print("SFA-TokenSHAP OFFLINE DEMO (Simulated Data)")
    print("=" * 70)
    
    prompt = "Why is the sky blue?"
    tokens = ["Why", "is", "the", "sky", "blue", "?"]
    
    stage1_values = np.array([0.10, 0.05, 0.02, 0.35, 0.45, 0.03])
    stage2_values = np.array([0.12, 0.03, 0.01, 0.38, 0.44, 0.02])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    
    print("\n--- Stage 1 Attributions ---")
    for token, value in zip(tokens, stage1_values):
        bar = "█" * int(value * 40)
        print(f"  {token:10s} {value:.4f} {bar}")
    
    print("\n--- Stage 2 Attributions ---")
    for token, value in zip(tokens, stage2_values):
        bar = "█" * int(value * 40)
        print(f"  {token:10s} {value:.4f} {bar}")
    
    # Plausibility
    print("\n--- Plausibility ---")
    plaus = PlausibilityEvaluator(tokens, stage2_values)
    print(f"  Stopword Suppression: {plaus.stopword_suppression():.2f}")
    print(f"  Question Word Importance: {plaus.question_word_importance():.2f}")
    
    # Consistency
    print("\n--- Consistency (simulated) ---")
    runs = [
        stage1_values,
        stage1_values + np.random.normal(0, 0.02, len(tokens)),
        stage1_values + np.random.normal(0, 0.02, len(tokens))
    ]
    consistency = ConsistencyEvaluator.evaluate(runs, k=2)
    print(f"  Spearman: {consistency.spearman_correlation:.4f}")
    print(f"  Top-2 Overlap: {consistency.top_k_overlap:.4f}")
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("1. Content words ('sky', 'blue') have highest attributions")
    print("2. Stopwords ('the', 'is') are suppressed")
    print("3. SFA refinement reduces noise from common tokens")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SFA-TokenSHAP Demo")
    
    parser.add_argument("--backend", type=str, default="ollama",
                       choices=["ollama", "huggingface"])
    parser.add_argument("--model", type=str, default="qwen3:0.6b")
    parser.add_argument("--offline", action="store_true")
    
    args = parser.parse_args()
    
    if args.offline:
        run_offline_demo()
    else:
        run_all_demos(args.backend, args.model)


if __name__ == "__main__":
    main()