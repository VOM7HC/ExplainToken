"""Run SFA-TokenSHAP evaluation on real data"""

import argparse
import json
import os
import numpy as np
from sfa_tokenshap import OllamaBackend, SFATokenSHAP
from evaluation import FaithfulnessEvaluator, ConsistencyEvaluator, BiasEvaluator, PlausibilityEvaluator
from data_loader import load_data
from charts import plot_faithfulness, plot_consistency, plot_bias, plot_tokens, plot_summary


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/charts", exist_ok=True)
    
    backend = OllamaBackend(model=args.model, seed=args.seed)
    explainer = SFATokenSHAP(backend, budget_per_token=args.budget_per_token, seed=args.seed, sentence_index=args.sentence_index)

    samples = load_data(args.dataset, n=args.n_samples, seed=args.seed)
    print(f"Loaded {len(samples)} samples")

    # Filter to specific sample if requested
    if args.sample_index is not None:
        if args.sample_index >= len(samples):
            print(f"Error: sample_index {args.sample_index} out of range (0-{len(samples)-1})")
            return
        samples = [samples[args.sample_index]]
        print(f"Processing only sample {args.sample_index}")

    results = []

    for i, sample in enumerate(samples):
        print(f"\n[Sample {i+1}/{len(samples)}] {sample.prompt[:50]}...")
        
        exp = explainer.explain(sample.prompt)
        tokens, s1, s2, weights = exp["tokens"], exp["stage1"], exp["stage2"], exp["weights"]
        
        ref_emb = backend.embed(backend.generate(sample.prompt))
        def value_fn(text):
            if not text or not text.strip():
                return 0.0
            out = backend.generate(text)
            if not out or not out.strip():
                return 0.0
            emb = backend.embed(out)
            if emb.size == 0:
                return 0.0
            return float(np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-10))
        
        faith_eval = FaithfulnessEvaluator(value_fn, tokens, 1.0)
        
        result = {
            "prompt": sample.prompt,
            "category": sample.category,
            "tokens": tokens,
            "stage1": s1.tolist(),
            "stage2": s2.tolist(),
            "faithfulness": {
                "stage1": faith_eval.evaluate(s1),
                "stage2": faith_eval.evaluate(s2)
            },
            "bias": BiasEvaluator.evaluate(tokens, s2),
            "plausibility": PlausibilityEvaluator.evaluate(tokens, s2, weights)
        }
        results.append(result)
        
        plot_tokens(tokens, s1, s2, f"{args.output_dir}/charts/tokens_{i}.png")
    
    # Consistency (multiple runs on first sample)
    if len(samples) > 0:
        attrs = []
        for seed in range(args.seed, args.seed + 3):
            exp = SFATokenSHAP(backend, budget_per_token=args.budget_per_token, seed=seed, sentence_index=args.sentence_index).explain(samples[0].prompt, verbose=False)
            attrs.append(exp["stage2"])
        consistency = ConsistencyEvaluator.evaluate(attrs)
    else:
        consistency = {}
    
    # Charts
    plot_faithfulness(results, f"{args.output_dir}/charts/faithfulness.png")
    plot_consistency(consistency, f"{args.output_dir}/charts/consistency.png")
    plot_bias(results, f"{args.output_dir}/charts/bias.png")
    plot_summary(results, f"{args.output_dir}/charts/summary.png")
    
    # Save
    output = {"results": results, "consistency": consistency}
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--dataset", default="realm", choices=["realm", "reddit", "news"])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--sample-index", type=int, default=None, help="Process only this sample index (0-based)")
    parser.add_argument("--sentence-index", type=int, default=None, help="Process only this sentence index (0-based)")
    parser.add_argument("--budget-per-token", type=int, default=5, help="Coalitions per token for Monte Carlo Shapley")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    run(parser.parse_args())