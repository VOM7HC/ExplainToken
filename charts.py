"""Charts for SFA-TokenSHAP results"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_faithfulness(results: List[Dict], save_path: str = None):
    s1 = [r["faithfulness"]["stage1"] for r in results]
    s2 = [r["faithfulness"]["stage2"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = ["comprehensiveness", "sufficiency", "deletion_auc", "insertion_auc"]
    
    for ax, m in zip(axes.flat, metrics):
        v1 = [s[m] for s in s1]
        v2 = [s[m] for s in s2]
        x = np.arange(len(v1))
        ax.bar(x - 0.2, v1, 0.4, label="Stage 1", color="#3498db")
        ax.bar(x + 0.2, v2, 0.4, label="Stage 2", color="#e74c3c")
        ax.set_title(m.replace("_", " ").title())
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_consistency(results: Dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["spearman", "kendall", "top_k_overlap"]
    vals = [results.get(m, 0) for m in metrics]
    ax.bar(metrics, vals, color=["#2ecc71", "#9b59b6", "#f39c12"])
    ax.set_ylim(0, 1)
    ax.set_title("Consistency Metrics")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bias(results: List[Dict], save_path: str = None):
    gaps = [r["bias"]["ctf_gap"] for r in results]
    pairs = [r["bias"]["pair_score"] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(gaps)), gaps, color=["#e74c3c" if g > 0.3 else "#2ecc71" for g in gaps])
    axes[0].set_title("CTF Gap")
    axes[0].axhline(0.3, color="red", linestyle="--", alpha=0.5)
    
    axes[1].bar(range(len(pairs)), pairs, color=["#e74c3c" if abs(p - 0.5) > 0.2 else "#2ecc71" for p in pairs])
    axes[1].set_title("PAIR Score")
    axes[1].axhline(0.5, color="gray", linestyle="--")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tokens(tokens: List[str], s1: np.ndarray, s2: np.ndarray, save_path: str = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    x = np.arange(len(tokens))
    
    colors1 = ["#e74c3c" if v > 0 else "#3498db" for v in s1]
    colors2 = ["#e74c3c" if v > 0 else "#3498db" for v in s2]
    
    axes[0].bar(x, s1, color=colors1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tokens, rotation=45, ha="right")
    axes[0].set_title("Stage 1")
    
    axes[1].bar(x, s2, color=colors2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tokens, rotation=45, ha="right")
    axes[1].set_title("Stage 2")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_summary(results: List[Dict], save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Faithfulness comparison
    s1_comp = np.mean([r["faithfulness"]["stage1"]["comprehensiveness"] for r in results])
    s2_comp = np.mean([r["faithfulness"]["stage2"]["comprehensiveness"] for r in results])
    s1_suff = np.mean([r["faithfulness"]["stage1"]["sufficiency"] for r in results])
    s2_suff = np.mean([r["faithfulness"]["stage2"]["sufficiency"] for r in results])
    
    axes[0, 0].bar([0, 1], [s1_comp, s2_comp], color=["#3498db", "#e74c3c"])
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Stage 1", "Stage 2"])
    axes[0, 0].set_title("Avg Comprehensiveness")
    
    axes[0, 1].bar([0, 1], [s1_suff, s2_suff], color=["#3498db", "#e74c3c"])
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Stage 1", "Stage 2"])
    axes[0, 1].set_title("Avg Sufficiency")
    
    # Category distribution (for REALM)
    categories = [r.get("category", "unknown") for r in results]
    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    axes[1, 0].bar(cat_counts.keys(), cat_counts.values(), color="#9b59b6")
    axes[1, 0].set_title("Use Case Categories")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plausibility
    ratios = [r["plausibility"]["suppression_ratio"] for r in results]
    axes[1, 1].hist(ratios, bins=10, color="#2ecc71")
    axes[1, 1].set_title("Suppression Ratio Distribution")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
