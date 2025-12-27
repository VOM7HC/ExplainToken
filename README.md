# SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution

A Python implementation of SFA-TokenSHAP for interpretable Large Language Model explanations, combining:

- **TokenSHAP**: Monte Carlo Shapley value estimation for token attribution
- **llmSHAP**: Principled approach to handling LLM stochasticity  
- **SFA (Shapley-based Feature Augmentation)**: Two-stage ensemble learning with SHAP feature augmentation

## 🎯 Key Features

- **Two-Stage Explanation Pipeline**:
  - Stage 1: TokenSHAP Monte Carlo Shapley estimation
  - Stage 2: SFA-style refinement using learned feature augmentation

- **Multiple LLM Backends**:
  - Ollama (local models)
  - HuggingFace Transformers

- **Comprehensive Evaluation**:
  - Faithfulness metrics (comprehensiveness, sufficiency)
  - Consistency evaluation
  - Bias detection (CTF Gap, PAIR scores)
  - Plausibility analysis

- **Ensemble Learning**:
  - K-fold cross-validation
  - Multiple base learners (XGBoost, LightGBM, CatBoost, AdaBoost)
  - Optuna hyperparameter tuning

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sfa-tokenshap.git
cd sfa-tokenshap

# Install dependencies
pip install -r requirements.txt

# For Ollama backend
pip install ollama
ollama pull llama3.2

# For HuggingFace backend
pip install transformers torch sentence-transformers
```

## 🚀 Quick Start

### Basic Usage

```python
from sfa_tokenshap import SFATokenSHAP, OllamaBackend

# Initialize backend
backend = OllamaBackend(model_name="llama3.2")

# Create explainer
explainer = SFATokenSHAP(
    backend=backend,
    budget=50,  # Number of coalition evaluations
    seed=42
)

# Explain a prompt
result = explainer.explain("Why is the sky blue?")

# Access results
print("Tokens:", result['tokens'])
print("Stage 1 values:", result['stage1_values'])
print("Stage 2 values:", result['stage2_values'])

# Visualize
html = explainer.visualize(result)
```

### With HuggingFace Backend

```python
from sfa_tokenshap import SFATokenSHAP, HuggingFaceBackend

backend = HuggingFaceBackend(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    temperature=0.0
)

explainer = SFATokenSHAP(backend=backend, budget=50)
result = explainer.explain("What is machine learning?")
```

### Running the Demo

```bash
# With Ollama
python demo.py --backend ollama --model llama3.2

# With HuggingFace
python demo.py --backend huggingface --model meta-llama/Llama-3.2-1B-Instruct

# Offline demo (no LLM required)
python demo.py --offline
```

## 📐 Architecture

```
SFA-TokenSHAP Pipeline
══════════════════════

Stage 0: Preparation
├── Tokenize input prompt
├── Configure coalition strategy (delete/mask/placeholder)
└── Set up value function (embedding similarity/logprob/task scorer)

Stage 1: TokenSHAP Monte Carlo
├── Generate coalitions (leave-one-out + stratified random)
├── Evaluate each coalition via LLM inference
├── Cache results for efficiency
└── Compute Shapley values: φ_i = E[v(S)|i∈S] - E[v(S)|i∉S]

Stage 2: SFA Refinement
├── Build feature vectors per token:
│   ├── Stage 1 attribution (φ^(1)_i)
│   ├── Token metadata (position, length, type)
│   ├── Coalition statistics (variance, leave-one-out delta)
│   └── Interaction signals (neighbor synergy)
├── Compute faithfulness-based training target
├── Train lightweight refiner (Ridge/GradientBoosting/MLP)
└── Generate refined attributions (φ^(final)_i)

Output: Token importance map with Stage 1 and Stage 2 values
```

## 🔧 Configuration Options

### Coalition Strategies

```python
from sfa_tokenshap import CoalitionStrategy

# Remove tokens entirely
explainer = SFATokenSHAP(coalition_strategy=CoalitionStrategy.DELETE)

# Replace with placeholder
explainer = SFATokenSHAP(coalition_strategy=CoalitionStrategy.PLACEHOLDER)

# Replace with [MASK]
explainer = SFATokenSHAP(coalition_strategy=CoalitionStrategy.MASK)
```

### Value Functions

```python
from sfa_tokenshap import ValueFunctionType

# Embedding cosine similarity (default)
explainer = SFATokenSHAP(value_function_type=ValueFunctionType.EMBEDDING_SIMILARITY)

# Custom task scorer
def my_scorer(output, reference):
    # Your scoring logic
    return score

explainer = SFATokenSHAP(value_function_type=ValueFunctionType.TASK_SCORER)
```

### Refiner Types

```python
# Ridge regression (interpretable, default)
explainer = SFATokenSHAP(refiner_type="ridge")

# Gradient boosting (stronger)
explainer = SFATokenSHAP(refiner_type="gradient_boosting")

# Small MLP
explainer = SFATokenSHAP(refiner_type="mlp")
```

## 📊 Evaluation

### Faithfulness Evaluation

```python
from sfa_tokenshap import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(
    model_fn=my_model_function,
    tokens=result['tokens'],
    baseline_output=baseline_score
)

faith_result = evaluator.evaluate(result['stage2_values'], k=3)
print(f"Comprehensiveness: {faith_result.comprehensiveness}")
print(f"Sufficiency: {faith_result.sufficiency}")
```

### Consistency Evaluation

```python
from sfa_tokenshap import ConsistencyEvaluator

# Run explanation multiple times
attributions_list = [run1_values, run2_values, run3_values]

consistency = ConsistencyEvaluator.evaluate(attributions_list, k=3)
print(f"Spearman: {consistency.spearman_correlation}")
print(f"Top-3 Overlap: {consistency.top_k_overlap}")
```

### Bias Detection

```python
from sfa_tokenshap import BiasDetectionEvaluator

bias_eval = BiasDetectionEvaluator(result['tokens'], result['stage2_values'])
bias_result = bias_eval.evaluate()

print(f"CTF Gap: {bias_result.ctf_gap}")
print(f"PAIR Score: {bias_result.pair_score}")
print(f"Detected Bias: {bias_result.detected_bias_type}")
```

## 🏗️ SFA Ensemble Training

For tabular data tasks (not LLM explanation), use the SFA ensemble:

```python
from sfa_tokenshap import SFAEnsemble, ModelType

# Create ensemble
ensemble = SFAEnsemble(
    base_model_type=ModelType.XGBOOST,
    task="classification",
    n_folds=5,
    use_optuna=True  # Enable hyperparameter tuning
)

# Train
result = ensemble.fit(X_train, y_train)

# Predict (averages base + P-augmented + SHAP-augmented + P+SHAP-augmented)
predictions = ensemble.predict(X_test)

# Get all model predictions
all_preds = ensemble.predict(X_test, return_all=True)
# Returns: {'base', 'p_augmented', 'shap_augmented', 'pshap_augmented', 'sfa'}
```

### Multi-Model Ensemble

```python
from sfa_tokenshap import MultiModelSFA, ModelType

# Use multiple base learner types
multi_ensemble = MultiModelSFA(
    model_types=[
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.CATBOOST,
        ModelType.ADABOOST
    ],
    task="classification",
    n_folds=5
)

multi_ensemble.fit(X_train, y_train)
predictions = multi_ensemble.predict(X_test)
```

## 📚 References

1. **TokenSHAP**: Horovicz & Goldshmidt (2024). "TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation"

2. **llmSHAP**: Naudot et al. (2025). "llmSHAP: A Principled Approach to LLM Explainability"

3. **SFA**: Antwarg et al. (2023). "Shapley-based Feature Augmentation" (Information Fusion)

4. **SHAP**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"

## 🔬 For Your Thesis

This implementation is designed to support your thesis on:
> "Enhancing Large Language Model Interpretability Using Shapley Values: Token-Level Analysis for Improving Transparency and Bias Detection"

Key thesis contributions supported:
- TokenSHAP methodology enhanced with SFA
- Faithfulness evaluation framework
- Bias detection using Shapley values
- EU AI Act Article 13 compliance metrics

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

**Author**: Wesley Vo  
**Thesis Advisor**: [Your Advisor]  
**University**: [Your University]
