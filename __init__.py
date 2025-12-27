"""
SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution

A Python implementation combining TokenSHAP and SFA methodologies for
interpretable LLM explanations.

Main Components:
- SFATokenSHAP: Main explainer class
- OllamaBackend / HuggingFaceBackend: LLM backends
- SFAEnsemble: Ensemble learning with Shapley feature augmentation
- Evaluation metrics: Faithfulness, consistency, bias detection

Usage:
    from sfa_tokenshap import SFATokenSHAP, OllamaBackend
    
    backend = OllamaBackend(model_name="llama3.2")
    explainer = SFATokenSHAP(backend, budget=50)
    result = explainer.explain("Why is the sky blue?")
"""

__version__ = "0.1.0"
__author__ = "Wesley Vo"
__license__ = "MIT"

# Main explainer components
from .sfa_tokenshap import (
    # Main classes
    SFATokenSHAP,
    Stage1TokenSHAP,
    Stage2SFARefinement,
    
    # Backends
    LLMBackend,
    OllamaBackend,
    HuggingFaceBackend,
    
    # Value functions
    ValueFunction,
    ValueFunctionType,
    
    # Coalition handling
    CoalitionGenerator,
    CoalitionCache,
    CoalitionStrategy,
    PromptRenderer,
    
    # Data classes
    TokenInfo,
    CoalitionResult,
    ShapleyResult,
    SFAFeatureVector,
    
    # Utility functions
    compute_exact_shapley,
    sliding_window_shapley,
)

# Ensemble training
from .sfa_ensemble import (
    SFAEnsemble,
    MultiModelSFA,
    ModelType,
    
    # Individual model wrappers
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    AdaBoostModel,
    
    # Utilities
    KFoldCV,
    FoldResult,
    EnsembleResult,
    create_model,
)

# Evaluation metrics
from .evaluation import (
    # Faithfulness
    FaithfulnessEvaluator,
    FaithfulnessResult,
    
    # Consistency
    ConsistencyEvaluator,
    ConsistencyResult,
    
    # Bias detection
    BiasDetectionEvaluator,
    BiasResult,
    
    # Plausibility
    PlausibilityEvaluator,
    
    # Quality metrics
    ExplanationQualityMetrics,
    evaluate_explanation_quality,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Main explainer
    "SFATokenSHAP",
    "Stage1TokenSHAP",
    "Stage2SFARefinement",
    
    # Backends
    "LLMBackend",
    "OllamaBackend",
    "HuggingFaceBackend",
    
    # Value functions
    "ValueFunction",
    "ValueFunctionType",
    
    # Coalition handling
    "CoalitionGenerator",
    "CoalitionCache",
    "CoalitionStrategy",
    "PromptRenderer",
    
    # Data classes
    "TokenInfo",
    "CoalitionResult",
    "ShapleyResult",
    "SFAFeatureVector",
    
    # Utility functions
    "compute_exact_shapley",
    "sliding_window_shapley",
    
    # Ensemble
    "SFAEnsemble",
    "MultiModelSFA",
    "ModelType",
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "AdaBoostModel",
    "KFoldCV",
    "FoldResult",
    "EnsembleResult",
    "create_model",
    
    # Evaluation
    "FaithfulnessEvaluator",
    "FaithfulnessResult",
    "ConsistencyEvaluator",
    "ConsistencyResult",
    "BiasDetectionEvaluator",
    "BiasResult",
    "PlausibilityEvaluator",
    "ExplanationQualityMetrics",
    "evaluate_explanation_quality",
]
