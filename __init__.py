"""
SFA-TokenSHAP: Two-Stage Shapley Refinement for LLM Token Attribution
"""

__version__ = "0.1.0"
__author__ = "Wesley Vo"

from .sfa_tokenshap import (
    SFATokenSHAP,
    Stage1TokenSHAP,
    Stage2SFARefinement,
    OllamaBackend,
    HuggingFaceBackend,
    ValueFunction,
    CoalitionCache,
    CoalitionGenerator,
    PromptRenderer,
    CoalitionStrategy,
    ShapleyResult,
    CoalitionResult,
    SFAFeature,
    LinguisticExtractor,
)

from .evaluation import (
    FaithfulnessEvaluator,
    FaithfulnessResult,
    ConsistencyEvaluator,
    ConsistencyResult,
    BiasDetectionEvaluator,
    BiasResult,
    PlausibilityEvaluator,
    evaluate_explanation_quality,
)

from .sfa_ensemble import (
    SFAEnsemble,
    MultiModelSFA,
    ModelType,
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    KFoldCV,
    FoldResult,
    EnsembleResult,
    create_model,
)

__all__ = [
    # Main explainer
    "SFATokenSHAP",
    "Stage1TokenSHAP",
    "Stage2SFARefinement",
    
    # Backends
    "OllamaBackend",
    "HuggingFaceBackend",
    
    # Core classes
    "ValueFunction",
    "CoalitionCache",
    "CoalitionGenerator",
    "PromptRenderer",
    "CoalitionStrategy",
    "LinguisticExtractor",
    
    # Data classes
    "ShapleyResult",
    "CoalitionResult",
    "SFAFeature",
    
    # Ensemble
    "SFAEnsemble",
    "MultiModelSFA",
    "ModelType",
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
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
    "evaluate_explanation_quality",
]