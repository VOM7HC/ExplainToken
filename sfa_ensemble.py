"""
SFA Ensemble Training Module

Implements the two-stage ensemble learning approach from:
"Shapley-based Feature Augmentation" (Antwarg et al., 2023)

This module provides:
- K-fold cross-validation for out-of-fold predictions
- Multiple base learner support (XGBoost, LightGBM, CatBoost, AdaBoost)
- Hyperparameter tuning with Optuna
- Feature augmentation with Shapley values
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum
import pickle
import json
from pathlib import Path


class ModelType(Enum):
    """Supported model types for ensemble."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ADABOOST = "adaboost"
    RANDOM_FOREST = "random_forest"
    RIDGE = "ridge"


@dataclass
class FoldResult:
    """Result from a single fold of cross-validation."""
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    predictions: np.ndarray          # Out-of-fold predictions
    shapley_values: np.ndarray       # SHAP values for predictions
    model: Any                       # Trained model
    
    
@dataclass
class EnsembleResult:
    """Result from full ensemble training."""
    fold_results: List[FoldResult]
    oof_predictions: np.ndarray       # Combined OOF predictions
    oof_shapley_values: np.ndarray    # Combined OOF SHAP values
    models: List[Any]                 # All trained models


class BaseModel(ABC):
    """Abstract base class for ensemble models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        pass
    
    @abstractmethod
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values for predictions."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Install with: pip install xgboost")
        
        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        
        self.model.fit(X, y)
        self._explainer = None  # Reset explainer after fitting
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        try:
            import shap
        except ImportError:
            raise ImportError("shap required. Install with: pip install shap")
        
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        
        shap_values = self._explainer.shap_values(X)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            if len(shap_values) == 2:
                return shap_values[1]
            # For multi-class, stack
            return np.stack(shap_values, axis=-1)
        
        return shap_values


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")
        
        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        
        self.model.fit(X, y)
        self._explainer = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        try:
            import shap
        except ImportError:
            raise ImportError("shap required. Install with: pip install shap")
        
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        
        shap_values = self._explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                return shap_values[1]
            return np.stack(shap_values, axis=-1)
        
        return shap_values


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise ImportError("catboost required. Install with: pip install catboost")
        
        # Suppress verbose output by default
        if 'verbose' not in self.params:
            self.params['verbose'] = False
        
        if self.task == "classification":
            self.model = CatBoostClassifier(**self.params)
        else:
            self.model = CatBoostRegressor(**self.params)
        
        self.model.fit(X, y)
        self._explainer = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        try:
            import shap
        except ImportError:
            raise ImportError("shap required. Install with: pip install shap")
        
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        
        shap_values = self._explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                return shap_values[1]
            return np.stack(shap_values, axis=-1)
        
        return shap_values


class AdaBoostModel(BaseModel):
    """AdaBoost model wrapper."""
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        
        if self.task == "classification":
            self.model = AdaBoostClassifier(**self.params)
        else:
            self.model = AdaBoostRegressor(**self.params)
        
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        AdaBoost SHAP values using KernelExplainer.
        Note: This is slower than TreeExplainer.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap required. Install with: pip install shap")
        
        # Use KernelExplainer for AdaBoost
        if self.task == "classification":
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1], 
                shap.sample(X, min(100, len(X)))
            )
        else:
            explainer = shap.KernelExplainer(
                self.model.predict, 
                shap.sample(X, min(100, len(X)))
            )
        
        return explainer.shap_values(X)


def create_model(model_type: ModelType, task: str = "classification", **params) -> BaseModel:
    """Factory function to create models."""
    if model_type == ModelType.XGBOOST:
        return XGBoostModel(task=task, **params)
    elif model_type == ModelType.LIGHTGBM:
        return LightGBMModel(task=task, **params)
    elif model_type == ModelType.CATBOOST:
        return CatBoostModel(task=task, **params)
    elif model_type == ModelType.ADABOOST:
        return AdaBoostModel(task=task, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class KFoldCV:
    """K-Fold Cross-Validation manager."""
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        from sklearn.model_selection import KFold
        
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        return list(kfold.split(X))


class SFAEnsemble:
    """
    SFA (Shapley-based Feature Augmentation) Ensemble
    
    Implements the two-stage learning approach:
    Stage 1: Train base models with K-fold CV, collect OOF predictions and SHAP values
    Stage 2: Train augmented models using original features + SHAP values + predictions
    """
    
    def __init__(
        self,
        base_model_type: ModelType = ModelType.XGBOOST,
        task: str = "classification",
        n_folds: int = 5,
        use_optuna: bool = False,
        n_optuna_trials: int = 15,
        random_state: Optional[int] = 42
    ):
        self.base_model_type = base_model_type
        self.task = task
        self.n_folds = n_folds
        self.use_optuna = use_optuna
        self.n_optuna_trials = n_optuna_trials
        self.random_state = random_state
        
        # Stage 1 components
        self.stage1_models: List[BaseModel] = []
        self.stage1_params: Dict = {}
        
        # Stage 2 components
        self.stage2_p_model: Optional[BaseModel] = None      # P augmented
        self.stage2_shap_model: Optional[BaseModel] = None   # SHAP augmented
        self.stage2_pshap_model: Optional[BaseModel] = None  # P+SHAP augmented
        
        # Training artifacts
        self.oof_predictions: Optional[np.ndarray] = None
        self.oof_shapley_values: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None
        
        self._is_fitted = False
    
    def _get_default_params(self, model_type: ModelType) -> Dict:
        """Get default parameters for model type."""
        if model_type == ModelType.XGBOOST:
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state
            }
        elif model_type == ModelType.LIGHTGBM:
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'verbosity': -1
            }
        elif model_type == ModelType.CATBOOST:
            return {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': self.random_state,
                'verbose': False
            }
        elif model_type == ModelType.ADABOOST:
            return {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': self.random_state
            }
        else:
            return {}
    
    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType
    ) -> Dict:
        """Tune hyperparameters using Optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            warnings.warn("Optuna not installed. Using default parameters.")
            return self._get_default_params(model_type)
        
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            if model_type == ModelType.XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state
                }
                model = create_model(model_type, self.task, **params)
            elif model_type == ModelType.LIGHTGBM:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state,
                    'verbosity': -1
                }
                model = create_model(model_type, self.task, **params)
            elif model_type == ModelType.CATBOOST:
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'random_seed': self.random_state,
                    'verbose': False
                }
                model = create_model(model_type, self.task, **params)
            else:
                params = self._get_default_params(model_type)
                model = create_model(model_type, self.task, **params)
            
            # Cross-validation score
            model.fit(X, y)
            if self.task == "classification":
                preds = model.predict_proba(X)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    preds = preds[:, 1]
                from sklearn.metrics import roc_auc_score
                try:
                    score = roc_auc_score(y, preds)
                except:
                    score = 0.5
            else:
                preds = model.predict(X)
                from sklearn.metrics import r2_score
                score = r2_score(y, preds)
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        return best_params
    
    def fit_stage1(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict] = None
    ) -> EnsembleResult:
        """
        Stage 1: Train base models with K-fold CV.
        
        Collects out-of-fold predictions and SHAP values.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Tune or use provided parameters
        if model_params is None:
            if self.use_optuna:
                model_params = self._tune_hyperparameters(X, y, self.base_model_type)
            else:
                model_params = self._get_default_params(self.base_model_type)
        
        self.stage1_params = model_params
        
        # Initialize arrays for OOF predictions and SHAP values
        if self.task == "classification":
            n_classes = len(np.unique(y))
            if n_classes == 2:
                self.oof_predictions = np.zeros(n_samples)
            else:
                self.oof_predictions = np.zeros((n_samples, n_classes))
        else:
            self.oof_predictions = np.zeros(n_samples)
        
        self.oof_shapley_values = np.zeros((n_samples, n_features))
        
        # K-Fold cross-validation
        kfold = KFoldCV(n_splits=self.n_folds, random_state=self.random_state)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Create and train model
            model = create_model(self.base_model_type, self.task, **model_params)
            model.fit(X_train, y_train)
            
            # Get predictions
            if self.task == "classification":
                proba = model.predict_proba(X_val)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    self.oof_predictions[val_idx] = proba[:, 1]
                else:
                    self.oof_predictions[val_idx] = proba
            else:
                self.oof_predictions[val_idx] = model.predict(X_val)
            
            # Get SHAP values
            shap_vals = model.get_shap_values(X_val)
            if shap_vals.ndim == 3:
                # Multi-class: average across classes or take max class
                shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
            self.oof_shapley_values[val_idx] = shap_vals
            
            # Store model
            self.stage1_models.append(model)
            
            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                predictions=self.oof_predictions[val_idx].copy(),
                shapley_values=self.oof_shapley_values[val_idx].copy(),
                model=model
            ))
        
        return EnsembleResult(
            fold_results=fold_results,
            oof_predictions=self.oof_predictions,
            oof_shapley_values=self.oof_shapley_values,
            models=self.stage1_models
        )
    
    def fit_stage2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict] = None
    ):
        """
        Stage 2: Train augmented models.
        
        Three augmentation strategies:
        - P augmented: Original features + OOF predictions
        - SHAP augmented: Original features + SHAP values
        - P+SHAP augmented: Original features + OOF predictions + SHAP values
        """
        if self.oof_predictions is None or self.oof_shapley_values is None:
            raise ValueError("Stage 1 must be fitted first. Call fit_stage1().")
        
        if model_params is None:
            model_params = self.stage1_params
        
        # Prepare predictions for augmentation
        if self.oof_predictions.ndim == 1:
            oof_pred_features = self.oof_predictions.reshape(-1, 1)
        else:
            oof_pred_features = self.oof_predictions
        
        # P augmented: X + predictions
        X_p = np.hstack([X, oof_pred_features])
        self.stage2_p_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_p_model.fit(X_p, y)
        
        # SHAP augmented: X + SHAP values
        X_shap = np.hstack([X, self.oof_shapley_values])
        self.stage2_shap_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_shap_model.fit(X_shap, y)
        
        # P+SHAP augmented: X + predictions + SHAP values
        X_pshap = np.hstack([X, oof_pred_features, self.oof_shapley_values])
        self.stage2_pshap_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_pshap_model.fit(X_pshap, y)
        
        self._is_fitted = True
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict] = None
    ) -> EnsembleResult:
        """
        Fit the complete SFA ensemble.
        
        Performs both Stage 1 and Stage 2 training.
        """
        result = self.fit_stage1(X, y, model_params)
        self.fit_stage2(X, y, model_params)
        return result
    
    def predict(
        self,
        X: np.ndarray,
        return_all: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions using the SFA ensemble.
        
        Averages predictions across:
        - Base model (average of Stage 1 fold models)
        - P augmented model
        - SHAP augmented model
        - P+SHAP augmented model
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Stage 1: Average predictions from all fold models
        stage1_preds = []
        stage1_shap = []
        
        for model in self.stage1_models:
            if self.task == "classification":
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    stage1_preds.append(proba[:, 1])
                else:
                    stage1_preds.append(proba)
            else:
                stage1_preds.append(model.predict(X))
            
            shap_vals = model.get_shap_values(X)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
            stage1_shap.append(shap_vals)
        
        # Average Stage 1 predictions and SHAP values
        avg_pred = np.mean(stage1_preds, axis=0)
        avg_shap = np.mean(stage1_shap, axis=0)
        
        # Prepare features for Stage 2 models
        if avg_pred.ndim == 1:
            pred_features = avg_pred.reshape(-1, 1)
        else:
            pred_features = avg_pred
        
        X_p = np.hstack([X, pred_features])
        X_shap = np.hstack([X, avg_shap])
        X_pshap = np.hstack([X, pred_features, avg_shap])
        
        # Stage 2 predictions
        if self.task == "classification":
            p_pred = self.stage2_p_model.predict_proba(X_p)
            shap_pred = self.stage2_shap_model.predict_proba(X_shap)
            pshap_pred = self.stage2_pshap_model.predict_proba(X_pshap)
            
            if p_pred.ndim == 2 and p_pred.shape[1] == 2:
                p_pred = p_pred[:, 1]
                shap_pred = shap_pred[:, 1]
                pshap_pred = pshap_pred[:, 1]
        else:
            p_pred = self.stage2_p_model.predict(X_p)
            shap_pred = self.stage2_shap_model.predict(X_shap)
            pshap_pred = self.stage2_pshap_model.predict(X_pshap)
        
        # Final SFA prediction: average of all models
        if avg_pred.ndim == 1:
            final_pred = (avg_pred + p_pred + shap_pred + pshap_pred) / 4
        else:
            final_pred = (avg_pred + p_pred + shap_pred + pshap_pred) / 4
        
        if return_all:
            return {
                'base': avg_pred,
                'p_augmented': p_pred,
                'shap_augmented': shap_pred,
                'pshap_augmented': pshap_pred,
                'sfa': final_pred
            }
        
        return final_pred
    
    def save(self, path: Union[str, Path]):
        """Save the trained ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'base_model_type': self.base_model_type.value,
            'task': self.task,
            'n_folds': self.n_folds,
            'stage1_params': self.stage1_params
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)
        
        # Save models
        for i, model in enumerate(self.stage1_models):
            with open(path / f'stage1_model_{i}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        with open(path / 'stage2_p_model.pkl', 'wb') as f:
            pickle.dump(self.stage2_p_model, f)
        
        with open(path / 'stage2_shap_model.pkl', 'wb') as f:
            pickle.dump(self.stage2_shap_model, f)
        
        with open(path / 'stage2_pshap_model.pkl', 'wb') as f:
            pickle.dump(self.stage2_pshap_model, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SFAEnsemble':
        """Load a trained ensemble from disk."""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        ensemble = cls(
            base_model_type=ModelType(config['base_model_type']),
            task=config['task'],
            n_folds=config['n_folds']
        )
        ensemble.stage1_params = config['stage1_params']
        
        # Load Stage 1 models
        for i in range(config['n_folds']):
            with open(path / f'stage1_model_{i}.pkl', 'rb') as f:
                ensemble.stage1_models.append(pickle.load(f))
        
        # Load Stage 2 models
        with open(path / 'stage2_p_model.pkl', 'rb') as f:
            ensemble.stage2_p_model = pickle.load(f)
        
        with open(path / 'stage2_shap_model.pkl', 'rb') as f:
            ensemble.stage2_shap_model = pickle.load(f)
        
        with open(path / 'stage2_pshap_model.pkl', 'rb') as f:
            ensemble.stage2_pshap_model = pickle.load(f)
        
        ensemble._is_fitted = True
        
        return ensemble


class MultiModelSFA:
    """
    Multi-model SFA ensemble using multiple base learner types.
    
    Combines XGBoost, LightGBM, CatBoost, and AdaBoost for
    even stronger ensemble predictions.
    """
    
    def __init__(
        self,
        model_types: List[ModelType] = None,
        task: str = "classification",
        n_folds: int = 5,
        use_optuna: bool = False,
        random_state: Optional[int] = 42
    ):
        if model_types is None:
            model_types = [
                ModelType.XGBOOST,
                ModelType.LIGHTGBM,
                ModelType.CATBOOST,
                ModelType.ADABOOST
            ]
        
        self.model_types = model_types
        self.task = task
        self.n_folds = n_folds
        self.use_optuna = use_optuna
        self.random_state = random_state
        
        self.ensembles: Dict[ModelType, SFAEnsemble] = {}
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[ModelType, EnsembleResult]:
        """Fit all model ensembles."""
        results = {}
        
        for model_type in self.model_types:
            print(f"Training {model_type.value} ensemble...")
            
            try:
                ensemble = SFAEnsemble(
                    base_model_type=model_type,
                    task=self.task,
                    n_folds=self.n_folds,
                    use_optuna=self.use_optuna,
                    random_state=self.random_state
                )
                result = ensemble.fit(X, y)
                self.ensembles[model_type] = ensemble
                results[model_type] = result
            except Exception as e:
                warnings.warn(f"Failed to train {model_type.value}: {e}")
        
        self._is_fitted = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions by averaging across all model types."""
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        predictions = []
        for model_type, ensemble in self.ensembles.items():
            try:
                pred = ensemble.predict(X)
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Prediction failed for {model_type.value}: {e}")
        
        if not predictions:
            raise ValueError("All predictions failed.")
        
        return np.mean(predictions, axis=0)


if __name__ == "__main__":
    # Example usage
    print("SFA Ensemble Training Module")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(1000) * 0.5 > 0).astype(int)
    
    # Split data
    train_size = 800
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train SFA ensemble
    print("\nTraining SFA ensemble with XGBoost...")
    ensemble = SFAEnsemble(
        base_model_type=ModelType.XGBOOST,
        task="classification",
        n_folds=5
    )
    
    result = ensemble.fit(X_train, y_train)
    
    # Evaluate
    predictions = ensemble.predict(X_test, return_all=True)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    print("\nTest Results:")
    for name, pred in predictions.items():
        auc = roc_auc_score(y_test, pred)
        acc = accuracy_score(y_test, (pred > 0.5).astype(int))
        print(f"  {name}: AUC={auc:.4f}, Accuracy={acc:.4f}")
