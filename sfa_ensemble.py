"""
SFA Ensemble Training Module
K-fold cross-validation with multiple base learners
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ADABOOST = "adaboost"


@dataclass
class FoldResult:
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    predictions: np.ndarray
    shapley_values: np.ndarray
    model: Any


@dataclass
class EnsembleResult:
    fold_results: List[FoldResult]
    oof_predictions: np.ndarray
    oof_shapley_values: np.ndarray
    models: List[Any]


class BaseModel:
    """Base class for ensemble models"""
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class XGBoostModel(BaseModel):
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        import xgboost as xgb
        if self.task == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self._explainer = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return shap_values[1]
        return shap_values


class LightGBMModel(BaseModel):
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
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
        return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return shap_values[1]
        return shap_values


class CatBoostModel(BaseModel):
    
    def __init__(self, task: str = "classification", **params):
        self.task = task
        self.params = params
        self.model = None
        self._explainer = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        from catboost import CatBoostClassifier, CatBoostRegressor
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
        return self.model.predict(X).reshape(-1, 1)
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return shap_values[1]
        return shap_values


def create_model(model_type: ModelType, task: str = "classification", **params) -> BaseModel:
    """Factory function to create models"""
    if model_type == ModelType.XGBOOST:
        return XGBoostModel(task=task, **params)
    elif model_type == ModelType.LIGHTGBM:
        return LightGBMModel(task=task, **params)
    elif model_type == ModelType.CATBOOST:
        return CatBoostModel(task=task, **params)
    raise ValueError(f"Unknown model type: {model_type}")


class KFoldCV:
    """K-Fold Cross-Validation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kfold.split(X))


class SFAEnsemble:
    """SFA Ensemble with two-stage learning"""
    
    def __init__(
        self,
        base_model_type: ModelType = ModelType.XGBOOST,
        task: str = "classification",
        n_folds: int = 5,
        use_optuna: bool = False,
        n_optuna_trials: int = 15,
        random_state: int = 42
    ):
        self.base_model_type = base_model_type
        self.task = task
        self.n_folds = n_folds
        self.use_optuna = use_optuna
        self.n_optuna_trials = n_optuna_trials
        self.random_state = random_state
        
        self.stage1_models: List[BaseModel] = []
        self.stage1_params: Dict = {}
        
        self.stage2_p_model = None
        self.stage2_shap_model = None
        self.stage2_pshap_model = None
        
        self.oof_predictions = None
        self.oof_shapley_values = None
        self._is_fitted = False
    
    def _get_default_params(self, model_type: ModelType) -> Dict:
        if model_type == ModelType.XGBOOST:
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 
                    'random_state': self.random_state}
        elif model_type == ModelType.LIGHTGBM:
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 
                    'random_state': self.random_state, 'verbosity': -1}
        elif model_type == ModelType.CATBOOST:
            return {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 
                    'random_seed': self.random_state, 'verbose': False}
        return {}
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: ModelType) -> Dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            if model_type == ModelType.XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'random_state': self.random_state
                }
            elif model_type == ModelType.LIGHTGBM:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'random_state': self.random_state, 'verbosity': -1
                }
            else:
                params = self._get_default_params(model_type)
            
            model = create_model(model_type, self.task, **params)
            model.fit(X, y)
            
            if self.task == "classification":
                preds = model.predict_proba(X)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    preds = preds[:, 1]
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y, preds)
            else:
                from sklearn.metrics import r2_score
                return r2_score(y, model.predict(X))
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        return best_params
    
    def fit_stage1(self, X: np.ndarray, y: np.ndarray, model_params: Dict = None) -> EnsembleResult:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        if model_params is None:
            if self.use_optuna:
                model_params = self._tune_hyperparameters(X, y, self.base_model_type)
            else:
                model_params = self._get_default_params(self.base_model_type)
        
        self.stage1_params = model_params
        
        if self.task == "classification":
            n_classes = len(np.unique(y))
            self.oof_predictions = np.zeros(n_samples) if n_classes == 2 else np.zeros((n_samples, n_classes))
        else:
            self.oof_predictions = np.zeros(n_samples)
        
        self.oof_shapley_values = np.zeros((n_samples, n_features))
        
        kfold = KFoldCV(n_splits=self.n_folds, random_state=self.random_state)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            model = create_model(self.base_model_type, self.task, **model_params)
            model.fit(X_train, y_train)
            
            if self.task == "classification":
                proba = model.predict_proba(X_val)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    self.oof_predictions[val_idx] = proba[:, 1]
                else:
                    self.oof_predictions[val_idx] = proba
            else:
                self.oof_predictions[val_idx] = model.predict(X_val)
            
            shap_vals = model.get_shap_values(X_val)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
            self.oof_shapley_values[val_idx] = shap_vals
            
            self.stage1_models.append(model)
            
            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                predictions=self.oof_predictions[val_idx].copy(),
                shapley_values=self.oof_shapley_values[val_idx].copy(),
                model=model
            ))
        
        return EnsembleResult(fold_results, self.oof_predictions, self.oof_shapley_values, self.stage1_models)
    
    def fit_stage2(self, X: np.ndarray, y: np.ndarray, model_params: Dict = None):
        if self.oof_predictions is None:
            raise ValueError("Stage 1 must be fitted first")
        
        if model_params is None:
            model_params = self.stage1_params
        
        oof_pred = self.oof_predictions.reshape(-1, 1) if self.oof_predictions.ndim == 1 else self.oof_predictions
        
        # P augmented
        X_p = np.hstack([X, oof_pred])
        self.stage2_p_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_p_model.fit(X_p, y)
        
        # SHAP augmented
        X_shap = np.hstack([X, self.oof_shapley_values])
        self.stage2_shap_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_shap_model.fit(X_shap, y)
        
        # P+SHAP augmented
        X_pshap = np.hstack([X, oof_pred, self.oof_shapley_values])
        self.stage2_pshap_model = create_model(self.base_model_type, self.task, **model_params)
        self.stage2_pshap_model.fit(X_pshap, y)
        
        self._is_fitted = True
    
    def fit(self, X: np.ndarray, y: np.ndarray, model_params: Dict = None) -> EnsembleResult:
        result = self.fit_stage1(X, y, model_params)
        self.fit_stage2(X, y, model_params)
        return result
    
    def predict(self, X: np.ndarray, return_all: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted")
        
        stage1_preds, stage1_shap = [], []
        
        for model in self.stage1_models:
            if self.task == "classification":
                proba = model.predict_proba(X)
                stage1_preds.append(proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba)
            else:
                stage1_preds.append(model.predict(X))
            
            shap_vals = model.get_shap_values(X)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals.mean(axis=2)
            stage1_shap.append(shap_vals)
        
        avg_pred = np.mean(stage1_preds, axis=0)
        avg_shap = np.mean(stage1_shap, axis=0)
        
        pred_features = avg_pred.reshape(-1, 1) if avg_pred.ndim == 1 else avg_pred
        
        X_p = np.hstack([X, pred_features])
        X_shap = np.hstack([X, avg_shap])
        X_pshap = np.hstack([X, pred_features, avg_shap])
        
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
        
        final_pred = (avg_pred + p_pred + shap_pred + pshap_pred) / 4
        
        if return_all:
            return {
                'base': avg_pred, 'p_augmented': p_pred,
                'shap_augmented': shap_pred, 'pshap_augmented': pshap_pred,
                'sfa': final_pred
            }
        
        return final_pred


class MultiModelSFA:
    """Multi-model SFA using multiple base learner types"""
    
    def __init__(
        self,
        model_types: List[ModelType] = None,
        task: str = "classification",
        n_folds: int = 5,
        random_state: int = 42
    ):
        if model_types is None:
            model_types = [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST]
        
        self.model_types = model_types
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.ensembles: Dict[ModelType, SFAEnsemble] = {}
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[ModelType, EnsembleResult]:
        results = {}
        
        for model_type in self.model_types:
            print(f"Training {model_type.value} ensemble...")
            
            ensemble = SFAEnsemble(
                base_model_type=model_type,
                task=self.task,
                n_folds=self.n_folds,
                random_state=self.random_state
            )
            result = ensemble.fit(X, y)
            self.ensembles[model_type] = ensemble
            results[model_type] = result
        
        self._is_fitted = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Not fitted")
        
        predictions = [ensemble.predict(X) for ensemble in self.ensembles.values()]
        return np.mean(predictions, axis=0)