import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from maxent_model import MaxEntModel


@dataclass
class ModelResult:
    model_name: str
    model: object
    cv_auc_scores: List[float]
    mean_auc: float
    std_auc: float
    weight: float
    predictions: np.ndarray


class SpeciesDistributionEnsemble:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.models: Dict[str, object] = {}
        self.results: Dict[str, ModelResult] = {}
        self.weights: Dict[str, float] = {}
        self.scaler = StandardScaler()
        
    def _initialize_models(self) -> Dict[str, object]:
        return {
            'MaxEnt': MaxEntModel(l1_reg=0.1, l2_reg=1.0, feature_type='quadratic'),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    probability=True,
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=self.random_state
                ))
            ]),
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    penalty='elasticnet',
                    l1_ratio=0.5,
                    solver='saga',
                    max_iter=1000,
                    random_state=self.random_state
                ))
            ]),
            'GaussianProcess': Pipeline([
                ('scaler', StandardScaler()),
                ('gp', GaussianProcessClassifier(
                    kernel=ConstantKernel(1.0) * RBF(1.0),
                    n_restarts_optimizer=5,
                    random_state=self.random_state
                ))
            ])
        }
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    
    def cross_validate_models(self, presence_data: np.ndarray, 
                               background_data: np.ndarray) -> Dict[str, ModelResult]:
        X_presence = presence_data
        X_background = background_data
        
        X = np.vstack([X_presence, X_background])
        y = np.hstack([np.ones(len(X_presence)), np.zeros(len(X_background))])
        
        self.models = self._initialize_models()
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"  Cross-validating {model_name}...", end=' ', flush=True)
            
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            auc_scores = []
            fold_predictions = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                X_train_presence = X_train[y_train == 1]
                X_train_background = X_train[y_train == 0]
                
                if model_name == 'MaxEnt':
                    model.fit(X_train_presence, X_train_background)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                
                auc = self._calculate_auc(y_test, y_pred)
                auc_scores.append(auc)
            
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            
            print(f"Mean AUC = {mean_auc:.4f} (±{std_auc:.4f})")
            
            cv_results[model_name] = {
                'auc_scores': auc_scores,
                'mean_auc': mean_auc,
                'std_auc': std_auc
            }
        
        return cv_results
    
    def _calculate_weights(self, cv_results: Dict[str, dict]) -> Dict[str, float]:
        model_names = list(cv_results.keys())
        auc_values = np.array([cv_results[name]['mean_auc'] for name in model_names])
        
        auc_adjusted = np.maximum(auc_values - 0.5, 1e-6)
        
        weights = auc_adjusted / np.sum(auc_adjusted)
        
        return {name: weight for name, weight in zip(model_names, weights)}
    
    def fit(self, presence_data: np.ndarray, background_data: np.ndarray) -> 'SpeciesDistributionEnsemble':
        print("Performing cross-validation for model weight assignment...")
        cv_results = self.cross_validate_models(presence_data, background_data)
        
        self.weights = self._calculate_weights(cv_results)
        
        print("\nFitting final models on full dataset...")
        X = np.vstack([presence_data, background_data])
        y = np.hstack([np.ones(len(presence_data)), np.zeros(len(background_data))])
        
        for model_name, model in self.models.items():
            print(f"  Fitting {model_name} on full data...")
            
            if model_name == 'MaxEnt':
                model.fit(presence_data, background_data)
            else:
                model.fit(X, y)
        
        self.results = {}
        for model_name in self.models.keys():
            result = cv_results[model_name]
            self.results[model_name] = ModelResult(
                model_name=model_name,
                model=self.models[model_name],
                cv_auc_scores=result['auc_scores'],
                mean_auc=result['mean_auc'],
                std_auc=result['std_auc'],
                weight=self.weights[model_name],
                predictions=None
            )
        
        return self
    
    def predict(self, env_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'MaxEnt':
                pred = model.predict(env_data)
            else:
                pred = model.predict_proba(env_data)[:, 1]
            
            pred = pred / np.max(pred) if np.max(pred) > 0 else pred
            individual_predictions[model_name] = pred
        
        ensemble_prediction = np.zeros_like(next(iter(individual_predictions.values())))
        for model_name, pred in individual_predictions.items():
            ensemble_prediction += self.weights[model_name] * pred
        
        return ensemble_prediction, individual_predictions
    
    def get_weights_summary(self) -> List[Dict]:
        summary = []
        for model_name, result in self.results.items():
            summary.append({
                'model': model_name,
                'mean_auc': result.mean_auc,
                'std_auc': result.std_auc,
                'weight': result.weight
            })
        
        summary.sort(key=lambda x: x['weight'], reverse=True)
        return summary
    
    def print_weights(self):
        print("\n" + "=" * 60)
        print("Ensemble Model Weights (Based on Cross-Validation AUC)")
        print("=" * 60)
        print(f"{'Model':<25} {'Mean AUC':<12} {'Std AUC':<12} {'Weight':<10}")
        print("-" * 60)
        
        for item in self.get_weights_summary():
            print(f"{item['model']:<25} {item['mean_auc']:<12.4f} "
                  f"{item['std_auc']:<12.4f} {item['weight']:<10.4f}")
        
        print("=" * 60)


class SimpleGAM:
    def __init__(self, n_splines: int = 5, lam: float = 1.0):
        self.n_splines = n_splines
        self.lam = lam
        self.coef_ = None
        self.scaler = StandardScaler()
        
    def _spline_basis(self, x: np.ndarray, n_knots: int = 5) -> np.ndarray:
        knots = np.linspace(0, 1, n_knots + 2)[1:-1]
        basis = np.column_stack([x ** i for i in range(3)])
        
        for knot in knots:
            basis = np.column_stack([
                basis,
                np.maximum(x - knot, 0) ** 2
            ])
        
        return basis
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        n_features = X_scaled.shape[1]
        basis_list = []
        
        for i in range(n_features):
            basis = self._spline_basis(X_scaled[:, i], self.n_splines)
            basis_list.append(basis)
        
        X_design = np.hstack(basis_list)
        X_design = np.column_stack([np.ones(len(X_design)), X_design])
        
        n_coef = X_design.shape[1]
        penalty = np.zeros((n_coef, n_coef))
        penalty[1:, 1:] = np.eye(n_coef - 1) * self.lam
        
        self.coef_ = np.linalg.inv(X_design.T @ X_design + penalty) @ X_design.T @ y
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        n_features = X_scaled.shape[1]
        basis_list = []
        
        for i in range(n_features):
            basis = self._spline_basis(X_scaled[:, i], self.n_splines)
            basis_list.append(basis)
        
        X_design = np.hstack(basis_list)
        X_design = np.column_stack([np.ones(len(X_design)), X_design])
        
        logit = X_design @ self.coef_
        prob = 1 / (1 + np.exp(-logit))
        
        return np.column_stack([1 - prob, prob])
