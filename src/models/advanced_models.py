"""Advanced Models with Hyperparameter Optimization"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import numpy as np
import joblib

class AdvancedModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_params = {}
    
    def train_xgboost(self, X_train, y_train, optimize=False):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        if optimize:
            # Hyperparameter optimization with Optuna
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                return model.score(X_train, y_train)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=True)
            self.best_params['XGBoost'] = study.best_params
            self.models['XGBoost'] = xgb.XGBClassifier(**study.best_params, random_state=self.random_state)
        else:
            self.models['XGBoost'] = xgb.XGBClassifier(random_state=self.random_state)
        
        self.models['XGBoost'].fit(X_train, y_train)
        print("✓ XGBoost trained")
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        self.models['LightGBM'] = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        self.models['LightGBM'].fit(X_train, y_train)
        print("✓ LightGBM trained")
    
    def train_neural_network(self, X_train, y_train):
        """Train Neural Network"""
        print("\nTraining Neural Network...")
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True
        )
        self.models['Neural Network'].fit(X_train, y_train)
        print("✓ Neural Network trained")
    
    def train_all(self, X_train, y_train, optimize_xgb=False):
        """Train all advanced models"""
        print("="*50)
        print("TRAINING ADVANCED MODELS")
        print("="*50)
        
        self.train_xgboost(X_train, y_train, optimize=optimize_xgb)
        self.train_lightgbm(X_train, y_train)
        self.train_neural_network(X_train, y_train)
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*50)
        print("EVALUATING ADVANCED MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            print(f"\n{name}:")
            for metric, value in self.results[name].items():
                print(f"  {metric}: {value:.4f}")
        
        return self.results
    
    def save_models(self, filepath='../results/models/advanced_models.pkl'):
        """Save all models"""
        joblib.dump(self.models, filepath)
        print(f"\n✓ Advanced models saved to {filepath}")
