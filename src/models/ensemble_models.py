"""Ensemble Models"""
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib

class EnsembleModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_voting_ensemble(self, X_train, y_train):
        """Train Voting Classifier"""
        print("\nTraining Voting Ensemble...")
        
        estimators = [
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=self.random_state, n_estimators=100)),
            ('lgbm', lgb.LGBMClassifier(random_state=self.random_state, verbose=-1))
        ]
        
        self.models['Voting (Soft)'] = VotingClassifier(estimators=estimators, voting='soft')
        self.models['Voting (Soft)'].fit(X_train, y_train)
        print("✓ Voting Ensemble trained")
    
    def train_stacking_ensemble(self, X_train, y_train):
        """Train Stacking Classifier"""
        print("\nTraining Stacking Ensemble...")
        
        estimators = [
            ('rf', RandomForestClassifier(random_state=self.random_state, n_estimators=100)),
            ('lgbm', lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)),
            ('svm', SVC(random_state=self.random_state, probability=True))
        ]
        
        self.models['Stacking'] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5
        )
        self.models['Stacking'].fit(X_train, y_train)
        print("✓ Stacking Ensemble trained")
    
    def train_all(self, X_train, y_train):
        """Train all ensemble models"""
        print("="*50)
        print("TRAINING ENSEMBLE MODELS")
        print("="*50)
        
        self.train_voting_ensemble(X_train, y_train)
        self.train_stacking_ensemble(X_train, y_train)
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all ensemble models"""
        print("\n" + "="*50)
        print("EVALUATING ENSEMBLE MODELS")
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
    
    def save_models(self, filepath='../results/models/ensemble_models.pkl'):
        """Save ensemble models"""
        joblib.dump(self.models, filepath)
        print(f"\n✓ Ensemble models saved to {filepath}")
