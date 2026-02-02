"""Baseline Models"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib

class BaselineModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_all(self, X_train, y_train):
        """Train all baseline models"""
        print("="*50)
        print("TRAINING BASELINE MODELS")
        print("="*50)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Naive Bayes': GaussianNB()
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            print(f"✓ {name} trained")
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
            }
            
            print(f"\n{name}:")
            for metric, value in self.results[name].items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        
        return self.results
    
    def save_models(self, filepath='../results/models/baseline_models.pkl'):
        """Save all models"""
        joblib.dump(self.models, filepath)
        print(f"\n✓ Models saved to {filepath}")
