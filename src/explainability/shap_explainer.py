"""SHAP Explainability Module"""
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SHAPExplainer:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """Create SHAP explainer"""
        print("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("✓ SHAP explainer created")
    
    def calculate_shap_values(self, X_test):
        """Calculate SHAP values"""
        print("\nCalculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X_test)
        
        # For binary classification, take positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print(f"✓ SHAP values calculated: {self.shap_values.shape}")
        return self.shap_values
    
    def plot_summary(self, X_test, max_display=10):
        """Plot SHAP summary"""
        print("\nGenerating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_test, 
                         feature_names=self.feature_names,
                         max_display=max_display,
                         show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        print("✓ Summary plot generated")
    
    def plot_bar(self, X_test, max_display=10):
        """Plot SHAP bar chart"""
        print("\nGenerating SHAP bar plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X_test,
                         feature_names=self.feature_names,
                         plot_type='bar',
                         max_display=max_display,
                         show=False)
        plt.title('SHAP Feature Importance (Bar)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        print("✓ Bar plot generated")
    
    def plot_waterfall(self, X_test, sample_idx=0):
        """Plot SHAP waterfall for single prediction"""
        print(f"\nGenerating waterfall plot for sample {sample_idx}...")
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X_test[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        print("✓ Waterfall plot generated")
    
    def get_feature_importance(self):
        """Get global feature importance"""
        importance = np.abs(self.shap_values).mean(axis=0)
        if self.feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return importance_df
