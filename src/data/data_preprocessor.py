"""Data Preprocessing Module"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.combine import SMOTETomek
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
    
    def handle_missing_values(self, df, strategy='iterative'):
        """Handle missing values"""
        print(f"\nHandling missing values - Strategy: {strategy}")
        df_imputed = df.copy()
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if len(missing_cols) == 0:
            print("✓ No missing values")
            return df_imputed
        
        if strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        df_imputed[missing_cols] = imputer.fit_transform(df[missing_cols])
        self.imputers['main'] = imputer
        print(f"✓ Missing values handled")
        return df_imputed
    
    def scale_features(self, df, numerical_columns):
        """Scale numerical features"""
        print("\nScaling features...")
        df_scaled = df.copy()
        scaler = StandardScaler()
        df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        self.scalers['numerical'] = scaler
        print("✓ Features scaled")
        return df_scaled
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance with SMOTE-Tomek"""
        print("\nHandling class imbalance...")
        print(f"Before: {dict(pd.Series(y).value_counts())}")
        sampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"After: {dict(pd.Series(y_resampled).value_counts())}")
        print("✓ Resampling complete")
        return X_resampled, y_resampled
    
    def save_preprocessors(self, filepath='results/models/preprocessors.pkl'):
        """Save preprocessors"""
        preprocessors = {
            'scalers': self.scalers,
            'imputers': self.imputers
        }
        joblib.dump(preprocessors, filepath)
        print(f"\n✓ Preprocessors saved to {filepath}")
