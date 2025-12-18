"""
Model Loader - Demo mode for free deployment
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class DemoRandomForest:
    """Demo RF that gives random but plausible predictions"""
    def predict_proba(self, X):
        np.random.seed(hash(str(X[0])) % 2**32)
        n_samples = len(X)
        proba = np.random.beta(2, 5, size=(n_samples, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class DemoMLP:
    """Demo MLP that gives random but plausible predictions"""
    def predict_proba(self, X):
        np.random.seed(hash(str(X[0])) % 2**32 + 100)
        n_samples = len(X)
        proba = np.random.beta(3, 4, size=(n_samples, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class ModelLoader:
    """Load and manage models - Demo mode for free tier"""
    
    def __init__(self):
        self.gaht_model = None
        self.rf_model = None
        self.mlp_model = None
        self.mlp_scaler = None
        
        self.load_models()
    
    def load_models(self):
        """Load demo models (no files needed)"""
        print("⚠️  Running in DEMO MODE (free tier)")
        print("⚠️  Predictions are simulated - deploy with real models for production")
        
        # GAHT disabled for free tier
        self.gaht_model = None
        print("❌ GAHT disabled (requires torch - too heavy for free tier)")
        
        # Create demo Random Forest
        self.rf_model = DemoRandomForest()
        print("✅ Random Forest model loaded (demo mode)")
        
        # Create demo MLP
        self.mlp_model = DemoMLP()
        self.mlp_scaler = StandardScaler()  # Dummy scaler
        print("✅ MLP model loaded (demo mode)")
    
    def get_gaht(self):
        return self.gaht_model
    
    def get_rf(self):
        return self.rf_model
    
    def get_mlp(self):
        return self.mlp_model

    def get_mlp_scaler(self):
        return self.mlp_scaler
