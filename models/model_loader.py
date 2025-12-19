"""
Model Loader - Load REAL trained models including GAHT (local only, not for GitHub)
"""
import pickle
import numpy as np
from pathlib import Path
from config import RF_MODEL_PATH, MLP_MODEL_PATH, GAHT_MODEL_PATH

# Try to import torch for GAHT
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class ModelLoader:
    """Load and manage real trained models including GAHT"""
    
    def __init__(self):
        self.gaht_model = None
        self.rf_model = None
        self.mlp_model = None
        self.mlp_scaler = None
        
        self.load_models()
    
    def load_models(self):
        """Load models - real models locally, demo models on Render"""
        # Load GAHT model (requires torch and model file)
        if HAS_TORCH and GAHT_MODEL_PATH and GAHT_MODEL_PATH.exists():
            try:
                # Import GAHT only when needed
                from models.gaht_model import GAHT
                
                self.gaht_model = GAHT(
                    node_features=9,
                    hidden_dim=128,
                    num_layers=3,
                    num_heads=4
                )
                state_dict = torch.load(GAHT_MODEL_PATH, map_location='cpu')
                self.gaht_model.load_state_dict(state_dict)
                self.gaht_model.eval()
                print("✅ GAHT model loaded (REAL - with explainability)")
            except Exception as e:
                print(f">> Failed to load GAHT: {e}")
                self.gaht_model = None
        else:
            self.gaht_model = None
            if GAHT_MODEL_PATH is None:
                print(">> GAHT disabled (demo mode)")
            elif not HAS_TORCH:
                print(">> GAHT disabled (torch not installed)")
            else:
                print(">> GAHT disabled (model file not found)")
        
        # Load Random Forest (real or demo)
        if RF_MODEL_PATH and RF_MODEL_PATH.exists():
            try:
                with open(RF_MODEL_PATH, 'rb') as f:
                    self.rf_model = pickle.load(f)
                print("✅ Random Forest model loaded (REAL)")
            except Exception as e:
                print(f">> Failed to load RF: {e}")
                self.rf_model = self._create_demo_rf()
        else:
            self.rf_model = self._create_demo_rf()
            print("✅ Random Forest model loaded (DEMO)")
        
        # Load MLP (real or demo)
        if MLP_MODEL_PATH and MLP_MODEL_PATH.exists():
            try:
                with open(MLP_MODEL_PATH, 'rb') as f:
                    mlp_obj = pickle.load(f)
                if isinstance(mlp_obj, tuple) and len(mlp_obj) >= 2:
                    self.mlp_model = mlp_obj[0]
                    self.mlp_scaler = mlp_obj[1]
                else:
                    self.mlp_model = mlp_obj
                    self.mlp_scaler = None
                print("✅ MLP model loaded (REAL)")
            except Exception as e:
                print(f">> Failed to load MLP: {e}")
                self.mlp_model = self._create_demo_mlp()
        else:
            self.mlp_model = self._create_demo_mlp()
            print("✅ MLP model loaded (DEMO)")
    
    def _create_demo_rf(self):
        """Create demo Random Forest that gives plausible predictions"""
        class DemoRandomForest:
            def predict_proba(self, X):
                np.random.seed(hash(str(X[0])) % 2**32)
                n_samples = len(X)
                proba = np.random.beta(2, 5, size=(n_samples, 2))
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba
        return DemoRandomForest()
    
    def _create_demo_mlp(self):
        """Create demo MLP that gives plausible predictions"""
        class DemoMLP:
            def predict_proba(self, X):
                np.random.seed(hash(str(X[0])) % 2**32 + 100)
                n_samples = len(X)
                proba = np.random.beta(3, 4, size=(n_samples, 2))
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba
        return DemoMLP()
    
    def get_gaht(self):
        return self.gaht_model
    
    def get_rf(self):
        return self.rf_model
    
    def get_mlp(self):
        return self.mlp_model

    def get_mlp_scaler(self):
        return self.mlp_scaler
