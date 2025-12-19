"""
Model Loader - Load REAL trained models (local only, not for GitHub)
"""
import pickle
import numpy as np
from pathlib import Path
from config import RF_MODEL_PATH, MLP_MODEL_PATH


class ModelLoader:
    """Load and manage real trained models"""
    
    def __init__(self):
        self.gaht_model = None
        self.rf_model = None
        self.mlp_model = None
        self.mlp_scaler = None
        
        self.load_models()
    
    def load_models(self):
        """Load real trained models from disk"""
        # GAHT disabled (requires torch)
        self.gaht_model = None
        print(">> GAHT disabled (requires torch)")
        
        # Load REAL Random Forest
        try:
            with open(RF_MODEL_PATH, 'rb') as f:
                self.rf_model = pickle.load(f)
            print(">> Random Forest model loaded (REAL)")
        except Exception as e:
            print(f">> Failed to load RF: {e}")
        
        # Load REAL MLP
        try:
            with open(MLP_MODEL_PATH, 'rb') as f:
                mlp_obj = pickle.load(f)
            if isinstance(mlp_obj, tuple) and len(mlp_obj) >= 2:
                self.mlp_model = mlp_obj[0]
                self.mlp_scaler = mlp_obj[1]
            else:
                self.mlp_model = mlp_obj
                self.mlp_scaler = None
            print(">> MLP model loaded (REAL)")
        except Exception as e:
            print(f">> Failed to load MLP: {e}")
    
    def get_gaht(self):
        return self.gaht_model
    
    def get_rf(self):
        return self.rf_model
    
    def get_mlp(self):
        return self.mlp_model

    def get_mlp_scaler(self):
        return self.mlp_scaler
