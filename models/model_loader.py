"""
Model Loader - Load trained models from disk
"""
import torch
import pickle
from pathlib import Path
from config import GAHT_MODEL_PATH, RF_MODEL_PATH, MLP_MODEL_PATH, DEVICE

try:
    from .gaht_model import GAHT
    GAHT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: GAHT model not available: {e}")
    GAHT_AVAILABLE = False


class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.gaht_model = None
        self.rf_model = None
        self.mlp_model = None
        self.mlp_scaler = None
        
        self.load_models()
    
    def load_models(self):
        """Load all three models"""
        # Load GAHT
        if GAHT_AVAILABLE:
            try:
                self.gaht_model = GAHT().to(self.device)
                state_dict = torch.load(GAHT_MODEL_PATH, map_location=self.device)
                self.gaht_model.load_state_dict(state_dict)
                self.gaht_model.eval()
                print("✅ GAHT model loaded")
            except Exception as e:
                print(f"❌ Failed to load GAHT: {e}")
        else:
            print("⚠️  GAHT model skipped (torch-geometric not available)")
        
        # Load Random Forest
        try:
            with open(RF_MODEL_PATH, 'rb') as f:
                self.rf_model = pickle.load(f)
            print("✅ Random Forest model loaded")
        except Exception as e:
            print(f"❌ Failed to load RF: {e}")
        
        # Load MLP
        try:
            with open(MLP_MODEL_PATH, 'rb') as f:
                mlp_obj = pickle.load(f)
            if isinstance(mlp_obj, tuple) and len(mlp_obj) >= 2:
                # Expecting (model, scaler[, metadata])
                self.mlp_model = mlp_obj[0]
                self.mlp_scaler = mlp_obj[1]
            else:
                self.mlp_model = mlp_obj
                self.mlp_scaler = None
            print("✅ MLP model loaded")
        except Exception as e:
            print(f"❌ Failed to load MLP: {e}")
    
    def get_gaht(self):
        return self.gaht_model
    
    def get_rf(self):
        return self.rf_model
    
    def get_mlp(self):
        return self.mlp_model

    def get_mlp_scaler(self):
        return self.mlp_scaler
