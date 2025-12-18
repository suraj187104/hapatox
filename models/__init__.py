"""
Models package for loading and using trained ML models
"""
try:
	from .gaht_model import GAHT  # noqa: F401
except Exception as e:  # pragma: no cover - best-effort import
	GAHT = None
	print(f"⚠️  GAHT model not imported: {e}")

from .model_loader import ModelLoader
from .predictor import ToxicityPredictor

__all__ = ['GAHT', 'ModelLoader', 'ToxicityPredictor']
