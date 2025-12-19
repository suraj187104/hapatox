"""
Toxicity Predictor - Unified prediction interface
"""
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    Data = None

import numpy as np
from rdkit import Chem

from utils.molecule_utils import smiles_to_mol
from utils.feature_extraction import extract_ecfp, extract_descriptors, mol_to_graph


class ToxicityPredictor:
    """Predict toxicity using all three models"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.device = torch.device('cpu') if HAS_TORCH else None
    
    def predict(self, smiles):
        """
        Predict toxicity for a single molecule
        
        Returns:
            dict with predictions from all models
        """
        mol = smiles_to_mol(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        results = {
            'smiles': smiles,
            'gaht': None,
            'rf': None,
            'mlp': None,
            'consensus': None
        }
        
        # GAHT prediction
        try:
            gaht_model = self.model_loader.get_gaht()
            if gaht_model is not None and HAS_TORCH:
                graph_data = mol_to_graph(mol)
                if graph_data is not None:
                    graph_data = graph_data.to(self.device)
                    with torch.no_grad():
                        logit = gaht_model(graph_data)
                        prob = torch.sigmoid(logit).item()
                    results['gaht'] = {
                        'probability': round(prob, 4),
                        'prediction': 'Toxic' if prob > 0.5 else 'Non-toxic',
                        'confidence': round(abs(prob - 0.5) * 2, 4)
                    }
        except Exception as e:
            print(f"GAHT prediction error: {e}")
        
        # Random Forest prediction
        try:
            rf_model = self.model_loader.get_rf()
            if rf_model is not None:
                ecfp = extract_ecfp(mol)
                if ecfp is not None:
                    prob = rf_model.predict_proba(ecfp.reshape(1, -1))[0][1]
                    results['rf'] = {
                        'probability': round(prob, 4),
                        'prediction': 'Toxic' if prob > 0.5 else 'Non-toxic',
                        'confidence': round(abs(prob - 0.5) * 2, 4)
                    }
        except Exception as e:
            print(f"RF prediction error: {e}")
        
        # MLP prediction
        try:
            mlp_model = self.model_loader.get_mlp()
            mlp_scaler = getattr(self.model_loader, 'get_mlp_scaler', lambda: None)()
            if mlp_model is not None:
                descriptors = extract_descriptors(mol)
                if descriptors is not None:
                    features = descriptors.reshape(1, -1)
                    if mlp_scaler is not None:
                        try:
                            features = mlp_scaler.transform(features)
                        except Exception as scale_err:
                            print(f"MLP scaler transform error: {scale_err}")
                    prob = mlp_model.predict_proba(features)[0][1]
                    results['mlp'] = {
                        'probability': round(prob, 4),
                        'prediction': 'Toxic' if prob > 0.5 else 'Non-toxic',
                        'confidence': round(abs(prob - 0.5) * 2, 4)
                    }
        except Exception as e:
            print(f"MLP prediction error: {e}")
        
        # Consensus prediction (average of available predictions)
        probs = []
        if results['gaht']: probs.append(results['gaht']['probability'])
        if results['rf']: probs.append(results['rf']['probability'])
        if results['mlp']: probs.append(results['mlp']['probability'])
        
        if probs:
            consensus_prob = np.mean(probs)
            results['consensus'] = {
                'probability': round(consensus_prob, 4),
                'prediction': 'Toxic' if consensus_prob > 0.5 else 'Non-toxic',
                'confidence': round(abs(consensus_prob - 0.5) * 2, 4),
                'agreement': len([p for p in probs if (p > 0.5) == (consensus_prob > 0.5)])
            }
        
        return results
    
    def explain(self, smiles):
        """
        Generate explanation for GAHT prediction
        
        Returns:
            dict with attention weights and important atoms
        """
        if not HAS_TORCH:
            raise ValueError("Explainability requires torch (not available in demo mode)")
        
        mol = smiles_to_mol(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        gaht_model = self.model_loader.get_gaht()
        if gaht_model is None:
            raise ValueError("GAHT model not loaded")
        
        graph_data = mol_to_graph(mol)
        if graph_data is None:
            raise ValueError("Could not convert molecule to graph")
        
        # Get prediction
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            logit = gaht_model(graph_data)
            prob = torch.sigmoid(logit).item()
        
        # Perturbation-based attribution
        attributions = []
        num_atoms = graph_data.x.size(0)
        
        for atom_idx in range(num_atoms):
            data_masked = graph_data.clone()
            data_masked.x[atom_idx] = 0
            
            with torch.no_grad():
                masked_logit = gaht_model(data_masked)
                masked_prob = torch.sigmoid(masked_logit).item()
            
            attribution = abs(prob - masked_prob)
            attributions.append(attribution)
        
        # Get atom info
        atoms = []
        for idx, atom in enumerate(mol.GetAtoms()):
            atoms.append({
                'index': idx,
                'symbol': atom.GetSymbol(),
                'importance': round(attributions[idx], 4)
            })
        
        # Sort by importance
        atoms_sorted = sorted(atoms, key=lambda x: x['importance'], reverse=True)
        
        return {
            'prediction': round(prob, 4),
            'label': 'Toxic' if prob > 0.5 else 'Non-toxic',
            'atoms': atoms,
            'top_atoms': atoms_sorted[:5]
        }
