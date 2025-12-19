"""
Feature extraction for ML models
"""
import numpy as np
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    Data = None

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski


def extract_ecfp(mol, radius=2, n_bits=2048):
    """Extract ECFP (Morgan) fingerprint"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


def extract_descriptors(mol):
    """Extract molecular descriptors for MLP"""
    try:
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            Descriptors.NumRadicalElectrons(mol)
        ])
        return desc
    except:
        return None


def generate_3d_conformer(mol):
    """Generate 3D conformer for molecule"""
    try:
        mol_copy = Chem.Mol(mol)
        mol_copy = Chem.AddHs(mol_copy)
        
        # Generate 3D coordinates
        if AllChem.EmbedMolecule(mol_copy, randomSeed=42) == -1:
            return None
        
        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol_copy)
        
        return mol_copy
    except:
        return None


def mol_to_graph(mol):
    """
    Convert RDKit molecule to PyTorch Geometric Data object
    """
    if not HAS_TORCH:
        return None
    
    try:
        # Generate 3D conformer
        mol_3d = generate_3d_conformer(mol)
        if mol_3d is None:
            return None
        
        # Extract atom features
        atom_features = []
        for atom in mol_3d.GetAtoms():
            feat = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.IsInRing()),
                atom.GetMass()
            ]
            atom_features.append(feat)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract bonds as edges
        edge_index = []
        for bond in mol_3d.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        
        if len(edge_index) == 0:
            return None
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Extract 3D positions
        conf = mol_3d.GetConformer()
        pos = torch.tensor([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] 
                           for i in range(mol_3d.GetNumAtoms())], dtype=torch.float)
        
        # Create batch attribute for single molecule
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, pos=pos, batch=batch)
    
    except Exception as e:
        print(f"Graph conversion error: {e}")
        return None
