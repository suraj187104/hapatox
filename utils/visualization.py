"""
Visualization utilities
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from io import BytesIO
import base64


def plot_molecule(mol, size=(400, 400)):
    """Generate molecule image"""
    try:
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None


def create_attention_heatmap(mol, atom_importances):
    """
    Create molecule visualization with atom importance heatmap
    
    Args:
        mol: RDKit molecule
        atom_importances: List of importance scores (0-1) for each atom
    
    Returns:
        Base64 encoded image
    """
    try:
        # Normalize importances to 0-1
        importances = np.array(atom_importances)
        if importances.max() > 0:
            importances = importances / importances.max()
        
        # Create color map (white = low, red = high)
        from matplotlib import cm
        colors = cm.Reds(importances)
        
        # Convert to RDKit colors
        atom_colors = {}
        for idx, color in enumerate(colors):
            atom_colors[idx] = tuple(color[:3])
        
        # Draw molecule with colors
        drawer = Draw.MolDraw2DCairo(400, 400)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        # Get image
        img_data = drawer.GetDrawingText()
        img_str = base64.b64encode(img_data).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None


def create_bar_chart(labels, values, title="", ylabel=""):
    """Create bar chart and return as base64 image"""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        
        # Convert to base64
        buffered = BytesIO()
        plt.savefig(buffered, format='png', dpi=100)
        plt.close()
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except:
        return None
