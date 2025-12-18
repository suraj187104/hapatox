"""
Molecule utilities - SMILES validation and RDKit operations
"""
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64


def validate_smiles(smiles):
    """
    Validate SMILES string
    
    Returns:
        (is_valid, error_message)
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES string is required"
    
    if len(smiles) > 500:
        return False, "SMILES string too long (max 500 characters)"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string - could not parse"
    
    return True, None


def smiles_to_mol(smiles):
    """Convert SMILES to RDKit molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def mol_to_image(mol, size=(300, 300), highlight_atoms=None):
    """
    Convert RDKit molecule to image
    
    Returns:
        Base64 encoded image string
    """
    try:
        img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except:
        return None


def mol_to_svg(mol, size=(300, 300)):
    """Convert RDKit molecule to SVG"""
    try:
        drawer = Draw.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except:
        return None
