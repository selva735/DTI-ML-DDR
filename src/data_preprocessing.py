import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("Please install RDKit to use molecular fingerprint features (conda install -c conda-forge rdkit)")

def load_data(file_path):
    """
    Load DTI data from a CSV file.
    Expects columns: 'drug', 'target', 'interaction', optionally 'smiles'
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Clean the input dataframe: drop missing values, remove duplicates.
    """
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to Morgan fingerprint (ECFP) with RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
