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
    arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def sequence_to_counts(seq):
    """
    Convert protein sequence to count vector of 20 standard amino acids.
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq = str(seq).upper()
    return np.array([seq.count(aa) for aa in amino_acids])

def featurize(df):
    """
    Add drug and protein features using fingerprints and sequence counts.
    Returns features X and labels y.
    Expects 'smiles', 'target' (amino acid sequence), and 'interaction' columns.
    """
    drug_feats = np.array([smiles_to_ecfp(smi) for smi in df['smiles']])
    prot_feats = np.array([sequence_to_counts(seq) for seq in df['target']])
    X = np.hstack([drug_feats, prot_feats])
    y = df['interaction'].values
    return X, y

def preprocess(file_path):
    """
    Complete pipeline: load, clean, featurize, and train-test split.
    Returns X_train, X_test, y_train, y_test.
    """
    df = load_data(file_path)
    df = clean_data(df)
    X, y = featurize(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
if __name__ == "__main__":
    # update 'data/dti_sample.csv' to your file path
    X_train, X_test, y_train, y_test = preprocess('data/dti_sample.csv')
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
