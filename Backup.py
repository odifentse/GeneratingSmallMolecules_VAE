#STEP 1: SETUP

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")

# STEP 2: ACCESS THE DATASET
# Dataset from Zinc (250k smiles) from Kaggle source code

df = pd.read_csv("/Users/odilehasa/PyCharmProjects_Aug2022/GeneratingSmallMoleculeswithVAE/250k_smiles.csv")
df['smiles']= df['smiles'].apply(lambda s: s.replace('\n',''))
df.head()

# STEP 3: DEFINE THE MOLECULES FROM THE SMILES

#Sanitize the data before storing it or using it for any purpose, to ensure it contains only valid data.
def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIT=True, force=True)
    return molecule

# print the Smiles, logP and QED data from the row 101
print(f"SMILE:\t{df.smiles[100]}\nlogP:\t{df.logP[100]}\nqed:\t{df.qed[100]}")
molecule = molecule_from_smiles(df.iloc[100].smiles)
print("Molecule:")
molecule
#m = Chem.MolFromSmiles(molecule)
#Draw.MolToImage(molecule)