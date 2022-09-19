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
# Dataset from Zinc (250k smiles) from Kaggle source code.
# The dataset comes with molecule formula in SMILE representation along with their respective molecular properties such as logP (water–octanal partition coefficient), SAS (synthetic accessibility score) and QED (Qualitative Estimate of Drug-likeness).

df = pd.read_csv("/Users/odilehasa/PyCharmProjects_Aug2022/GeneratingSmallMoleculeswithVAE/250k_smiles.csv")
df['smiles']= df['smiles'].apply(lambda s: s.replace('\n',''))
df.head()

# STEP 3: DEFINE THE MOLECULES FROM THE SMILES

#Sanitize the data before storing it or using it for any purpose, to ensure it contains only valid data.
def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


# print the Smiles, logP and QED data from the row 101
print(f"SMILE:\t{df.smiles[100]}\nlogP:\t{df.logP[100]}\nqed:\t{df.qed[100]}")
molecule = molecule_from_smiles(df.iloc[29].smiles)
print("Molecule:")
molecule

# save the image of the molecule
img = Draw.MolToImage(molecule)
img.save('output.png')

# below are a list of Reactive Nonmetals (as per the periodic table).
SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S",' \
                '"P", "Cl", "Br"]'

bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

MAX_MOLSIZE = max(df['smiles'].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

print("Max molecule size: {}".format(MAX_MOLSIZE))
print("Character set Length: {}".format(len(SMILE_CHARSET)))


#STEP 4: DETERMINE THE HYPERPARAMETERS.

# A hyperparameter is a parameter that is set before the learning process begins. These parameters are tunable and can directly affect how well a model trains.

BATCH_SIZE = 32
EPOCHS = 10            #An epoch is one cycle through the full training dataset

VAE_LR = 5e-4
NUM_ATOMS = 120        #Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET) #Number of atom types, as listed in SILE_CHARSET.
BOND_DIM= 4+1          #Number of bond types.
LATENT_DIM = 435       # Size of latent space.

# Below function smiles_to_graph referenced by author from: https://keras.io/examples/generative/wgan-graphs/
def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles) #convert smiles to molecule object
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.xeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in the molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]

        #loop over one-hop neighbour
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j )
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where there is no bond, add 1 to the last channel to indicate "non-bond"
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where there is no atom, add 1 to the last column to indicate "non-atoms"
    features[np.where(np.sum(features, axis = 1) == 0)[0], -1] = 1

    return adjacency, features

# Below function graph_to_molecule referenced by author from: Reference: https://keras.io/examples/generative/wgan-graphs/
def graph_to_molecule(graph):
    adjacency, features = graph      # unpack the graph
    molecule = Chem.RWMol()         # RWMol is a molecule object intended to be edited

    # Remove "no atoms" and atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis =1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis = (0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atoms_i, atoms_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atoms_i == atoms_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atoms_i), int(atoms_j), bond_type)

    # Sanitize the molecule. For information about molecule sanitization, follow this link: https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # If sanitization fails, return NONE
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule


# STEP 5: DATALOADER

# DataLoader is a generic utility to be used as part of your application's data fetching layer.

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, mapping, max_len, batch_size = 6, shuffle= True):
        # Initialise
        self.data = data
        self.indices = self.data.index.tolist()
        self.mapping = mapping
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
            # Generate one batch of data.
            # Generate indices of the batch
            index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

            # Find list of IDs
            batch = [self.indices[k] for k in index]
            mol_features, mol_property = self.data_generation(batch)

            return mol_features, mol_property

    def on_epoch_end(self):
        # Should update the indexes after each epoch.
        self.index = np.arrange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, idx):
        # Load molecules adjacency matrix and features matrix from SMILE representation and their respective SAS values.
        qed = self.data.loc[idx]['qed']
        adjacency, features = smiles_to_graph(self.data.loc[idx]['smiles'])
        return adjacency, features, qed

    def data_generation(self, batch):
        x1 = np.empty((self.batch_size, BOND_DIM, self.max_len, self.max_len))
        x2 = np.empty((self.batch_size, self.max_len, len(self.mapping)))
        x3 = np.empty((self.batch_size, ))

        for i, batch_id in enumerate(batch):
            x1[i,], x2[i,], x3[i,] = self.load(batch_id)

            return [x1, x2], x3

# STEP 6: GENERATE THE TRAINING DATA SET

train_df = df.sample(frac=0.75, random_state=42)  #random state used for initializing the internal random number generator.
test_df = df.drop(train_df.index)    #Basically the test df consists of anything that is not in the trian df.
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Below class created using author's reference of: https://keras.io/examples/generative/wgan-graphs/
class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(
            self,
            units = 128,
            activation = "relu",
            use_bias = False,
            kernel_initializer = "glorot_uniform",
            bias_initializer = "zeros",
            kernel_regularizer = None,
            bias_regularizer = None,
            **kwargs
    ):
        # Call the objects created above
        super().__init__(**kwargs)

        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        self.kernel = self.add_weight(
            shape = (bond_dim, atom_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kerne_regularizer,
            trainable= True,
            name = "W",
            dtype = tf.float32,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape= (bond_dim, 1, self.units),
                initializer = self.bias.initializer,
                regularizer = self.bias_regularizer,
                trainable = True,
                name = "b",
                dtype = tf.float32,
            )
        self.built = True

    def call(self, inputs, training = False):
        adjacency, features = inputs
        # Aggregaete information from neighbors
        x = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
            # Reduce bond types dim
            x_reduced = tf.reduce_sum(x, axis=1)
            # Apply non-linear transformation.
            return self.activation(x_reduced)
