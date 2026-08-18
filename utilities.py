import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import lightning as L
from pathlib import Path
from sklearn.preprocessing import RobustScaler





class RegularDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.input_size = X.shape[1]
        if type(y) == np.ndarray:
            self.y= torch.tensor(y, dtype = torch.long)
        else:
            self.y= torch.tensor(y.to_numpy(), dtype = torch.long)
        self.len = len(self.x)
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        sample = self.x[idx]
        sample = torch.squeeze(sample)
        return sample



class RegressionDataset(Dataset):
    def __init__(self, X_tx, X_sm, y):
        self.X_tx = torch.tensor(X_tx.to_numpy(), dtype=torch.float32)
        self.X_sm = torch.tensor(X_sm.to_numpy(), dtype=torch.float32)
        self.len = len(y)
        self.input_size = X_tx.shape[1]
        
        if type(y) == np.ndarray:
            self.y = torch.tensor(y, dtype = torch.float32)
        else:
            self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        
        assert len(X_tx) == len(X_sm) == len(y), "Mismatch: X_tx, X_sm, and y must have the same length!"

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        tx = self.X_tx[idx]
        sm = self.X_sm[idx]
        y = self.y[idx].unsqueeze(-1)
        
        return tx, sm, y


class SiameseDataset_Fast(Dataset):
    def __init__(self, data, labels):
        assert(len(data) == len(labels)), "len(df) is not the same as len(classes)"
        self.data = torch.tensor(data.to_numpy(), dtype=torch.float32)
        self.input_size = self.data.shape[1]

        if type(labels) == np.ndarray:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)

        # Precompute indices as a dictionary, e.g { profile_A : [0,4,7] }
        self.class_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.class_indices[label.item()].append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # === Anchor ===
        anchor = self.data[idx]
        anchor = torch.squeeze(anchor)
        anchor_label = self.labels[idx].item()
        
        # === Positive Sample ===
        positive_indices = self.class_indices[anchor_label]

        if len(positive_indices) > 1:
            # Randomly selecting a positive_index
            positive_index = positive_indices[torch.randint(0, len(positive_indices), (1,)).item()]
            while positive_index == idx:
                positive_index = positive_indices[torch.randint(0, len(positive_indices), (1,)).item()]
            positive = self.data[positive_index]
            
        
        else:
            positive_index = idx
            positive = anchor + torch.rand_like(anchor)*0.8
        
        positive_label = anchor_label

        # === Negative Sample ===
        # Filter out keys that are the same as anchor_label
        negative_labels = [label for label in self.class_indices.keys() if label != anchor_label]
        # Randomly selecting a negative_label
        negative_label = negative_labels[torch.randint(0, len(negative_labels), (1,)).item()]
        # Randomly selecting an index from negative_label
        negative_index = self.class_indices[negative_label][torch.randint(0, len(self.class_indices[negative_label]), (1,)).item()]
        negative = self.data[negative_index]

        return anchor, positive, negative




def get_molecule_radius(smiles):
    """Determine a suitable radius based on the size of the molecule."""
    molecule = Chem.MolFromSmiles(smiles)
    num_atoms = molecule.GetNumAtoms()
    
    # Example heuristic: adjust radius based on the number of atoms
    if num_atoms <= 10:
        return 1  # Small molecule
    elif num_atoms <= 30:
        return 2  # Medium molecule
    else:
        return 3  # Large molecule

def smiles_to_morgan_fp(smiles, n_bits=128):
    """Generate a Morgan fingerprint with an appropriate radius."""
    radius = get_molecule_radius(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    fpgen = AllChem.GetMorganGenerator(radius)
    fingerprint = fpgen.GetFingerprintAsNumPy(molecule)
    return fingerprint


class CVDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        type: str,
        experiment: str,
        fold_n: int,
        batch_size: int,
        GEX_path: str,
        SMILES_path: str,
        num_workers: int = 12,
    ):

        super().__init__()
        self.root = Path(root)
        self.type = type
        self.experiment = experiment
        self.fold_n = fold_n
        self.bs = batch_size
        self.num_workers = num_workers

        # cache matrices
        self.GEX = pd.read_csv(GEX_path, index_col=0)
        self.SMILES = pd.read_csv(SMILES_path, index_col=0)

        # cache train statistics
        self.gex_mean = None
        self.gex_std = None
        self.ic50_mean = None
        self.ic50_std = None
        self.robust_scaler = None

    
    def _load_set(self, set_name:str):
        set = self.root / self.type / self.experiment / f'fold_{self.fold_n}' / f'{set_name}_set.csv'
        return pd.read_csv(set, index_col=0)

    def setup(self, stage:str = None):
        train_set = self._load_set(set_name = "train")
        val_set = self._load_set(set_name = "val")

        #############
        # GEX
        #############
        train_gex = self.GEX.loc[train_set.CELL_LINE_NAME]
        self.gex_mean = train_gex.mean(0)
        self.gex_std = train_gex.std(0)
        train_gex = (train_gex - self.gex_mean) / self.gex_std

        #############
        # SMILES
        #############
        train_sm = self.SMILES.loc[train_set.DRUG_NAME]

        #############
        # IC50
        #############
        self.ic50_mean = train_set.LN_IC50.mean()
        self.ic50_std = train_set.LN_IC50.std()
        
        train_y_true = (train_set["LN_IC50"] - self.ic50_mean) / self.ic50_std
        train_y_true = train_y_true.to_numpy()

        #################
        # VALIDATION SET
        #################

        val_gex = self.GEX.loc[val_set.CELL_LINE_NAME]
        val_gex = (val_gex - self.gex_mean) / self.gex_std

        val_sm = self.SMILES.loc[val_set.DRUG_NAME]
        
        val_y_true = (val_set["LN_IC50"] - self.ic50_mean) / self.ic50_std
        val_y_true = val_y_true.to_numpy()

        #############
        # DATASETS
        #############
        self.train_ds = RegressionDataset(train_gex, train_sm, train_y_true)
        self.val_ds = RegressionDataset(val_gex, val_sm, val_y_true)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size = self.bs,
            shuffle = True,
            num_workers = self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = self.bs,
            shuffle = False,
            num_workers = self.num_workers
        )
    
    def test_dataloader(self):
        # Train
        train_set = self._load_set(set_name = "train")
        train_gex = self.GEX.loc[train_set.CELL_LINE_NAME]
        ## GEX stats
        gex_mean = train_gex.mean(0)
        gex_std = train_gex.std(0)
        ## IC50 stats
        ic50_mean = train_set.LN_IC50.mean()
        ic50_std = train_set.LN_IC50.std()

        # Test set
        test_set = self._load_set(set_name = "test")
        ## GEX
        test_gex = self.GEX.loc[test_set.CELL_LINE_NAME]
        test_gex = (test_gex-gex_mean)/gex_std

        ## SMILES
        smiles = self.SMILES.loc[test_set.DRUG_NAME]

        ## Y TRUE
        test_set["Y_TRUE"] = (test_set["LN_IC50"]-ic50_mean)/ic50_std

        gex = torch.tensor(test_gex.to_numpy(), dtype=torch.float32)
        sm = torch.tensor(smiles.to_numpy(), dtype=torch.float32)

        return gex, sm, test_set





class EpsilonInsensitiveLoss(nn.Module):
    def __init__(self, epsilon, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        error = torch.abs(y_pred - y_true) - self.epsilon
        loss = torch.clamp(error, min=0.0)  # hinge-like
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# loss_fn = EpsilonInsensitiveLoss(epsilon=0.1)




def extract_state_dicts(state_dict):
    # Separate dictionaries for tx_nn, chem_nn, and comb_nn
    tx_state_dict = {k.replace("tx_nn.", ""): v for k, v in state_dict.items() if k.startswith("tx_nn")}
    chem_state_dict = {k.replace("chem_nn.", ""): v for k, v in state_dict.items() if k.startswith("chem_nn")}
    comb_state_dict = {k.replace("comb_nn.", ""): v for k, v in state_dict.items() if k.startswith("comb_nn")}

    return tx_state_dict, chem_state_dict, comb_state_dict

def precision_at_q(y_true, y_hat, q=0.25):
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)

    thr = np.quantile(y_true, q)
    true_pos = np.where(y_true <= thr)[0]
    k = len(true_pos)

    pred_topk = np.argsort(y_hat)[:k]

    return len(set(true_pos) & set(pred_topk)) / k

def ndcg_at_q(y_true, y_hat, q=0.25):
    "Normalized discounted cumulative gain"
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)

    thr = np.quantile(y_true, q)
    true_pos = np.where(y_true <= thr)[0]
    k = len(true_pos)

    # relevance: higher is better
    rel = -y_true
    
    # predicted ranking
    order = np.argsort(y_hat)
    rel_pred = rel[order][:k]
    
    discounts = 1 / np.log2(np.arange(2, k + 2)) # rank weight
    dcg = np.sum((2 ** rel_pred - 1) * discounts)

    # ideal ranking
    ideal_order = np.argsort(y_true)
    rel_ideal = rel[ideal_order][:k]
    idcg = np.sum((2 ** rel_ideal - 1) * discounts)

    return dcg / idcg if idcg > 0 else 0.0


# def prepare_input(cell_line:str,
#                   drug:str,
#                   conc_scaler=None,
#                   time_scaler=None,
#                   smiles_df=None,
#                   smiles=False,
#                  ):
    
#     conc_scaler = conc_scaler
#     time_scaler = time_scaler
    
#     idx = label_complete[(label_complete.str.contains(cell_line)) & 
#               (label_complete.str.contains(drug))].index.sort_values()
#     label = label_complete[(label_complete.str.contains(cell_line)) & 
#               (label_complete.str.contains(drug))].to_list()
    
#     cline_df = df.loc[idx]
    
#     experimental = pd.Series(label)
#     # concentration preprocessing
#     concentration = np.log10(
#     experimental.str.split(
#         "---", expand=True)[1].str.split(
#         " ", expand=True)[0].astype(float).to_numpy())
#     concentration = conc_scaler.transform(concentration.reshape(-1,1))
#     # time preprocessing
#     time_hours = experimental.str.split("---", expand=True)[2].str.split(" ", expand=True)[0].astype(int)
#     time_hours = time_hours.map(mapping).to_numpy().reshape(-1,1)
#     time_hours = time_scaler.transform(time_hours)
    
#     cline_df = np.concatenate([cline_df.to_numpy(), concentration, time_hours], axis=1)
    
#     if smiles==True:
#         assert smiles_df is not None, "Missing df containing vector smiles information"
        
#         smiles = smiles_df[smiles_df.cmap_name.str.contains("enzalutamide")].canonical_smiles
        
#         if smiles.isna().item() is True:
#             vs = np.zeros((cline_df.shape[0], 2048), dtype=int)
#         else:
#             vs = smiles_to_morgan_fp(smiles.to_list()[0]).reshape(1,-1)
#             assert np.sum(vs)>0, "Error on SMILES vector for drug, check manually"
        
#         cline_df = np.hstack([cline_df,
#                                 np.tile(vs, (cline_df.shape[0], 1))])
    
#     return cline_df, label