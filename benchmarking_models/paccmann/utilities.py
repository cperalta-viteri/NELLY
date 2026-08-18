import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import pandas as pd
import lightning as L
from pathlib import Path






class RegressionDataset(Dataset):
    def __init__(self, X_tx, X_sm, y):
        self.X_tx = torch.tensor(X_tx.to_numpy(), dtype=torch.float32)
        self.X_sm = X_sm
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
        sm = self.X_sm[idx].to(torch.float32)
        y = self.y[idx].unsqueeze(-1)
        
        return sm, tx, y




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
        SMILES_language,
        num_workers: int = 12,
    ):

        super().__init__()
        self.root = Path(root)
        self.type = type
        self.experiment = experiment
        self.fold_n = fold_n
        self.bs = batch_size
        self.num_workers = num_workers
        self.SMILES_language = SMILES_language 

        # cache matrices
        self.GEX = pd.read_csv(GEX_path, index_col=0)
        self.SMILES = pd.read_csv(SMILES_path, index_col=0, sep="\t")

    
    def _load_set(self, set_name:str):
        set = self.root / self.type / self.experiment / f'fold_{self.fold_n}' / f'{set_name}_set.csv'
        return pd.read_csv(set, index_col=0)

    def setup(self, stage:str = None):
        train_set = self._load_set(set_name = "train")

        #############
        # GEX
        #############
        train_gex = self.GEX.loc[train_set.CELL_LINE_NAME]
        gex_mean = train_gex.mean(0)
        gex_std = train_gex.std(0) + 1e-6
        train_gex = (train_gex - gex_mean) / gex_std

        #############
        # SMILES
        #############
        train_sm = self.SMILES.loc[train_set.DRUG_NAME]
        train_sm = list(
            map(self.SMILES_language.smiles_to_token_indexes, train_sm["smiles"].tolist())
        )

        #############
        # IC50
        #############
        ic50_mean = train_set.LN_IC50.mean()
        ic50_std = train_set.LN_IC50.std()
        train_y_true = (train_set["LN_IC50"] - ic50_mean) / ic50_std
        train_y_true = train_y_true.to_numpy()

        
        #############
        # DATASETS
        #############
        self.train_ds = RegressionDataset(train_gex, train_sm, train_y_true)
        
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
        gex_std = train_gex.std(0) + 1e-6
        ## IC50 stats
        ic50_mean = train_set.LN_IC50.mean()
        ic50_std = train_set.LN_IC50.std()

        # Test set
        test_set = self._load_set(set_name = "test")
        ## GEX
        test_gex = self.GEX.loc[test_set.CELL_LINE_NAME]
        test_gex = (test_gex-gex_mean)/gex_std

        y_true = test_set.LN_IC50

        ## SMILES
        smiles = self.SMILES.loc[test_set.DRUG_NAME]
        smiles = list(
            map(self.SMILES_language.smiles_to_token_indexes, smiles["smiles"].tolist())
        )
        smiles = [t.reshape(1,-1) for t in smiles]
        smiles = torch.cat(smiles, dim=0)
        smiles = smiles.to(torch.float32)

        ## Y TRUE
        test_set["Y_TRUE"] = (test_set["LN_IC50"]-ic50_mean)/ic50_std

        #gex = torch.tensor(test_gex.to_numpy(), dtype=torch.float32)

        self.test_ds = RegressionDataset(test_gex, smiles, y_true)

        test_dl = DataLoader(
            self.test_ds,
            batch_size = self.bs,
            shuffle = False,
            num_workers = self.num_workers
        )

        return test_dl, test_set





        