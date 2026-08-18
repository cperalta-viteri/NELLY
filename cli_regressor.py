from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics as tm
import numpy as np
import pandas as pd
from utilities import RegressionDataset, CVDataModule
from exp_regression import Transcriptomic_Head, SMILES_Head, CombinedModel



# USAGE:
# python cli_regressor.py
# OR, tunning parameters: 
# python cli_regressor.py --sn_1_dim 256 --sn_2_dim 128 --lr 1e-4 --wd 0.0 --max_epochs 100 --train_batch_size 512




#=======================================================================#
#
#               Model Module
#
#=======================================================================#


    # tx_head = Transcriptomic_Head(input_size=978,
    #                               hd_1=512,
    #                               hd_2=256
    #                               )
        
    # chem_head = SMILES_Head(input_size=512,
    #                         hd_1=512, 
    #                         hd_2=256, 
    #                         hd_3=128  
    #                         )
        
    # combined_model = CombinedModel(input_size=2048,
    #                                 hd_1=args.sn_1_dim,
    #                                 hd_2=args.sn_2_dim
    #                                 )
    

class Module_training_reg(L.LightningModule):
    def __init__(self,
                tx_nn,
                chem_nn,
                comb_nn,
                loss_fn,
                lr,
                weight_decay):
        super().__init__()
        self.tx_nn = tx_nn
        self.chem_nn = chem_nn
        self.comb_nn = comb_nn
        self.lr=lr
        self.weight_decay = weight_decay
        self.loss = loss_fn
        self.train_pcc = tm.PearsonCorrCoef()
        self.val_pcc = tm.PearsonCorrCoef()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
                    
        return optimizer

        
    def training_step(self, batch, batch_idx):
        tx, sm, y_true = batch

        sm = self.chem_nn(sm)
        
        tx = self.tx_nn(tx, sm)
        
        input = tx+sm
        y_hat = self.comb_nn(input)

        train_loss = self.loss(y_hat, y_true)
        self.train_pcc.update(y_hat.squeeze(), y_true.squeeze())
        
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)
        self.log("train_PCC", self.train_pcc, prog_bar=False, on_epoch=True, on_step=False)
        # self.log('lr',  self.trainer.optimizers[0].param_groups[0]['lr'])

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        tx, sm, y_true = batch
        
        sm = self.chem_nn(sm)
        
        tx = self.tx_nn(tx, sm)
        
        input = tx+sm
        y_hat = self.comb_nn(input)
        # y_hat = self.forward(gex, sm)

        val_loss = self.loss(y_hat, y_true)
        self.val_pcc.update(y_hat.squeeze(), y_true.squeeze())

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_PCC", self.val_pcc, prog_bar=False)

        return val_loss
    
    def on_train_epoch_end(self):
        self.log("epoch_train_PCC", self.train_pcc.compute())
        self.train_pcc.reset()
    
    def on_validation_epoch_end(self):
        self.log("epoch_val_PCC", self.val_pcc.compute())
        self.val_pcc.reset()

    def predict(self, gex, sm, mode="predict"):
        z_sm = self.chem_nn.predict(sm)
        z_gex = self.tx_nn.predict(gex, z_sm, mode=mode)

        return self.comb_nn.predict(z_gex+z_sm)


if __name__ == "__main__":
    np.random.seed(9497)
    torch.manual_seed(9497)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser()

    # model-related arguments
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--sn_1_dim", type=int, default=256, nargs="?")
    parser.add_argument("--sn_2_dim", type=int, default=128, nargs="?")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0, nargs="?", const=0)
    parser.add_argument("--max_epochs", type=int, default=100)

    parser.add_argument("--limit_train_batches", type=float, default=None, nargs="?")

    parser.add_argument("--type", type=str, default="pancancer", nargs="?")

    # parser.add_argument("--logname", type=str, default="L1000", nargs="?")
    # parser.add_argument("--chpoint", type=str, nargs="?", help= "Path to checkpoint")

    args = parser.parse_args()



############################################
#
#   CROSS VALIDATION
#
############################################

    base_seed = 9497

    for n in range(10):
        seed = base_seed+n
        L.seed_everything(seed, workers=True)

        # loss_fn = EpsilonInsensitiveLoss(epsilon=0.1)
        loss_fn = nn.MSELoss()
        
        root = "output/regression/cross_validation"
        cancer_type = args.type
        experiment = "NBS_cells"


        datamodule = CVDataModule(
            root = root,
            type = cancer_type,
            experiment = experiment,
            fold_n = n,
            GEX_path = "output/regression/L1000_GEX_data_filtered_logCPM.csv",
            SMILES_path = "output/regression/vector_smiles_512.csv",
            batch_size = args.train_batch_size,
            num_workers = 12
        )

        datamodule.setup()
        
        tx_head = Transcriptomic_Head(
            input_size=datamodule.train_ds.input_size, # 946 | 4377
            hd_1=512,
            hd_2=256
        )

        
        chem_head = SMILES_Head(
            input_size=512,
            hd_1=512, 
            hd_2=256,
            hd_3=128
        )

        
        combined_model = CombinedModel(
            input_size=1024,
            hd_1=args.sn_1_dim, #256
            hd_2=args.sn_2_dim  #128
        )

        
        # Instanciate Module
        regressor_model = Module_training_reg(
            tx_nn=tx_head,
            chem_nn = chem_head,
            comb_nn = combined_model,
            loss_fn = loss_fn,
            lr=args.lr,
            weight_decay=args.wd
        )

        tb_logger = TensorBoardLogger(save_dir="cDWM",
                                      name=f"L1000_{cancer_type}_{experiment}") #name=args.logname

        # Instanciating trainer
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            logger=tb_logger,
            limit_train_batches=args.limit_train_batches,
            log_every_n_steps= len(datamodule.train_dataloader()),
            num_sanity_val_steps=0
        )

            # Fitting model
        trainer.fit(
            model=regressor_model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
        )
                        
        torch.cuda.empty_cache()


    ############################################
    #
    #   STANDARD TRAINING
    #
    ############################################

    # regressor_model = Module_training_reg(tx_nn=tx_head,
    #                                 chem_nn = chem_head,
    #                                 comb_nn = combined_model,
    #                                 loss_fn = loss_fn,
    #                                 lr=args.lr,
    #                                 weight_decay=args.wd)

    # tb_logger = TensorBoardLogger(save_dir="ic50_logs", name="final_training") #name=args.logname

    # # Instanciating trainer
    # trainer = L.Trainer(max_epochs=args.max_epochs,
    #                         logger=tb_logger,
    #                         limit_train_batches=args.limit_train_batches,
    #                         log_every_n_steps= len(train_DL),
    #                         num_sanity_val_steps=0
    #                         )

    # # Fitting model
    # trainer.fit(model=regressor_model,
    #                 train_dataloaders=train_DL,
    #                 val_dataloaders=val_DL)