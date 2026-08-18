import torch
from torch import nn
import torch.nn.functional as F






################################################################################
#
#                   DYNAMIC WEIGHTING MECHANISM
#
################################################################################
        
    
class DrugAware_DWM(nn.Module):
    def __init__(self, input_size, smiles_emb_size=1024):
        super(DrugAware_DWM, self).__init__()

        self.dropout = nn.Dropout(.25) # .25
        self.batchnorm = nn.BatchNorm1d(input_size, affine=False, track_running_stats=False)
        
        self.weights = nn.Linear(input_size+smiles_emb_size, input_size)

    def forward(self, x, z_smiles, return_weights=False):

        x_do = self.dropout(x)
        x_do = self.batchnorm(x_do)
        x_do = F.tanh(x_do)

        c_x = torch.concat([x_do, z_smiles], dim=1)

        # Sample-specific weights
        weights = self.weights(c_x) # [batch_size, input_size]

        # Apply sigmoid to get weights
        weights = F.sigmoid(weights) # [batch_size, input_size] #sigmoid
        
        # Apply weights to input
        attended = (x * weights) + x # [batch_size, input_size]

        return (attended, weights) if return_weights else attended
    
    def predict(self, x, z_smiles, return_weights=False):
        self.eval()
        with torch.no_grad():
            if return_weights:
                output, w = self.forward(x, z_smiles, return_weights=True)
                return output, w
            else:
                output = self.forward(x, z_smiles, return_weights=return_weights)
                return output
    

class Transcriptomic_Head(nn.Module):
    def __init__(self, input_size, hd_1, hd_2):
        super(Transcriptomic_Head, self).__init__()
        self.input_size = input_size

        self.scaler = DrugAware_DWM(input_size)

        self.sequential = nn.Sequential(
            nn.Dropout(.5), # .3 
            nn.BatchNorm1d(input_size), 

            nn.Linear(input_size, hd_1),  
            nn.BatchNorm1d(hd_1),
            nn.LeakyReLU(), 
            nn.Dropout(.2), # .2 

            nn.Linear(hd_1, hd_2),
            nn.BatchNorm1d(hd_2),
            nn.LeakyReLU(),
            nn.Dropout(.2), # .2

            nn.Linear(hd_2, 1024),

            nn.Tanh(),
            nn.Dropout(.2)
            
            )

    def forward(self, x, sm_emb):
        x = self.scaler(x, sm_emb)
        x = self.sequential(x)

        return x
    
    def predict(self, input_tensor, z_smiles=None, mode="predict"):
        modes = ["exploratory", "dim_red", "predict"]
        assert(mode in modes), f"Mode is not one of the modes, choose between {modes}"

        self.eval()
        if input_tensor.dim() ==  1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            if mode == "exploratory" : # returns both z_gex and W matrix from DWM
                gex_tilde, w = self.scaler.predict(input_tensor, z_smiles, return_weights=True)
                z_gex = self.sequential(gex_tilde)
                return z_gex, w

            elif mode == "dim_red": # does not appy DWM
                z_gex = self.sequential(input_tensor)
                return z_gex

            elif mode == "predict": # applies DWM but does not return W matrix
                # z_gex = self.sequential(input_tensor) ---> uncomment for naive & comment out next lines
                
                gex_tilde = self.scaler.predict(                      # ----> comment out for naive
                    input_tensor, z_smiles, return_weights = False    # ----> comment out for naive
                )                                                     # ----> comment out for naive
                z_gex = self.sequential(gex_tilde)                    # ----> comment out for naive

                return z_gex



class SMILES_Head(nn.Module):
    def __init__(self, input_size, hd_1, hd_2, hd_3):
        super(SMILES_Head, self).__init__()

        self.ln = nn.LayerNorm(input_size)

        self.sequential = nn.Sequential(
            #### 
            nn.Linear(input_size, hd_1),
            nn.LayerNorm(hd_1), 
            nn.LeakyReLU(),
            nn.Dropout(.2),

            nn.Linear(hd_1, hd_2),
            nn.LayerNorm(hd_2),
            nn.LeakyReLU(),
            nn.Dropout(.2),

            nn.Linear(hd_2, hd_3),
            nn.LayerNorm(hd_3),
            nn.LeakyReLU(),
            nn.Dropout(.2),

            nn.Linear(hd_3, 1024),
            nn.Tanh(),
            nn.Dropout(.2)
            )
        
    def forward(self, x):
        x = self.ln(x)
        
        # Forward pass for SMILES of anchor, positive, and negative samples
        x = self.sequential(x)

        return x
    
    def predict(self, input_tensor):
        self.eval()
        if input_tensor.dim() ==  1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            input_tensor = self.ln(input_tensor)
            output = self.sequential(input_tensor)
        return output


class CombinedModel(nn.Module):
    def __init__(self, input_size, hd_1, hd_2):
        super(CombinedModel, self).__init__()

        self.input_size = input_size

        self.sequential = nn.Sequential(
            
            nn.BatchNorm1d(input_size), 
            
            #### 1024
            nn.Linear(input_size, hd_1),
            nn.BatchNorm1d(hd_1),
            nn.LeakyReLU(),
            nn.Dropout(.2), #.2 | .25

            #### 256
            nn.Linear(hd_1, hd_2),
            nn.BatchNorm1d(hd_2),
            nn.LeakyReLU(),
            nn.Dropout(.2), #.2 | .25

            nn.Linear(hd_2, 1)
            )
        
    def forward(self, x):

        # Forward pass 
        y_pred = self.sequential(x)

        return y_pred
    
    def predict(self, input_tensor):
        self.eval()
        if input_tensor.dim() ==  1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.sequential(input_tensor) 
        return output
