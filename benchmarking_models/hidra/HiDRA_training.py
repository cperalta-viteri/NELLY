"""
Training new HiDRA model
Requirement:
    expression.csv: Expression of all genes for all cell lines
    geneset.gmt: Gene set description file. Consists of Gene set name, Source, Gene set member 1, Gene set member 2, ...
    Training.csv: Training pair list. Consists of idx, Drug name, Cell line name, IC50 value for that pair.
    Validation.csv: Validation pair list. Consists of idx, Drug name, Cell line name, IC50 value for that pair.
    input_dir: The directory that includes input files.
"""

#Import basic packages
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse

#Import keras modules
import tensorflow as tf
import keras.backend as K
#import keras.backend.tensorflow_backend as KTF
from keras import backend as KTF
import keras
import keras.layers
from keras.layers import Layer
import keras.initializers
from keras.models import Model, Sequential,load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Multiply, multiply,dot
from keras.layers import Concatenate,concatenate
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import Sequence


#Fix the random seed
np.random.seed(5)



class HiDRADataGenerator(Sequence):
    def __init__(self, GDSC_df, train_gex, smiles, cellline_input, pathway_name, batch_size=256):
        self.GDSC_df = GDSC_df
        self.cellline_input = cellline_input
        # self.Input_directory = Input_directory
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.GDSC_df))
        self.shuffle = True
        self.on_epoch_end()

        # Load metadata once - GENE EXPRESSION OF THE SET
        self.GeneExpression_with_Symbol = train_gex

        # Load only pathway names first
        self.pathway_names = pathway_name

        # Pre-calculate pathway sizes
        self.pathway_sizes = {}
        with open("RawFile/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt") as f:
            for line in f:
                parts = line.strip().split('\t')
                self.pathway_sizes[parts[0]] = len(self.GeneExpression_with_Symbol.index.intersection(parts[2:]))

        # Drug features can be loaded in batches too
        self.drug_features = None
        self.drug_data = smiles

    def __len__(self):
        return (len(self.GDSC_df) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.GDSC_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_slice = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.GDSC_df.iloc[batch_slice]

        # Load drug features for this batch only
        drug_batch = self.drug_data.loc[batch_df.DRUG_NAME]

        pathway_data = [
            cell_lines.loc[batch_df.CELL_LINE_NAME].to_numpy()
            for cell_lines in self.cellline_input
        ]

        # Combine all inputs
        X = pathway_data + [drug_batch.values]
        X = tuple(a.astype('float32', copy=False) for a in X)
        y = batch_df['IC50']
        y = y.astype('float32', copy=False)

        return X, y

#Make new HiDRA model
def Making_Model(train_gex):
    #Read Gene expression and Gene set for model making
    #They are same with the source codes in the function 'Read_feils'
    #Read Gene expression file
    GeneExpression_with_Symbol=train_gex


    #Read Gene set file
    GeneSet_List=[]
    with open("RawFile/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt") as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter='\t')) #reads csv into a list of lists
        for row in data:
            GeneSet_List.append(row)

    GeneSet_Dic={}
    for GeneSet in GeneSet_List:
        GeneSet_Dic[GeneSet[0]]=GeneSet[2:]

    GeneSet_Dic_withoutNA={}
    for GeneSet in GeneSet_Dic:
        GeneSet_Dic_withoutNA[GeneSet]=GeneExpression_with_Symbol.index.intersection(GeneSet_Dic[GeneSet])
        

    #HiDRA model with keras
    #Drug-level network
    Drug_feature_length=512
    Drug_Input=Input((Drug_feature_length,), dtype='float32', name='Drug_Input')

    Drug_Dense1=Dense(256, name='Drug_Dense_1')(Drug_Input)
    Drug_Dense1=BatchNormalization(name='Drug_Batch_1')(Drug_Dense1)
    Drug_Dense1=Activation('relu', name='Drug_RELU_1')(Drug_Dense1)

    Drug_Dense2=Dense(128, name='Drug_Dense_2')(Drug_Dense1)
    Drug_Dense2=BatchNormalization(name='Drug_Batch_2')(Drug_Dense2)
    Drug_Dense2=Activation('relu', name='Drug_RELU_2')(Drug_Dense2)

    #Drug network that will be used to attention network in the Gene-level network and Pathway-level network
    Drug_Dense_New1=Dense(128, name='Drug_Dense_New1')(Drug_Input)
    Drug_Dense_New1=BatchNormalization(name='Drug_Batch_New1')(Drug_Dense_New1)
    Drug_Dense_New1=Activation('relu', name='Drug_RELU_New1')(Drug_Dense_New1)

    Drug_Dense_New2=Dense(32, name='Drug_Dense_New2')(Drug_Dense_New1)
    Drug_Dense_New2=BatchNormalization(name='Drug_Batch_New2')(Drug_Dense_New2)
    Drug_Dense_New2=Activation('relu', name='Drug_RELU_New2')(Drug_Dense_New2)

    #Gene-level network
    GeneSet_Model=[]
    GeneSet_Input=[]

    #Making networks whose number of node is same with the number of member gene in each pathway
    for GeneSet in GeneSet_Dic_withoutNA.keys():
        Gene_Input=Input(shape=(len(GeneSet_Dic_withoutNA[GeneSet]),),dtype='float32', name=GeneSet+'_Input')
        Drug_effected_Model_for_Attention=[Gene_Input]
        #Drug also affects to the Gene-level network attention mechanism
        Drug_Dense_Geneset=Dense(int(len(GeneSet_Dic_withoutNA[GeneSet])/4)+1,dtype='float32',name=GeneSet+'_Drug')(Drug_Dense_New2)
        Drug_Dense_Geneset=BatchNormalization(name=GeneSet+'_Drug_Batch')(Drug_Dense_Geneset)
        Drug_Dense_Geneset=Activation('relu', name=GeneSet+'Drug_RELU')(Drug_Dense_Geneset)
        Drug_effected_Model_for_Attention.append(Drug_Dense_Geneset) #Drug feature to attention layer

        Gene_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1,name=GeneSet+'_Concat')
        #Gene-level attention network
        Gene_Attention = Dense(len(GeneSet_Dic_withoutNA[GeneSet]), activation='tanh', name=GeneSet+'_Attention_Dense')(Gene_Concat)
        Gene_Attention=Activation(activation='softmax', name=GeneSet+'_Attention_Softmax')(Gene_Attention)
        Attention_Dot=dot([Gene_Input,Gene_Attention],axes=1,name=GeneSet+'_Dot')
        Attention_Dot=BatchNormalization(name=GeneSet+'_BatchNormalized')(Attention_Dot)
        Attention_Dot=Activation('relu',name=GeneSet+'_RELU')(Attention_Dot)

    #Append the list of Gene-level network (attach new pathway)
        GeneSet_Model.append(Attention_Dot)
        GeneSet_Input.append(Gene_Input)

    Drug_effected_Model_for_Attention=GeneSet_Model.copy()

    #Pathway-level network
    Drug_Dense_Sample=Dense(int(len(GeneSet_Dic_withoutNA)/16)+1,dtype='float32',name='Sample_Drug_Dense')(Drug_Dense_New2)
    Drug_Dense_Sample=BatchNormalization(name=GeneSet+'Sample_Drug_Batch')(Drug_Dense_Sample)
    Drug_Dense_Sample=Activation('relu', name='Sample_Drug_ReLU')(Drug_Dense_Sample)    #Drug feature to attention layer
    Drug_effected_Model_for_Attention.append(Drug_Dense_Sample)
    GeneSet_Concat=concatenate(GeneSet_Model,axis=1, name='GeneSet_Concatenate')
    Drug_effected_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1, name='Drug_effected_Concatenate')
    #Pathway-level attention
    Sample_Attention=Dense(len(GeneSet_Dic_withoutNA.keys()),activation='tanh', name='Sample_Attention_Dense')(Drug_effected_Concat)
    Sample_Attention=Activation(activation='softmax', name='Sample_Attention_Softmax')(Sample_Attention)
    Sample_Multiplied=multiply([GeneSet_Concat,Sample_Attention], name='Sample_Attention_Multiplied')
    Sample_Multiplied=BatchNormalization(name='Sample_Attention_BatchNormalized')(Sample_Multiplied)
    Sample_Multiplied=Activation('relu',name='Sample_Attention_Relu')(Sample_Multiplied)

    #Making input list
    Input_for_model=[]
    for GeneSet_f in GeneSet_Input:
        Input_for_model.append(GeneSet_f)
    Input_for_model.append(Drug_Input)

    #Concatenate two networks: Pathway-level network, Drug-level network
    Total_model=[Sample_Multiplied,Drug_Dense2]
    Model_Concat=concatenate(Total_model,axis=1, name='Total_Concatenate')

    #Response prediction network
    Concated=Dense(128, name='Total_Dense')(Model_Concat)
    Concated=BatchNormalization(name='Total_BatchNormalized')(Concated)
    Concated=Activation(activation='relu', name='Total_RELU')(Concated)

    Final=Dense(1, name='Output')(Concated)
    model=Model(inputs=Input_for_model,outputs=Final)

    return model

def main():
    #KTF.set_session(get_session())

    #Reading argument
    parser=argparse.ArgumentParser(
        description='HiDRA:Hierarchical Network for Drug Response Prediction with Attention-Training'
    )

    #Options
    parser.add_argument('--type', default="pancancer",type=str,help='Cancer type collection')
    parser.add_argument('--experiment', default="NBS_cells",type=str,help='')
    parser.add_argument('--fold', type=int, required=True, help='CV fold id')
    parser.add_argument('-e',type=str,help='The epoch in the training process')
    parser.add_argument('-o',type=str,help='The output path that model file be stored')

    args=parser.parse_args()
    

    type = args.type
    experiment = args.experiment

    type_options = ["pancancer", "solid_tumors"]
    experiment_options = ["NBS_cells", "NBS_drugs"]
    assert(type in type_options), f"Type is not available, choose {type_options}"
    assert(experiment in experiment_options), f"Experiment not available, chosse {experiment_options}"

    save_folder = f"{args.o}_{type}_{experiment}"
    os.makedirs(save_folder, exist_ok=True)

    expression_data = pd.read_csv(
        "../../msigdb_GEX_data_filtered_logCPM.csv",
        index_col=0
    )

    smiles = pd.read_csv(
        "../../vector_smiles_512.csv",
        index_col=0
    )

    n = args.fold

    train_idx = pd.read_csv(
        f"../../cross_validation/{type}/{experiment}/fold_{n}/train_set.csv",
        index_col=0
    )
    
    train_pairs = train_idx[["DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"]]
    train_pairs = train_pairs.rename(columns={"LN_IC50":"IC50"})

    train_gex = expression_data.loc[train_idx.CELL_LINE_NAME]
    gex_mean = train_gex.mean(0)
    gex_std = train_gex.std(0)
    train_gex = (train_gex-gex_mean)/gex_std

    train_drug = smiles.loc[train_idx.DRUG_NAME]

    assert(len(train_pairs)==len(train_gex))

    
    print("Entering cell feature extraction for Training")

    GeneSet_List=[]
    GeneSetFile='RawFile/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'
    with open(GeneSetFile) as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter='\t')) #reads csv into a list of lists
        for row in data:
            GeneSet_List.append(row)

    GeneSet_Dic={}
    for GeneSet in GeneSet_List:
        GeneSet_Dic[GeneSet[0]]=GeneSet[2:]
    
    GeneSet_Dic_withoutNA={}
    for GeneSet in GeneSet_Dic:
        GeneSet_Dic_withoutNA[GeneSet] = expression_data.columns.intersection(GeneSet_Dic[GeneSet]).to_list()


    pathway_name = list(GeneSet_Dic_withoutNA.keys())
    ### custom
    cellline_input = [
        expression_data[GeneSet_Dic_withoutNA[path]]  # Subset all cell lines at once for this gene set
        for path in pathway_name
    ]
    

    train_gex = train_gex.T
    train_gex.index = train_gex.index.rename("Gene_Symbol")


    training_input=HiDRADataGenerator(
        train_pairs,
        train_gex,
        smiles,
        cellline_input,
        pathway_name
    )

    #Training
    epoch=int(args.e)
    model=Making_Model(train_gex=train_gex)
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(training_input,
                shuffle=True,
                epochs=epoch,
                # batch_size=256,
                verbose=1,
                #validation_data=(validation_input,validation_label)
                )
    
    save_path = os.path.join(save_folder, f"model_{args.fold}.keras")
    model.save(save_path) #Save the model to the output directory



if __name__=="__main__":
    main()
