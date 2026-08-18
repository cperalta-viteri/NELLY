#Import basic packages
import numpy as np
import pandas as pd
import csv

import os
import argparse

#Import keras modules
import tensorflow as tf
import keras.backend as K
#import keras.backend.tensorflow_backend as KTF
import keras
import keras.layers
from keras.layers import Layer 
import keras.initializers
from keras.models import Model, Sequential,load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Multiply, multiply,dot
from keras.layers import Concatenate,concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

#Fix the random seed
np.random.seed(5)





def main():
    #KTF.set_session(get_session())

    #Reading argument 
    parser=argparse.ArgumentParser(
        description='HiDRA:Hierarchical Network for Drug Response Prediction with Attention-Predict'
    )
    
    #Options
    parser.add_argument('--type', default="pancancer",type=str,help='Cancer type collection')
    parser.add_argument('--experiment', default="NBS_cells",type=str,help='')
    parser.add_argument('--fold', type=int, required=True, help='CV fold id -- Job Array')
    parser.add_argument('-o', default="prediction",type=str,help='The output file path that prediction result be stored')
    parser.add_argument('-m',type=str,help='Path to the models')

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

    #################################################
    # TRAIN STATS
    #################################################

    train_idx = pd.read_csv(
        f"../../cross_validation/{type}/{experiment}/fold_{n}/train_set.csv",
        index_col=0
    )
    
    train_pairs = train_idx[["DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"]]
    train_pairs = train_pairs.rename(columns={"LN_IC50":"IC50"})

    train_gex = expression_data.loc[train_idx.CELL_LINE_NAME]
    gex_mean = train_gex.mean(0)
    gex_std = train_gex.std(0)
    # train_gex = (train_gex-gex_mean)/gex_std

    # train_drug = smiles.loc[train_idx.DRUG_NAME]

    assert(len(train_pairs)==len(train_gex))

    #################################################
    # TEST SET
    #################################################
    test_idx = pd.read_csv(
        f"../../cross_validation/{type}/{experiment}/fold_{n}/test_set.csv",
        index_col=0
    )

    test_gex = expression_data.loc[test_idx.CELL_LINE_NAME]
    test_gex = (test_gex-gex_mean)/gex_std

    test_drug = smiles.loc[test_idx.DRUG_NAME]
    assert(len(test_idx) == len(test_gex))

    
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


    pathway_data = [
        cell_lines.loc[test_idx.CELL_LINE_NAME].to_numpy()
        for cell_lines in cellline_input
    ]

    # Combine all inputs
    X = pathway_data + [test_drug.values]
    X = tuple(a.astype('float32', copy=False) for a in X)

   
    #Read input files from predict list 
    # model_input=read_files(args.p,args.i)
    #Load model in hdf5 file format
    model_path = f"{args.m}/model_{n}.keras"
    model=load_model(model_path,compile=False)
    #Predict
    result=model.predict(X)
    result=np.ravel(result).astype(np.float64)
    # result=[y[0] for y in result]
    # predict_list=pd.read_csv(args.p)
    # predict_list['result']=result
    #Save the predict results to the output directory
    pd.Series(result).to_csv(f"{save_folder}/model_{n}.csv", index=False)
    
if __name__=="__main__":
    main()
