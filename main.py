from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point
import random
import pickle
import os

from KG_sets import get_entity2id_relation2id, produce_train_val_test, get_train_val_test, KGDataset, process_dataset, get_embeddings
from train import train

class ArgsNamespace:
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)

if __name__ == "__main__":

    dataset_path = os.path.join(os.getcwd(), "Data_processed/NYC")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    KG_NYC = "Data_processed/NYC/UrbanKG_NYC.txt"
    entity2id_NYC = "./Data_processed/NYC/entity2id_NYC.txt"
    relation2id_NYC = "./Data_processed/NYC/relation2id_NYC.txt"
    triple_NYC = "./Data_processed/NYC/triplets_NYC.txt"
    train_NYC = './Data_processed/NYC/train'
    valid_NYC = './Data_processed/NYC/valid'
    test_NYC = './Data_processed/NYC/test'
    
    get_entity2id_relation2id(KG_NYC, entity2id_NYC, relation2id_NYC)
    produce_train_val_test(KG_NYC, entity2id_NYC, relation2id_NYC, triple_NYC)
    get_train_val_test(triple_NYC, train_NYC, valid_NYC, test_NYC)

    data_path = os.path.join(os.getcwd(), "Data_processed")

    for dataset_name in os.listdir(data_path):
        if dataset_name != "NYC":
            continue
        else:
            print(dataset_name)
            print("1",dataset_path)
            dataset_path = os.path.join(data_path, dataset_name)
            dataset_examples, dataset_filters = process_dataset(dataset_path, dataset_name)
            for dataset_split in ["train", "valid", "test"]:
                save_path = os.path.join(dataset_path, dataset_split + ".pickle")
                with open(save_path, "wb") as save_file:
                    pickle.dump(dataset_examples[dataset_split], save_file)
            with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
                pickle.dump(dataset_filters, save_file)

    #KG = KGDataset(dataset_path, False)
"""
    # Configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    args_embeddings = {
        "dataset": "NYC",
        "model": "GIE",
        "optimizer": "Adam",
        "max_epochs": 300,
        "patience": 10,
        "valid": 3,
        "rank": 32,
        "batch_size": 4120,
        "learning_rate": 1e-3,
        "neg_sample_size": 50,
        "init_size": 1e-3,
        "multi_c": True,
        "regularizer": "N3",
        "reg": 0,
        "dropout": 0,
        "gamma": 0,
        "bias": "constant",
        "dtype": "double",
        "double_neg": False,
        "debug": False,
    }
    args_embeddings_obj = ArgsNamespace(args_embeddings)
    entity_embeddings, pca_embeddings, tsne_embeddings = get_embeddings(args_embeddings_obj)

    
## Model Embedding 
    args_model = {
        "dataset": "NYC",  # Dataset à utiliser : "NYC" ou "CHI"
        "model": "GIE",  # Nom du modèle
        "optimizer": "Adam",  # Optimiseur : "Adagrad", "Adam" ou "SparseAdam"
        "max_epochs": 300,  # Nombre maximal d'époques d'entraînement
        "patience": 10,  # Nombre d'époques avant arrêt anticipé
        "valid": 3,  # Nombre d'époques entre les validations
        "rank": 32,  # Dimension des embeddings
        "batch_size": 4120,  # Taille des lots
        "learning_rate": 1e-3,  # Taux d'apprentissage
        "neg_sample_size": 50,  # Taille des échantillons négatifs (-1 pour désactiver)
        "init_size": 1e-3,  # Échelle des embeddings initiaux
        "multi_c": True,  # Utiliser plusieurs courbures par relation
        "regularizer": "N3",  # Régularisateur : "N3" ou "F2"
        "reg": 0.0,  # Poids de la régularisation
        "dropout": 0.0,  # Taux de dropout
        "gamma": 0.0,  # Marge pour les pertes basées sur la distance
        "bias": "constant",  # Type de biais : "constant", "learn" ou "none"
        "dtype": "double",  # Précision machine : "single" ou "double"
        "double_neg": False,  # Échantillonner négativement les entités tête et queue
        "debug": False,  # Mode debug : utiliser 1000 exemples seulement
    }
    args_model_obj = ArgsNamespace(args_model)    
    train(args_model_obj)
"""