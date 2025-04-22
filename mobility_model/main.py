from SF_dataset import create_batch
from train_evaluate import train_evaluate
#from model_metaGNN import METAGCNConv, loss_gnn

import torch
import argparse
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="newyork", help="Name of the town in lower case and without spaces")
    parser.add_argument("--hour", default=12, help="Time interval in hours")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train the model")
    parser.add_argument("--embedding_dim", default=128, type=int, help="Number of dimensions of the dynamic embedding")
    parser.add_argument("--name_motif", default="motifs_1.json", help="Name of json file with patterns")
    parser.add_argument("--KG_model", default="TransE", help="Which model to use for knowledge Grapk Embedding")
    parser.add_argument("--lr", default=1e-3, type=float, help="Choose learning rate")
    parser.add_argument("--KG", default=False, type=bool, help="KGE or not")



    args = parser.parse_args()

    path_motif = os.path.join(os.getcwd(), args.name_motif)
   
    path = os.path.join(os.getcwd(), "processed/"+args.city, args.KG_model)

    if args.city == "newyork":
        dataset = create_batch(root="./", interval=args.hour, data_path="./datasets/dataset_TSMC2014_NYC.csv", name_city=args.city, matching_path="../knowledge_graph/logs/09_04/NYC_JSC/KG_13_23_32/entity2id_NYC_JSC.txt", kge_path="../knowledge_graph/logs/09_04/NYC_JSC/KG_13_23_32", kg_model=args.KG_model)
    
    if args.city == "sanfrancisco":
        dataset = create_batch(root="./", interval=args.hour, data_path="./datasets/sanfrancisco.csv", name_city=args.city, matching_path="../knowledge_graph/logs/09_04/NYC_JSC/KG_13_23_32/entity2id_NYC_JSC.txt", kge_path="../knowledge_graph/logs/09_04/NYC_JSC/KG_13_23_32", kg_model=args.KG_model)

    if args.city == "tokyo":
        dataset = create_batch(root="./", interval=args.hour, data_path="./datasets/dataset_TSMC2014_TKY.csv", name_city=args.city, matching_path="../knowledge_graph/Data_processed/TKY/entity2id_TKY.txt", kge_path="../knowledge_graph/logs/03_21/TKY/GIE_17_27_47")


    folder_path = os.path.join(os.getcwd(), "processed/" + args.city, args.KG_model + "/data_" + str(args.hour))
    
    if args.city == "sanfrancisco":
        nb_data = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]) - 2
    else:
        nb_data = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]) - 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = args.epochs
    embedding_dynamic_size = args.embedding_dim
    out_channels = args.embedding_dim

    epoch_loss_train, epoch_loss_val, epoch_loss_test = train_evaluate(path, embedding_dynamic_size, epoch, device, nb_data, args.hour, args.KG_model, out_channels, args.city, args.lr, args.KG)