from MetaMobility import MetaMobility
from model_jodie import EmbeddingInitializer, PositionalEncoder
#from adjacency_tensor import build_A

import torch
import numpy as np
import tqdm
import os
import wandb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from itertools import chain
import seaborn as sns


def index_set(nb_data_start, nb_data_end, nb_events, path):
    sum = 0
    for i in range(nb_data_start, nb_data_end):
        events = torch.load(path + "/data_"+str(i)+".pt", weights_only=False).events
        sum += events.shape[0]
        if sum > nb_events:
            return i 

def train_evaluate(path, embedding_dynamic_size, epoch, device, nb_data, interval, kg_model, out_channels, city, lr, KG):

    wandb.init(project="hour={}_emb_size={}".format(interval, embedding_dynamic_size))

    data = torch.load("./processed/"+str(city)+"/"+kg_model+"/data_"+str(interval)+"/data_0.pt", weights_only=False)
    num_interaction = data.num_interaction
    num_users = data.num_users
    num_locas = data.num_locations
    
    if hasattr(data, 'num_actis'):
        num_actis = data.num_actis
    else: 
        num_actis = 222
    
    idx_train = index_set(0, nb_data, int(num_interaction * 0.7), path + "/data_" + str(interval))
    idx_val = index_set(idx_train, nb_data, int(num_interaction * 0.1), path + "/data_" + str(interval))
    
    #Gmulti_relabel, active_days = torch.load(path + "/data_"+str(interval)+"/processed/graph-days.pt")
    
    #build_A(os.getcwd(), Gmulti_relabel, path_motif, nb_motifs)
    #A = torch.load(os.getcwd() +"/"+ str(city) + "/A.pt")
    
    #super_edge_index = [A[i][0].coalesce().indices() for i in range(len(A))]
    #num_context = len(super_edge_index)

    super_edge_index = []
    num_context = len(super_edge_index)

    # positional encoder
    #if os.path.exists(path + "/data_" + str(interval) + "/processed/lon_lat_vector.pt"):
    #    path_lon_lat = path + "/data_" + str(interval) + "/processed/lon_lat_vector.pt"

    # model
    path_kg = "/home/rhaton/creat-knowledge-graph/knowledge_graph/logs/10_04/NYC_JSC/KG_11_08_47/model.pt"
    path_kge = path + "/data_" + str(interval) + "/kg_embedding.pt"
    edges_index_path = path + "/data_" + str(interval) + "/edges_index.pt"
    #metaMo = MetaMobility(num_users, num_locas, num_actis, embedding_dynamic_size, path_static, path_kg, device, num_context, out_channels, path_lon_lat).to(device)
    metaMo = MetaMobility(num_users, num_locas, num_actis, embedding_dynamic_size, path_kge, path_kg, device, num_context, out_channels, KG, city, edges_index_path).to(device)
        
    # initialize embedding
    embedding_initializer = EmbeddingInitializer(embedding_dynamic_size, num_users, num_locas, num_actis, device)
    embedding = embedding_initializer()
    X_meta = embedding.clone()
    X_jodie = embedding

    tsne = TSNE(n_components=2, random_state=0)
    kmeans = KMeans(n_clusters=5, random_state=0)

    #opt = torch.optim.Adam(metaMo.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_jodie = torch.optim.Adam(metaMo.jodie.parameters(), lr=lr, weight_decay=1e-5)
    #opt_meta = torch.optim.Adam(metaMo.metaGCNConv.parameters(), lr=1e-3, weight_decay=1e-5)

    train_loss = 0
    val_loss = 0
    test_loss = 0
    
    jodie_loss_epoch_train = 0
    jodie_loss_epoch_val = 0
    jodie_loss_epoch_test = 0

    meta_loss_epoch_train = 0
    meta_loss_epoch_val = 0
    meta_loss_epoch_test = 0

    epoch_loss_train = []
    epoch_loss_val = []
    epoch_loss_test = []

    torch.autograd.set_detect_anomaly(True)


    for ep in range(epoch):

        val_rank = []
        test_rank = []

        for i in tqdm.tqdm(range(idx_train), "Train progress bar"):
            break
            
            events = torch.load(path + "/data_" + str(interval) +"/data_" + str(i) + ".pt", weights_only=False).events.to(device)

            # with GNN
            #X_jodie, X_meta, loss_jodie, loss_meta = metaMo.forward(X_jodie, X_meta, events, super_edge_index)
            
            # without GNN
            X_jodie, _, loss_jodie, _ = metaMo.forward(X_jodie, X_jodie, events, super_edge_index)
            
            jodie_loss_epoch_train += loss_jodie.item()
            #meta_loss_epoch_train += loss_meta.item()
            
            train_loss += loss_jodie.item()
            #train_loss += loss_meta.item()

            # with one optimizer
            #loss = loss_jodie + loss_meta
            #loss.backward()
            #opt.step()
            #opt.zero_grad()
            #loss = 0

            # with two optimizers

            
            opt_jodie.zero_grad()
            loss_jodie.backward()

            for idx, param in enumerate(metaMo.parameters()):
                if param.grad is not None:
                    grad = param.grad.detach().cpu().numpy()
                    grad_mean = np.mean(grad)
                    grad_min = np.min(grad)
                    grad_max = np.max(grad)
                    wandb.log({"grad_mean":grad_mean})
                    wandb.log({"grad_min":grad_min})
                    wandb.log({"grad_max":grad_max})            
            
            opt_jodie.step()
            loss_jodie = 0
            X_jodie.detach_()
            #break
            #loss_meta.backward()
            #opt_meta.step()
            #opt_meta.zero_grad()
            #loss_meta = 0
            #X_meta.detach_()
               
        # with GNN
        #wandb.log({"loss_jodie":jodie_loss_epoch_train, "loss_meta":meta_loss_epoch_train})
        
        # without GNN
        wandb.log({"loss_jodie":jodie_loss_epoch_train})
        
        tsne_jodie = tsne.fit_transform(X_jodie.cpu().detach().numpy())
        kmeans_labels = kmeans.fit_predict(tsne_jodie)
        plt.figure(figsize=(10, 7))
        scatter_jodie = plt.scatter(tsne_jodie[:, 0], tsne_jodie[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.5)
        plt.colorbar(scatter_jodie)
        plt.title("t-SNE Visualization of JODIE Embeddings")
        wandb.log({"t-SNE Visualization of JODIE Embeddings": plt})
        """
        tsne_meta = tsne.fit_transform(X_meta[:num_users, :].cpu())
        kmeans_labels = kmeans.fit_predict(tsne_meta)
        plt.figure(figsize=(10, 7))
        scatter_meta = plt.scatter(tsne_meta[:, 0], tsne_meta[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.5)
        plt.colorbar(scatter_meta)
        plt.title("t-SNE Visualization of META Embeddings")
        wandb.log({"t-SNE Visualization of META Embeddings": plt})
        """
        # save model and optimizer before validation and testing
        #storage_embedding = embedding_dynamic
        state = {
            "model": metaMo.state_dict(),
            #"opt": opt.state_dict()
            "opt_jodie": opt_jodie.state_dict()
            #"opt_meta" : opt_meta.state_dict()
        }

        if not os.path.exists(path + "/data_" + str(interval) + "/saved_models_" + str(embedding_dynamic_size) + "/"):
            os.mkdir(path + "/data_" + str(interval) + "/saved_models_" + str(embedding_dynamic_size) +"/")

        filename = os.path.join(path + "/data_" + str(interval) + "/saved_models_" + str(embedding_dynamic_size), "models_{}".format(ep))
        torch.save(state, filename)
        
        top1_val = 0
        top5_val = 0
        top10_val = 0
        top20_val = 0

        top1_test = 0
        top5_test = 0
        top10_test = 0
        top20_test = 0

        num_interaction_val = 0
        num_interaction_test = 0

        for i in tqdm.tqdm(range(idx_train, nb_data), "Validation and Test"):
            
            events = torch.load(path + "/data_" + str(interval) + "/data_" + str(i) + ".pt", weights_only=False).events.to(device)

            # with GNN
            #X_jodie, loss_jodie, X_meta, loss_meta, top1, top5, top10, top20, num_interaction = metaMo.evaluate(X_jodie, X_meta, super_edge_index, events)
            
            # without GNN
            X_jodie, loss_jodie, _, _, top1, top5, top10, top20, num_interaction = metaMo.evaluate(X_jodie, X_jodie, super_edge_index, events)

            if i <= idx_val:
                top1_val += top1
                top5_val += top5
                top10_val += top10
                top20_val += top20
                num_interaction_val += num_interaction
                val_loss += loss_jodie.item()
                #val_loss += loss_meta.item()
                jodie_loss_epoch_val += loss_jodie.item()
                #meta_loss_epoch_val += loss_meta.item()

            else:
                top1_test += top1
                top5_test += top5
                top10_test += top10
                top20_test += top20
                num_interaction_test += num_interaction
                test_loss += loss_jodie.item()
                #test_loss += loss_meta.item()
                jodie_loss_epoch_test += loss_jodie.item()
                #meta_loss_epoch_test += loss_meta.item()
            
            # with one optimizer
            #loss = loss_jodie + loss_meta
            #loss.backward()
            #opt.step()
            #opt.zero_grad()
            #loss = 0

            # with two optimizers
            loss_jodie.backward()
            opt_jodie.step()
            opt_jodie.zero_grad()
            loss_jodie = 0
            X_jodie.detach_()
            
            #loss_meta.backward()
            #opt_meta.step()
            #opt_meta.zero_grad()
            #loss_meta = 0
            #X_meta.detach_()
            
        # with GNN
        #wandb.log({"val_loss_jodie":jodie_loss_epoch_val, "val_loss_meta":meta_loss_epoch_val})
        #wandb.log({"test_loss_jodie":jodie_loss_epoch_test, "test_loss_meta":meta_loss_epoch_test})

        # without GNN
        wandb.log({"val_loss_jodie":jodie_loss_epoch_val})
        wandb.log({"test_loss_jodie":jodie_loss_epoch_test})

        # calculate accuracy
        val_acc_1 = top1_val / num_interaction_val
        val_acc_5 = top5_val / num_interaction_val
        val_acc_10 = top10_val / num_interaction_val
        val_acc_20 = top20_val / num_interaction_val

        test_acc_1 = top1_test / num_interaction_test
        test_acc_5 = top5_test / num_interaction_test
        test_acc_10 = top10_test / num_interaction_test
        test_acc_20 = top20_test / num_interaction_test

        wandb.log({"perf_val_1":val_acc_1})
        wandb.log({"perf_test_1":test_acc_1})

        wandb.log({"perf_val_5":val_acc_5})
        wandb.log({"perf_test_5":test_acc_5})

        wandb.log({"perf_val_10":val_acc_10})
        wandb.log({"perf_test_10":test_acc_10})

        # calculate recall
        #val_recall_1 = sum(np.array(val_rank) <= 1) * 1.0 / len(val_rank)
        #val_recall_5 = sum(np.array(val_rank) <= 5) * 1.0 / len(val_rank)
        #val_recall_10 = sum(np.array(val_rank) <= 10) * 1.0 / len(val_rank)
        #val_recall_20 = sum(np.array(val_rank) <= 20) * 1.0 / len(val_rank)
        
        #test_recall_1 = sum(np.array(test_rank) <= 1) * 1.0 / len(test_rank)
        #test_recall_5 = sum(np.array(test_rank) <= 5) * 1.0 / len(test_rank)
        #test_recall_10 = sum(np.array(test_rank) <= 10) * 1.0 / len(test_rank)
        #test_recall_20 = sum(np.array(test_rank) <= 20) * 1.0 / len(test_rank)
        
        # calculate NDCG
        #val_ndcg_1 = sum(np.where(np.array(val_rank) <= 1, 1 / (np.log2(np.array(val_rank) + 1)), 0)) / len(val_rank)
        #val_ndcg_5 = sum(np.where(np.array(val_rank) <= 5, 1 / (np.log2(np.array(val_rank) + 1)), 0)) / len(val_rank) 
        #val_ndcg_10 = sum(np.where(np.array(val_rank) <= 10, 1 / (np.log2(np.array(val_rank) + 1)), 0)) / len(val_rank)
        #val_ndcg_20 = sum(np.where(np.array(val_rank) <= 20, 1 / (np.log2(np.array(val_rank) + 1)), 0)) / len(val_rank)
        
        #test_ndcg_1 = sum(np.where(np.array(test_rank) <= 1, 1 / (np.log2(np.array(test_rank) + 1)), 0)) / len(test_rank)
        #test_ndcg_5 = sum(np.where(np.array(test_rank) <= 5, 1 / (np.log2(np.array(test_rank) + 1)), 0)) / len(test_rank)
        #test_ndcg_10 = sum(np.where(np.array(test_rank) <= 10, 1 / (np.log2(np.array(test_rank) + 1)), 0)) / len(test_rank)
        #test_ndcg_20 = sum(np.where(np.array(test_rank) <= 20, 1 / (np.log2(np.array(test_rank) + 1)), 0)) / len(test_rank)
        
        if ep == 0:
            if os.path.exists(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) +  "/res_val.txt"):
                os.remove(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) + "/res_val.txt")

            if os.path.exists(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) + "/res_test.txt"):
                os.remove(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) + "/res_test.txt")

        if not os.path.exists(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size)):
            os.makedirs(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size))

        with open(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) + "/res_val.txt", "a") as fichier:
            fichier.write("Epoch: " + str(ep) + "\nAcc@1: " + str(val_acc_1) + ", Acc@5: " + str(val_acc_5) + ", Acc@10: " + str(val_acc_10) + ", Acc@20: " + str(val_acc_20) + "\n")
        #    fichier.write("Recall@1: " + str(val_recall_1) + ", Recall@5: " + str(val_recall_5) + ", Recall@10: " + str(val_recall_10) + ", Recall@20: " + str(val_recall_20) + "\n")
        #    fichier.write("NDCG@1: " + str(val_ndcg_1) + ", NDCG@5: " + str(val_ndcg_5) + ", NDCG@10: " + str(val_ndcg_10) + ", NDCG@20: " + str(val_ndcg_20) + "\n\n")
        
        with open(city +"/data_" + str(interval) + "/results_" + str(embedding_dynamic_size) + "/res_test.txt", "a") as fichier:
            fichier.write("Epoch: " + str(ep) + "\nAcc@1: " + str(test_acc_1) + ", Acc@5: " + str(test_acc_5) + ", Acc@10: " + str(test_acc_10) + ", Acc@20: " + str(test_acc_20) + "\n")
        #    fichier.write("Recall@1: " + str(test_recall_1) + ", Recall@5: " + str(test_recall_5) + ", Recall@10: " + str(test_recall_10) + ", Recall@20: " + str(test_recall_20) + "\n")
        #    fichier.write("NDCG@1: " + str(test_ndcg_1) + ", NDCG@5: " + str(test_ndcg_5) + ", NDCG@10: " + str(test_ndcg_10) + ", NDCG@20: " + str(test_ndcg_20) + "\n\n")
        
        embedding = embedding_initializer()
        X_jodie = embedding
        #X_meta = embedding.clone()

        print("Epoch: {}, Loss Train: {}, Loss Validation: {}, Loss Test: {}".format(ep, train_loss, val_loss, test_loss))
        print("          Loss Train JODIE: {}, Loss Validation JODIE: {}, Loss Test JODIE: {}".format(jodie_loss_epoch_train, jodie_loss_epoch_val, jodie_loss_epoch_test))
        print("          Loss Train META : {}, Loss Validation META : {}, Loss Test META : {}".format(meta_loss_epoch_train, meta_loss_epoch_val, meta_loss_epoch_test))

        epoch_loss_train.append(train_loss)
        epoch_loss_val.append(val_loss)
        epoch_loss_test.append(test_loss)
        
        train_loss = 0
        val_loss = 0
        test_loss = 0

        jodie_loss_epoch_train = 0
        meta_loss_epoch_train = 0
        jodie_loss_epoch_val = 0
        meta_loss_epoch_val = 0
        jodie_loss_epoch_test = 0
        meta_loss_epoch_test = 0

        # load model and optimizer for the next train epoch
        check = torch.load(path + "/data_" + str(interval) + "/saved_models_" + str(embedding_dynamic_size) +"/models_{}".format(ep), weights_only=False)
        metaMo.load_state_dict(check["model"])
        #opt.load_state_dict(check["opt"])
        opt_jodie.load_state_dict(check["opt_jodie"])
        #opt_meta.load_state_dict(check["opt_meta"])

    wandb.finish()

    return epoch_loss_train, epoch_loss_val, epoch_loss_test
