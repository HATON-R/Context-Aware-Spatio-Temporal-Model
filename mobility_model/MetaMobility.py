from model_jodie import JODIE
#from model_metaGNN import METAGCNConv, loss_gnn
import torch
from torch import nn
import math

class MetaMobility(nn.Module):
    def __init__(self, num_users, num_locas, num_actis, embedding_dynamic_size, path_kge, path_kg, device, num_context, out_channels, KG=False, city="newyork", edges_index_path=None):#, path_lon_lat):
        super(MetaMobility, self).__init__()
        self.device = device
        self.embedding_dynamic_size = embedding_dynamic_size
        #self.embedding_static = torch.Tensor(torch.load(path_static).X_static).to(self.device)
        #self.embedding_static_size = self.embedding_static.shape[1]
        self.num_users = num_users
        self.num_locas = num_locas
        self.num_actis = num_actis
        self.num_context = num_context
        self.out_channels = out_channels
        self.KG = KG
        self.city = city
        self.edges_index_path = edges_index_path
        #self.path_lon_lat = path_lon_lat
        #self.jodie = JODIE(self.embedding_dynamic_size, path_static, path_kg, self.num_users, self.num_locas, self.num_actis, self.path_lon_lat, self.device).to(self.device)
        self.jodie = JODIE(self.embedding_dynamic_size, path_kge, path_kg, self.num_users, self.num_locas, self.num_actis, self.device, self.KG, self.city, self.edges_index_path).to(self.device)
        #self.metaGCNConv = METAGCNConv(self.num_context, self.embedding_dynamic_size, self.out_channels, "cpu")#self.device)
        
    def forward(self, X_jodie, X_meta, events, super_edge_index):
        """
        idx = set()
        for idx_u, idx_l, _, _, _, _ in events:
            idx.add(int(idx_u))
            idx.add(int(idx_l))

        H_meta, loss_meta = self.train_metagnn(X_meta, super_edge_index)
        
        idx = torch.tensor(list(idx))
        H_clone = X_meta.clone()
        H_clone[idx, :] =  H_meta[idx, :].to(self.device).detach()
        X_meta = H_clone
        
        embedding_meta = X_meta.detach()
        """
        X_jodie, loss_jodie = self.train_jodie(X_jodie, events)

        return X_jodie, X_meta, loss_jodie, loss_jodie


    def evaluate(self, X, X_meta, super_edge_index, events):
        """
        idx = set()
        for idx_u, idx_l, _, _, _, _ in events:
            idx.add(int(idx_u))
            idx.add(int(idx_l))

        H_meta, loss_meta = self.train_metagnn(X_meta, super_edge_index)
        X_meta[list(idx), :] =  H_meta[list(idx), :].to(self.device).detach()
        embedding_meta = X_meta.detach()
        """
        X, loss, top1, top5, top10, top20, num_interaction = self.evaluate_jodie(X, events, X)

        return X, loss, X, loss, top1, top5, top10, top20, num_interaction

    
    def train_metagnn(self, X, super_edge_index):
        X = self.metaGCNConv.forward(X, super_edge_index)
        loss = loss_gnn(self.metaGCNConv, X.to("cpu"), super_edge_index, self.num_context)
        return X, loss


    def train_jodie(self, X, events):
        X, loss = self.jodie.forward(X, events)
        return X, loss

  
    def evaluate_jodie(self, X, events, X_meta):

        [embedding,
        loss,
        top1, 
        top5, 
        top10, 
        top20, 
        num_interaction] = self.jodie.evaluate(X, events, X_meta)
        
        return embedding, loss, top1, top5, top10, top20, num_interaction