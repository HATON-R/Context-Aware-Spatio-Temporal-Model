import torch
from torch import nn
from torch.nn import RNNCell
from torch.nn import functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import wandb

def PositionalEncoder(path_lon_lat, sigma_min=1e-6, sigma_max=360, frequence=100):

    coordinate_vector = torch.tensor(torch.load(path_lon_lat))

    # frequency 
    log_timescale_increment = (math.log(float(sigma_max) / float(sigma_min)) / (frequence*1.0 - 1))
    timescales = sigma_min * np.exp(np.arange(frequence).astype(float) * log_timescale_increment)
    freq = 1.0/timescales
    freq_mat = np.expand_dims(freq, axis = 1)
    freq_mat = np.repeat(freq_mat, 2, axis = 1)

    # positional encoder
    if coordinate_vector.shape == (coordinate_vector.shape[0], coordinate_vector.shape[1]):
        coordinate_vector = coordinate_vector.reshape(1, coordinate_vector.shape[0], coordinate_vector.shape[1])

    coordinate_vector_mat = np.asarray(coordinate_vector).astype(float)
    
    batch_size = coordinate_vector_mat.shape[0]
    num_context_pt = coordinate_vector_mat.shape[1]
    
    coordinate_vector_mat = np.expand_dims(coordinate_vector_mat, axis = 3)

    coordinate_vector_mat = np.expand_dims(coordinate_vector_mat, axis = 4)
    
    coordinate_vector_mat = np.repeat(coordinate_vector_mat, frequence, axis = 3)
    
    coordinate_vector_mat = np.repeat(coordinate_vector_mat, 2, axis = 4)
    
    embeds = coordinate_vector_mat * freq_mat
    
    embeds[:, :, :, :, 0::2] = np.sin(embeds[:, :, :, :, 0::2])
    embeds[:, :, :, :, 1::2] = np.cos(embeds[:, :, :, :, 1::2])
    
    embeds = np.reshape(embeds, (batch_size, num_context_pt, -1))
    embeds = torch.FloatTensor(embeds)
    
    return F.normalize(embeds.squeeze(0))

class EmbeddingInitializer(nn.Module):
    def __init__(self, embedding_size, num_users, num_locas, num_actis, device):
        super(EmbeddingInitializer, self).__init__()
        self.device = device
        self.num_users = num_users
        self.num_locas = num_locas
        self.num_actis = num_actis
        self.init_embedding_user = nn.Parameter(F.normalize(torch.rand(embedding_size).to(self.device), dim=0)) 
        self.init_embedding_loca = nn.Parameter(F.normalize(torch.rand(embedding_size).to(self.device), dim=0)) 
        #self.init_embedding_acti = nn.Parameter(F.normalize(torch.rand(embedding_size).to(self.device), dim=0)) 


    def forward(self):
        embedding_user = self.init_embedding_user.repeat(self.num_users, 1)
        embedding_loca = self.init_embedding_loca.repeat(self.num_locas, 1)
        #embedding_acti = self.init_embedding_acti.repeat(self.num_actis, 1)
        embe_dynamic = torch.cat([embedding_user, embedding_loca])#, embedding_acti])
        return embe_dynamic

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class JODIE(nn.Module):
    def __init__(self, embedding_dynamic_size, path_static, path_kg, num_users, num_locas, num_actis, path_lon_lat, device):
        super(JODIE, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.num_users = num_users
        self.num_locas = num_locas
        self.num_actis = num_actis
        self.embedding_dynamic_size = embedding_dynamic_size
        self.device = device

        self.initial_user_embedding = nn.Parameter(torch.Tensor(self.embedding_dynamic_size)).to(self.device)
        self.initial_item_embedding = nn.Parameter(torch.Tensor(self.embedding_dynamic_size)).to(self.device)
        
        self.embedding_static = torch.Tensor(torch.load(path_static).X_static).to(self.device)
        self.embedding_static = self.embedding_static.to(torch.float64)
        self.embedding_static_size = self.embedding_static.size(1)

        self.embedding_kg = torch.load(path_kg)["entity.weight"].to(self.device)

        self.positional_embedding = PositionalEncoder(path_lon_lat).to(self.device)

        self.activity = torch.eye(self.num_actis).to(self.device) # identity matrix
        #self.activity = torch.zeros(self.num_actis, self.num_actis).to(self.device) # null matrix
        
        input_rnn_user_size = self.embedding_dynamic_size + self.embedding_kg.size(1) + 1 #+ self.activity.size(1) #self.positional_embedding.size(1)  
        input_rnn_loca_size = self.embedding_dynamic_size + self.embedding_kg.size(1) + 1 #+ self.activity.size(1) #self.positional_embedding.size(1)
        self.rnn_user = RNNCell(input_rnn_user_size, self.embedding_dynamic_size, nonlinearity="tanh").to(self.device)
        self.rnn_loca = RNNCell(input_rnn_loca_size, self.embedding_dynamic_size, nonlinearity="tanh").to(self.device)
        self.layer_projection = NormalLinear(1, self.embedding_dynamic_size).to(self.device)
        self.layer_norm = nn.LayerNorm((1, self.embedding_dynamic_size + self.embedding_kg.size(1))).to(self.device)
        
        #self.layer_before_concat = nn.Linear(self.embedding_kg.size(1), self.embedding_dynamic_size)
        self.layer_prediction = nn.Linear(2 * self.embedding_static_size + 2 * self.embedding_dynamic_size + self.embedding_kg.size(1) , self.embedding_static_size + self.embedding_dynamic_size)

    def replace_update(self, X, idx, vector):
        tmp = X.detach().clone()
        tmp[[idx[0]], :] = vector
        X = tmp
        return X
    
    def forward(self, embedding, events):
        loss = 0
        #print(events[0])
        for idx_user, idx_loca, time, delta_u, delta_l, idx_prev, idx_cate, idx_prev_cate, idx_know in events:
            #print(idx_user, idx_loca, idx_prev, idx_cate)
            idx_spat = [int(idx_prev)]
            idx_loca = [int(idx_loca + self.num_users)]
            idx_prev = [int(idx_prev + self.num_users)]
            idx_user = [int(idx_user)]
            idx_prev_cate = [int(idx_prev_cate)]
            idx_know = [int(idx_know)]

            #break

            projected_embedding_user = self.projection(embedding[idx_user, :],
                                                       torch.Tensor([delta_u]).to(self.device))

            if idx_know == -1:
                embedding_loca_acti_norm = self.layer_norm(torch.cat([embedding[idx_prev, :], torch.mean(self.embedding_kg, dim=0)], dim=1))
            else:
                embedding_loca_acti_norm = self.layer_norm(torch.cat([embedding[idx_prev, :], self.embedding_kg[idx_know, :]], dim=1))

            embedding_user_loca_meta = torch.cat([projected_embedding_user,
                                                  embedding_loca_acti_norm,
                                                  self.embedding_static[idx_prev, :],
                                                  self.embedding_static[idx_user, :]],
                                                  dim=1)

            embedding_predict = self.predict_embedding_loca(embedding_user_loca_meta)

            loss = loss + torch.nn.MSELoss()(embedding_predict, 
                                             torch.cat([embedding[idx_loca, :], self.embedding_static[idx_loca, :]], dim=1).detach())
            wandb.log({"loss_pred":torch.nn.MSELoss()(embedding_predict, 
                                             torch.cat([embedding[idx_loca, :], self.embedding_static[idx_loca, :]], dim=1).detach())})

            update_embedding_user = self.update_rnn_user(embedding[idx_user, :], 
                                                         embedding[idx_loca, :],
                                                         idx_know,
                                                         #self.positional_embedding[idx_spat, :],
                                                         #self.activity[idx_cate, :],
                                                         torch.Tensor([delta_u]).to(self.device))
            
            update_embedding_loca = self.update_rnn_loca(embedding[idx_loca, :], 
                                                         embedding[idx_user, :],
                                                         idx_know,
                                                         #self.positional_embedding[idx_spat, :],
                                                         #self.activity[idx_cate, :],
                                                         torch.Tensor([delta_l]).to(self.device))
            
            loss = loss + torch.nn.MSELoss()(update_embedding_user, embedding[idx_user, :].detach())
            loss = loss + torch.nn.MSELoss()(update_embedding_loca, embedding[idx_loca, :].detach())

            wandb.log({"loss_user":torch.nn.MSELoss()(update_embedding_user, embedding[idx_user, :].detach())})
            wandb.log({"loss_loca":torch.nn.MSELoss()(update_embedding_loca, embedding[idx_loca, :].detach())})

            embedding[idx_user, :] = update_embedding_user
            embedding[idx_loca, :] = update_embedding_loca
            
        return embedding, loss
    
    def update_rnn_user(self, embedding_user, embedding_loca, idx_know, delta_u):
        #if idx_know == -1:
        #    input_concat = torch.cat([embedding_loca, self.embedding_kg.mean(dim=0), delta_u.reshape(-1, 1)], dim=1)
        #    output = self.rnn_user(input_concat, embedding_user)
        #    return F.normalize(output)
        #else:
        input_concat = torch.cat([embedding_loca, self.embedding_kg[idx_know, :], delta_u.reshape(-1, 1)], dim=1)
        output = self.rnn_user(input_concat, embedding_user)
        return F.normalize(output)

    def update_rnn_loca(self, embedding_loca, embedding_user, idx_know, delta_l):
        #if idx_know == -1:
        #    input_concat = torch.cat([embedding_user, self.embedding_kg.mean(dim=0), delta_l.reshape(-1, 1)], dim=1)
        #    output = self.rnn_loca(input_concat, embedding_loca)
        #    return F.normalize(output)
        #else:
        input_concat = torch.cat([embedding_user, self.embedding_kg[idx_know, :], delta_l.reshape(-1, 1)], dim=1)
        output = self.rnn_loca(input_concat, embedding_loca)
        return F.normalize(output)

    def predict_embedding_loca(self, embedding_user):
        output = self.layer_prediction(embedding_user)
        return output

    def projection(self, embedding_user, delta_u):
        projected_user = embedding_user * (1 + self.layer_projection(delta_u))
        return projected_user
    
    
    def evaluate(self, embedding, events, embedding_meta):

        loss = 0
        num_interaction = 0
        top1 = 0
        top5 = 0
        top10 = 0
        top20 = 0
        #print(events[0])
        for idx_user, idx_loca, time, delta_u, delta_l, idx_prev, idx_cate, idx_prev_cate, idx_know in events:
            #print(idx_user, idx_loca, idx_prev, idx_cate)
            idx_spat = [int(idx_prev)]
            idx_loca = [int(idx_loca + self.num_users)]
            idx_prev = [int(idx_prev + self.num_users)]
            idx_user = [int(idx_user)]
            idx_prev_cate = [int(idx_prev_cate)]
            idx_know = [int(idx_know)]
            
            num_interaction += 1
            X_proj = self.projection(embedding[idx_user, :], torch.Tensor([delta_u]).to(self.device))

            embedding_loca_acti_norm = self.layer_norm(torch.cat([embedding[idx_prev, :], self.embedding_kg[idx_know, :]], dim=1))
            
            embedding_user_loca_meta = torch.cat([X_proj,
                                                  embedding_loca_acti_norm, 
                                                  self.embedding_static[idx_prev, :], 
                                                  self.embedding_static[idx_user, :]],
                                                  dim=1)
    
            embedding_predict = self.predict_embedding_loca(embedding_user_loca_meta)    
            
            loss += torch.nn.MSELoss()(embedding_predict, 
                                       torch.cat([embedding[idx_loca, :], 
                                                  self.embedding_static[idx_loca, :]], dim=1).detach())
    
            euclidean_dist = torch.nn.PairwiseDistance()(embedding_predict.repeat(self.num_locas, 1), 
                                                         torch.cat([embedding[self.num_users:, :], 
                                                                    self.embedding_static[self.num_users:, :]], dim=1).detach())
            #print(torch.argmin(euclidean_dist).item(), idx_loca[0])
            if torch.argmin(euclidean_dist).item() == idx_loca[0] - self.num_users:
                top1 += 1
            
            val, index = torch.topk(euclidean_dist, 5, largest=False)
            if idx_loca[0] - self.num_users in list(index ):
                top5 += 1

            val, index = torch.topk(euclidean_dist, 10, largest=False)
            if idx_loca[0] - self.num_users in list(index):
                top10 += 1

            val, index = torch.topk(euclidean_dist, 20 , largest=False)
            if idx_loca[0] - self.num_users in list(index):
                top20 += 1
             
            update_embedding_user = self.update_rnn_user(embedding[idx_user, :], 
                                                          embedding[idx_loca, :],
                                                          idx_know,
                                                          #self.positional_embedding[idx_spat, :],
                                                          #self.activity[idx_cate, :],
                                                          torch.Tensor([delta_u]).to(self.device))
            
            update_embedding_loca = self.update_rnn_loca(embedding[idx_loca, :], 
                                                          embedding[idx_user, :], 
                                                          idx_know,
                                                          #self.positional_embedding[idx_spat, :],
                                                          #self.activity[idx_cate, :],
                                                          torch.Tensor([delta_l]).to(self.device))

            loss = loss + torch.nn.MSELoss()(update_embedding_user, embedding[idx_user, :].detach())
            loss = loss + torch.nn.MSELoss()(update_embedding_loca, embedding[idx_loca, :].detach()) 

            embedding[idx_user, :] = update_embedding_user
            embedding[idx_loca, :] = update_embedding_loca            
            #break
        return embedding, loss, top1, top5, top10, top20, num_interaction
