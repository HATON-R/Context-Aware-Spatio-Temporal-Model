from preprocessing import SanFranciscoDataset
from train_evaluate import train_evaluate
from model_metaGNN import METAGCNConv, loss_gnn

import torch
import argparse
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="sanfrancisco", help="Name of the town in lower case and without spaces")
    parser.add_argument("--hour", default=12, help="Time interval in hours")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train the model")
    parser.add_argument("--embedding_dim", default=128, type=int, help="Number of dimensions of the dynamic embedding")
    parser.add_argument("--name_motif", default="motifs_1.json", help="Name of json file with patterns")
    parser.add_argument("--h3", default=False, help="Use h_3 default False")

    args = parser.parse_args()

    path_motif = os.path.join(os.getcwd(), args.name_motif)
    
    if args.name_motif == "motifs_0.json":
        nb_motifs = os.path.getsize(path_motif)
    else:
        with open(path_motif, "r") as json_file:
            data = json.load(json_file)
        nb_motifs = len(data)

    path = os.path.join(os.getcwd(), args.city)
    interval = args.hour
    
    if args.city == "sanfrancisco":
        dataset = SanFranciscoDataset(root=path + "/data_" + str(interval) + "/", interval=interval, path=os.getcwd(), path_motif=path_motif, h_3=args.h3)
        nb_data = len(dataset)

    folder_path = "/home/rhaton/test/MetaMobility/" + args.city + "/data_" + str(interval) + "/processed"
    nb_data = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]) - 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = args.epochs
    embedding_dynamic_size = args.embedding_dim
    out_channels = args.embedding_dim
    
    epoch_loss_train, epoch_loss_val, epoch_loss_test = train_evaluate(path, embedding_dynamic_size, epoch, device, nb_data, interval, path_motif, nb_motifs, out_channels, args.city)


    """A = torch.load("/home/rhaton/MetaMobility/A_3.pt")
    super_edge_index = [A[i][0].coalesce().indices() for i in range(len(A))]#, 
                        #A[1][0].coalesce().indices(),
                        #A[3][0].coalesce().indices(),
                        #A[3][0].coalesce().indices()]
    num_context = len(super_edge_index)
    print(num_context)
    model = METAGCNConv(num_context, embedding_dynamic_size, 100, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.rand(17310, embedding_dynamic_size)"""

    """for epoch in range(10):
        opt.zero_grad()
        H = model(X, super_edge_index)
        l = loss_gnn(model, H.to("cpu"), super_edge_index, num_context)
        print("Epoch : {}, Loss : {}".format(epoch, l.item()))
        l.backward()
        opt.step()
        L.append(l.item())
        l = 0"""