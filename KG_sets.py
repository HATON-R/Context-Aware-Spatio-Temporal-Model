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
import collections
import os
import pickle
import numpy as np
import pickle as pkl
import torch
import pandas as pd
import models
import optimizers.regularizers as regularizers

#########################################################################
############################ Knowledge Graph ############################
#########################################################################

## Training, validation and test sets
def get_entity2id_relation2id(KG, entity2id, relation2id):
    entity = []
    relations = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            entity.append(temp[0])
            entity.append(temp[2])
            relations.append(temp[1])
    entity = list(set(entity))
    relations = list(set(relations))
    f.close()
    with open(entity2id,'w') as f2:
        for i in range(len(entity)):
            f2.write(entity[i] + ' ')
            f2.write(str(i))
            f2.write('\n')
        f2.close()
    with open(relation2id,'w') as f3:
        for j in range(len(relations)):
            f3.write(relations[j]+' ')
            f3.write(str(j))
            f3.write('\n')
        f3.close()

def produce_train_val_test(KG, entity2id, realtion2id, triple):
    h_r_t = []
    h = []
    r = []
    t = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])
    entity_category_dict = {}
    relation_category_dict = {}
    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0] : temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0] : temp1[1]})
    with open(triple, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + ' ')
            f2.write(str(relation_category_dict[r[i]]) + ' ')
            f2.write(str(entity_category_dict[t[i]]))
            f2.write('\n')

def get_train_val_test(triple, train_address, valid_address, test_address):
    h_r_t = []
    with open(triple) as f:
        for line in f.readlines():
            temp = line.split()
            h_r_t.append(temp)
    random.shuffle(h_r_t)
    train = int(len(h_r_t) * 0.9)
    valid = int(len(h_r_t) * 0.05)
    test = len(h_r_t) - train - valid
    with open(train_address, 'w') as f:
        for i in range(train):
            f.write(h_r_t[i][0] + '\t')
            f.write(h_r_t[i][1] + '\t')
            f.write(h_r_t[i][2])
            f.write('\n')
    f.close()
    with open(valid_address, 'w') as f:
        for i in range(valid):
            f.write(h_r_t[train + i][0] + '\t')
            f.write(h_r_t[train + i][1] + '\t')
            f.write(h_r_t[train + i][2])
            f.write('\n')
    f.close()
    with open(test_address, 'w') as f:
        for i in range(test):
            f.write(h_r_t[train + valid + i][0] + '\t')
            f.write(h_r_t[train + valid + i][1] + '\t')
            f.write(h_r_t[train + valid + i][2])
            f.write('\n')
    f.close()


## Dictionary
def get_idx(path):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        print(os.path.join(path, split))
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(int(lhs))
                entities.add(int(rhs))
                relations.add(int(rel))
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    return ent2idx, rel2idx

def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[int(lhs)], rel2idx[int(rel)], ent2idx[int(rhs)]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")

def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final

def process_dataset(path, dataset_name):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    data_path = "/home/rhaton/test/MetaMobility/knowledge_graph/newyork/Data_processed"
    dataset_path = os.path.join(data_path, dataset_name)
    ent2idx, rel2idx = get_idx(dataset_path)

    entity_idx = list(ent2idx.keys())
    relations_idx = list(rel2idx.keys())
    for i in range(len(entity_idx)):
        entity_idx[i] = int(entity_idx[i])
    for i in range(len(relations_idx)):
        relations_idx[i] = int(relations_idx[i])
    entiy_id_embeddings = np.array(entity_idx)
    relations_id_embeddings = np.array(relations_idx)
    # The index between UrbanKG id and embedding
    np.savetxt(path + "/relations_idx_embeddings.csv", relations_id_embeddings, encoding="utf-8", delimiter=",")
    np.savetxt(path + "/entity_idx_embedding.csv", entiy_id_embeddings, encoding="utf-8", delimiter=",")
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


## KG dataset object
"""Dataset class for loading and processing UrbanKG datasets."""
class KGDataset(object):
    """Knowledge Graph dataset class."""
    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = os.path.join(os.getcwd(), data_path)
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            #print(self.data_path, split + ".pickle")
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2
    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip
    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities


## Embeddings functions
def get_embeddings(args):
    DATA_PATH = 'Data_processed'
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    #print("Dataset path:", dataset_path)

    dataset = KGDataset(dataset_path, args.debug)
    #print("Dataset example:", dataset.get_examples(split="valid"))

    args.sizes = dataset.get_shape()
    #print("Dataset shape:", args.sizes)

    model = getattr(models, args.model)(args)
    print(model)

    # Uncomment to load a pre-trained model
    # model.load_state_dict(torch.load(os.path.join("../logs/XXX/", "model.pt")))

    entity_embeddings = model.entity.weight.detach().numpy()
    #print("Entity embeddings shape:", entity_embeddings.shape)

    return entity_embeddings, entity_embeddings, entity_embeddings

def process_embeddings(input_path, entity_embeddings, save_path, embedding_shape, entity_columns):
    data = pd.read_csv(input_path)
    entity_data = data[entity_columns].values

    processed_embeddings = np.zeros(embedding_shape)
    for i in range(processed_embeddings.shape[0]):
        processed_embeddings[i][:32] = entity_embeddings[int(entity_data[i][1])]
        if len(entity_columns) > 2:
            processed_embeddings[i][32] = int(entity_data[i][2])

    print(f"Processed embeddings saved to {save_path}")
    np.save(save_path, processed_embeddings)