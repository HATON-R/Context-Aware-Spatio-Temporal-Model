import skmob
import time
import numpy as np
import networkx as nx
import csv
import datetime
import torch 
import tqdm
import gzip
import shutil
import os.path as osp
import os
import pandas as pd
from sklearn.preprocessing import scale
import glob
from pathlib import Path
from torch_geometric.data import Dataset, download_url, Data
import re
from torch_scatter import scatter_add
from collections import defaultdict
from networkx.algorithms import isomorphism
import json
import warnings
warnings.filterwarnings('ignore')

class create_batch(Dataset):

    def __init__(self, root, interval, data_path, name_city, matching_path, kge_path, transform=None, pre_transform=None, pre_filter=None):
        self.interval = interval
        self.data_path = data_path
        self.name_city = name_city
        #self.matching_path = matching_path
        #self.kge_path = kge_path
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        """Utilisation des fichiers depuis le chemin fourni"""
        return self.data_path
        
    @property
    def processed_file_names(self):
        return [os.path.basename(f) for f in os.listdir(self.processed_dir) if f.startswith("data_") and f.endswith(".pt")]

    @property
    def kg_embedding(self):
        kge = torch.load(self.kge_path +"/model.pt")["entity.weight"]
        return kge

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        idx = 0

        print("Files saved at this path",osp.join(self.processed_dir, self.name_city+"/data_" + str(self.interval)))
        kge = torch.load(self.kge_path +"/model.pt")["entity.weight"]
        os.makedirs(osp.join(self.processed_dir, self.name_city+"/data_" + str(self.interval)), exist_ok=True)
        torch.save(kge, osp.join(self.processed_dir, self.name_city+"/data_" + str(self.interval) + "/kg_embedding.pt"))
        
        df = pd.read_csv(self.data_path)
        df_time = self.create_time(df)
        df_filter = self.filter_trajectory(df_time)
        df_cat = self.mapping_category(df_time, df_filter)
        df_user_loca_cate = self.mapping_user_loca_cate(df_cat)
        df_interval = self.columns_interval(df_user_loca_cate)
        df_delta = self.delta(df_interval)
        df_previous = self.previous(df_delta)
        #df_final = df_previous
        df_final = self.matching_GM_KG(df_previous)
        df_final.to_csv(osp.join(osp.join(self.processed_dir, self.name_city), self.name_city + ".csv"), index=False)
        #df_final["test"] = df_final.iloc[:, 0]
        
        # Create batch
        batch = [group.values.tolist() for _, group in df_final.groupby(str(self.interval) + "h_interval")]
        num_interaction = df_final.shape[0]
        num_users = len(df_final.user_id.unique())
        num_locas = len(df_final.location_id.unique())
        num_actis = len(df_final.categorie_id.unique())
        for i, interaction in tqdm.tqdm(enumerate(batch)):
            events = []
            for user, loca, time, delta_u, delta_l, _, _, _, _, _, _, lat, lon, _, prev_loca_id, cate, cate_id, prev_cate_id, know in interaction:
                events.append([user, loca, time, delta_u, delta_l, prev_loca_id, cate_id, prev_cate_id, know])
            events = np.array(events)
            events = torch.tensor(events)

            data = Data(events=events,
                        num_interaction=num_interaction,
                        events_col=["user_id", "location_id", "timestamp", "delta_u", "delta_l", "previous", "categorie_id", "previous_cat", "KG" ],
                        num_users=num_users,
                        num_locations=num_locas,
                        num_activities=num_actis
                       )
            
            torch.save(data, osp.join(self.processed_dir, self.name_city+"/data_" + str(self.interval) + "/data_" + str(idx) + ".pt"))
            idx += 1

    def create_time(self, dataframe):
        if "utcTimestamp" in dataframe.columns:
            dataframe['check-in_time'] = pd.to_datetime(dataframe['utcTimestamp'])
        else:
            dataframe['check-in_time'] = pd.to_datetime(dataframe['check-in_time'])
            
        dataframe['check-in_time'] = dataframe['check-in_time'].dt.strftime('%a %b %d %H:%M:%S %z %Y')
        
        if "user_id" in dataframe.columns:
            dataframe = dataframe.loc[:, ["user_id", "check-in_time", "latitude", "longitude", "location_id"]]
            dataframe = dataframe.sort_values(by=['user_id','check-in_time'])
            return dataframe
        else:
            dataframe = dataframe.loc[:, ["userId", "check-in_time", "latitude", "longitude", "venueId", "venueCategory", "venueCategoryId"]]
            dataframe = dataframe.sort_values(by=['userId','check-in_time'])
            return dataframe

    def filter_trajectory(self, df):
        data = {}
        venues = {}

        if len(df.columns) < 7:
            nb = 7 - len(df.columns)
            for i in range(nb):
                df[str(i)] = df.iloc[:,0]
        
        for uid, tim, _, _, pid, _, _ in df.itertuples(index=False):
            if uid not in data:
                data[uid] = [[pid, tim]]
            else:
                data[uid].append([pid, tim])
            if pid not in venues:
                venues[pid] = 1
            else:
                venues[pid] += 1
        
        trace_min = 10
        location_global_visit_min = 10
        
        uid_3 = [x for x in data if len(data[x]) > trace_min]
        pick3 = sorted([(x, len(data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        pid_3 = [x for x in venues if venues[x] > location_global_visit_min]
        pid_pic3 = sorted([(x, venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)
        
        from collections import Counter
        import time
        session_len_list = []
        hour_gap = 72
        session_max = 10
        min_gap = 10
        filter_short_session = 5
        sessions_count_min = 5
        data_filter = {}
        
        for u in pick3:
            uid = u[0]
            info = data[uid]
            topk = Counter([x[0] for x in info]).most_common()
            topk1 = [x[0] for x in topk if x[1] > 1]
            sessions = {}
            for i, record in enumerate(info):
                poi, tmd = record
                tid = int(time.mktime(time.strptime(tmd, "%a %b %d %H:%M:%S %z %Y")))
                sid = len(sessions)
                if poi not in pid_3 and poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) / 3600 > hour_gap or len(sessions[sid - 1]) > session_max:
                        sessions[sid] = [record]
                    elif (tid - last_tid) / 60 > min_gap:
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= sessions_count_min:
                data_filter[uid] = {
                    'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                    'sessions': sessions_filter, 'raw_sessions': sessions
                }
        user_filter3 = [x for x in data_filter if
                       data_filter[x]['sessions_count'] >= sessions_count_min]
        #print('User Filter ', len(user_filter3))
        
        uid_list = {}
        vid_list = {'unk': [0, -1]}
        vid_list_lookup = {}
        
        for u in user_filter3:
            sessions = data_filter[u]['sessions']
            if u not in uid_list:
                uid_list[u] = [len(uid_list), len(sessions)]
            for sid in sessions:
                poi = [p[0] for p in sessions[sid]]
                for p in poi:
                    if p not in vid_list:
                        vid_list_lookup[len(vid_list)] = p
                        vid_list[p] = [len(vid_list), 1]
                    else:
                        vid_list[p][1] += 1
        
        pid_loc_lat = {}
        for uid,  tim, lat, lon, pid, _, _ in df.itertuples(index=False):
            pid_loc_lat[pid] = [float(lon), float(lat)]
        
        vid_lookup = {}
        
        for vid in vid_list_lookup:
            pid = vid_list_lookup[vid]
            lon_lat = pid_loc_lat[pid]
            vid_lookup[vid] = lon_lat
    
        train_split = 0.7
        validation_split = 0.1
        data_neural = {}
        
        for u in uid_list:
            sessions = data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]
                sessions_id.append(sid)
                
            split_id = int(np.floor(train_split * len(sessions_id)))
            split_validation = int(np.floor(validation_split * len(sessions_id)))
            
            if split_validation == 0:
                split_validation = 1
            
            split_validation = split_id + split_validation
                
            train_id = sessions_id[:split_id]
            validation_id = sessions_id[split_id : split_validation]
            test_id = sessions_id[split_validation:]
            
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = self.entropy_spatial(sessions)
        
            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location)
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set
            test_unique = whole_location - train_location_set
            location_ratio = len(test_unique) / len(whole_location)
        
            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(pid_loc_lat[pid])
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat)
            center = np.mean(lon_lat, axis=0, keepdims=True)
            center = np.repeat(center, axis=0, repeats=len(lon_lat))
            rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0]
        
            data_neural[uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg, 'validation': validation_id}
    
        data_df = []
        for uid in data_filter:
            #print(uid)
            sessions = data_filter[uid]['sessions']    
            for session_id, records in sessions.items():
                for lid, time in records:
                    #print(lid)
                    #print(time)
                    if lid in pid_loc_lat:
                        lat, lon = pid_loc_lat[lid]
                        #print(lat, lon)
                        data_df.append([uid, lid, time, lat, lon])

        df_data = pd.DataFrame(data_df, columns=['user_id', 'location_id', 'timestamp', 'latitude', 'longitude'])
            
        return df_data
    
    def tid_list_48(self, tmd):
        tm = time.strptime(tmd, "%a %b %d %H:%M:%S %z %Y")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid
    
    def entropy_spatial(self, sessions):
        locations = {}
        days = sorted(sessions.keys())
        for d in days:
            session = sessions[d]
            for s in session:
                if s[0] not in locations:
                    locations[s[0]] = 1
                else:
                    locations[s[0]] += 1
        frequency = np.array([locations[loc] for loc in locations])
        frequency = frequency / np.sum(frequency)
        entropy = - np.sum(frequency * np.log(frequency))
        return entropy

    def mapping_category(self, df_before, df_after):
        cat_id_dict = {}

        for u, t, lat, lon, loc, cat, cat_id in df_before.itertuples(index=False):
            if loc not in cat_id_dict:
                cat_id_dict[loc] = [cat, cat_id]

        df_after['categorie'] = df_after['location_id'].map(lambda x: cat_id_dict.get(x, [None, None])[0])
        df_after['categorie_id'] = df_after['location_id'].map(lambda x: cat_id_dict.get(x, [None, None])[1])

        return df_after

    def mapping_user_loca_cate(self, df):
        new_id_user = {}
        new_id_loca = {}
        new_id_cat = {}
        user = 0
        loca = 0
        cat = 0
        
        for u, l, t, lat, lon, c, c_id in df.itertuples(index=False):
            if u not in new_id_user:
                new_id_user[u] = user
                user += 1
        
            if l not in new_id_loca:
                new_id_loca[l] = loca
                loca += 1
        
            if c not in new_id_cat:
                new_id_cat[c] = cat
                cat += 1

        df["user_id"] = df["user_id"].map(new_id_user)
        df["location_id"] = df["location_id"].map(new_id_loca)
        df["categorie_id"] = df["categorie"].map(new_id_cat)

        return df

    def columns_interval(self, df):
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["datetime"].astype(int) // 10**9
        df["3h_interval"] = df["timestamp"] // (3 * 60 * 60)
        df["6h_interval"] = df["timestamp"] // (6 * 60 * 60)
        df["12h_interval"] = df["timestamp"] // (12 * 60 * 60)
        df["24h_interval"] = df["timestamp"] // (24 * 60 * 60)
        df["48h_interval"] = df["timestamp"] // (48 * 60 * 60)
        df["72h_interval"] = df["timestamp"] // (72 * 60 * 60)
        return df.sort_values(by=["datetime"])

    def delta(self, df):
        df['delta_u'] = df.groupby('user_id')['datetime'].diff()
        df['delta_u'] = df['delta_u'].dt.total_seconds()
        df['delta_u'].fillna(df['timestamp'], inplace=True)
        df["delta_u"] = scale(df.delta_u.values + 1)

        df['delta_l'] = df.groupby('location_id')['datetime'].diff()
        df['delta_l'] = df['delta_l'].dt.total_seconds()
        df['delta_l'].fillna(df['timestamp'], inplace=True)
        df["delta_l"] = scale(df.delta_l.values + 1)
        return df

    def previous(self, df):
        df["previous"] = df.groupby("user_id")["location_id"].shift()
        df["previous"].fillna(df["location_id"], inplace=True)
        df["previous"] = df["previous"].astype(int)
        
        df["previous_cat"] = df.groupby("user_id")["categorie_id"].shift()
        df["previous_cat"].fillna(df["categorie_id"], inplace=True)
        df["previous_cat"] = df["previous_cat"].astype(int)

        return df[["user_id", "location_id", "timestamp", "delta_u", "delta_l", "3h_interval", "6h_interval", "12h_interval", "24h_interval", "48h_interval", "72h_interval", "latitude", "longitude", "datetime", "previous", "categorie", "categorie_id", "previous_cat"]].sort_values("timestamp")

    def matching_GM_KG(self, df):
                
        matching = pd.read_csv(self.matching_path, header=None, names=["location_id", "KG"], sep=" ")
        matching = matching[matching['location_id'].str.startswith('POI/')]
        matching['location_id'] = matching['location_id'].str.extract(r'POI/(\d+)').astype("int")

        df_merge = df.merge(matching[['location_id', 'KG']], on='location_id', how='left').sort_values("KG")
        df_merge['KG'] = df_merge['KG'].fillna(-1)
        df_merge['KG'] = df_merge['KG'].astype(int)
        
        return df_merge.sort_values("timestamp")