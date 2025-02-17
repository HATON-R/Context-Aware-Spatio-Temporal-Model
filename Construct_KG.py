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

#########################################################################
############################ Create relation ############################
#########################################################################

## Borough
dataframe1 = gpd.read_file('./Data/Administrative_data/Borough/Borough.shp')
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['BoroCode', 'BoroName', 'geometry']
borough_dataframe = dataframe1[seleceted_colums1]
borough_dataframe.to_csv('./Data processed/NYC/data/NYC_borough.csv')


## Area
dataframe2 = gpd.read_file('./Data/Administrative_data/Area/Area.shp')
dataframe2 = dataframe2.to_crs('EPSG:4326')
seleceted_colums2 = ['OBJECTID', 'zone', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 1]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 103]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 104]
area_dataframe.to_csv('./Data processed/NYC/data/NYC_area.csv')


## POI align to BOROUGH / AREA
poi_dataframe = pd.read_csv('./Data/newyork.csv')
poi_datanumpy = poi_dataframe[['longitude', 'latitude']].values
poi_borough_area_id = np.full((poi_datanumpy.shape[0], 2), 999)
for i in tqdm(range(poi_datanumpy.shape[0])):
    poi_point = Point(poi_datanumpy[i][1], poi_datanumpy[i][0])
    
    # BOROUGH
    for j in range(borough_dataframe.shape[0]):
        borough_polygon = borough_dataframe.iloc[j].geometry
        if borough_polygon.contains(poi_point):
            poi_borough_area_id[i][0] = borough_dataframe.iloc[j].BoroCode
            break
    # AREA
    for k in range(area_dataframe.shape[0]):
        area_polygon = area_dataframe.iloc[k].geometry
        if area_polygon.contains(poi_point):
            poi_borough_area_id[i][1] = area_dataframe.iloc[k].OBJECTID
            break
poi_dataframe[['borough_id', 'area_id']] = poi_borough_area_id
poi_dataframe = poi_dataframe[ (poi_dataframe['borough_id'] != 999)]
poi_dataframe = poi_dataframe[ (poi_dataframe['area_id'] != 999)]
poi_dataframe.to_csv('./Data processed/NYC/data/NYC_poi.csv')


#########################################################################
####################### Construct Knowledge Graph #######################
#########################################################################


## Load data
dataframe1 = gpd.read_file('./Data/Administrative_data/Borough/Borough.shp')
dataframe1 = dataframe1.to_crs('EPSG:4326')
seleceted_colums1 = ['BoroCode', 'BoroName', 'geometry']
borough_dataframe = dataframe1[seleceted_colums1]
dataframe2 = gpd.read_file('./Data/Administrative_data/Area/Area.shp')
dataframe2 = dataframe2.to_crs('EPSG:4326')
seleceted_colums2 = ['OBJECTID', 'zone', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]
## filter 1, 103, 104 area as they are very small
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 1]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 103]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 104]


## Relation 1 Borough Nearby Borough BNB
BNB = []
for i in tqdm(range(borough_dataframe.shape[0])):
    head_borough = borough_dataframe.iloc[i].geometry
    for j in range(borough_dataframe.shape[0]):
        tail_borough = borough_dataframe.iloc[j].geometry
        if head_borough.touches(tail_borough):
            BNB.append('Borough/' + str(borough_dataframe.iloc[i].BoroCode) + ' BNB ' + 'Borough/' + str(borough_dataframe.iloc[j].BoroCode))


## Relation 2 Area Nearby Area ANA
ANA = []
for i in tqdm(range(area_dataframe.shape[0])):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        if head_area.touches(tail_area):
            ANA.append('Area/' + str(area_dataframe.iloc[i].OBJECTID) + ' ANA ' + 'Area/' + str(area_dataframe.iloc[j].OBJECTID))


## Relation 3 POI Locates at Area PLA
PLA = []
poi_dataframe = pd.read_csv('./Data processed/NYC/data/NYC_poi.csv')
poi_datanumpy = np.array(poi_dataframe[[ "location_id", "borough_id", "area_id", "categorie"]])
for i in tqdm(range(poi_datanumpy.shape[0])):
    PLA.append('POI/' + str(poi_datanumpy[i][0]) + ' PLA ' + 'Area/' + str(poi_datanumpy[i][2]))


## Relation 4 POI Belongs to Borough PBB
PBB = []
for i in tqdm(range(poi_datanumpy.shape[0])):
    PBB.append('POI/' + str(poi_datanumpy[i][0]) + ' PBB ' + 'Borough/' + str(poi_datanumpy[i][1]))


## Relation 5 POI has POI Category PHPC
PHPC = []
for i in tqdm(range(poi_datanumpy.shape[0])):
    PHPC.append('POI/' + str(poi_datanumpy[i][0]) + ' PHPC ' + 'PC/' + str(poi_datanumpy[i][3]))


## Relation 6 Area Locates at Borough ALB
ALB = []
for i in tqdm(range(area_dataframe.shape[0])):
    area = area_dataframe.iloc[i].geometry
    for j in range(borough_dataframe.shape[0]):
        borough = borough_dataframe.iloc[j].geometry
        if area.within(borough) or area.intersects(borough):
            ALB.append('Area/' + str(area_dataframe.iloc[i].OBJECTID) + ' ALB ' + 'Borough/' + str(borough_dataframe.iloc[j].BoroCode))


##
PLA.extend(PBB)
PLA.extend(ALB)
PLA.extend(BNB)
PLA.extend(ANA)
PLA.extend(PHPC)
with open(r'./Data processed/NYC/UrbanKG_NYC.txt','w') as file:
    for i in range(len(PLA)):
        file.write(PLA[i])
        file.write('\n')
file.close()