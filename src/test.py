#%%
import dask.dataframe as dd
import pandas as pd
import os
import geopandas as gpd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')
callvariant = 'speed' #'heading'

#%%
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs')).get_partition(0)
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp')).get_partition(0)
ais_bulkers.head()

