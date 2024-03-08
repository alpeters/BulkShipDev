"""
Computes aggregate summary stats of interpolated AIS data
Input(s): ais_bulkers_interp.parquet
Output(s): ais_bulkers_interp_stats.csv
Runtime:
"""

#%%
import os, time
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from datetime import timedelta
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt

filepath = 'src/data/'
datapath = os.path.join(filepath, 'AIS')
#%%
cluster = LocalCluster(
    n_workers=2,
    threads_per_worker=3
)

client = Client(cluster)

#%%
# ais_bulkers = dd.read_parquet(os.path.join(datapath, 'ais_bulkers_interp')).get_partition(0)
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'ais_bulkers_interp'))
ais_bulkers.head()
ais_bulkers.columns
#%%
ais_bulkers['year'] = ais_bulkers.timestamp.dt.year

yearly_stats = (
    ais_bulkers
    .groupby(['mmsi', 'year', 'interpolated', 'phase'])
    .agg({
        'timestamp': ['size', 'count'],
        'time_interval': ['size', 'count', 'sum', 'mean', 'min', 'max'],
        'speed': ['size', 'count', 'mean', 'min', 'max'],
        'distance': ['size', 'count', 'sum', 'mean', 'min', 'max'],
        'draught': ['size', 'count'],
        'draught_interpolated': ['size', 'count', 'sum'],
        'interpolated': ['size', 'count', 'sum'],
        'phase': ['size', 'count']
        })).compute()

# Collapse multi columns
yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]

yearly_stats.to_csv(os.path.join(datapath, 'ais_bulkers_interp_stats.csv'))

client.close()
cluster.close()



