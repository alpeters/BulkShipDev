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
# ais_bulkers.get_partition(1).loc[205125000, ['timestamp', 'distance']].compute()
# ais_bulkers.get_partition(1).loc[205125000].compute()
ais_bulkers.columns

#%% Check if improbably high distances are due to a path change from previous row
# print(ais_bulkers.loc[205125000].head())
# ais_bulkers['path_change'] = ais_bulkers.path.astype(int).diff()
# ais_bulkers = ais_bulkers.loc[ais_bulkers['distance'] > 30]
# high_dist = ais_bulkers.loc[:, ['timestamp', 'distance', 'path', 'path_change']]
# ais_bulkers.path_change.value_counts().compute()
# No, almost all do not change path

#%% Count the number of each value of path
ais_bulkers['path'].value_counts().compute()

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



