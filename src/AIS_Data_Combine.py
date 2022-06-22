"""
Combine dynamic and static message by ship-hour.
Input(s): ais_bulkers_mmsi_timestamp.parquet
Output(s): ais_bulkers_merged.parquet
Runtime: 4m20s
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed_sorted'))
ais_bulkers.dtypes

#%%
def static_merge(df):
    df['hour'] = df['timestamp'].dt.floor('H')
    dynamic = (
        df.loc[df['msg_type'].isin([1,2,3,27])]
        .drop(['msg_type', 'imo', 'name', 'draught'], axis = 'columns')
    )
    static = (
        df.loc[df['msg_type'] == 5]
        .drop(['msg_type', 'latitude', 'longitude', 'speed', 'heading'], axis = 'columns')
        .rename(columns = {'timestamp':'ts_static'})
    )
    df = dynamic.merge(static, how='outer', on=['mmsi', 'hour'])
    df['timestamp'] = df['timestamp'].fillna(df['ts_static'])
    df = (
        df
        .drop(['hour', 'ts_static'], axis = 'columns')
        .sort_values(['mmsi', 'timestamp'])
    )
    df['draught'] = df.groupby('mmsi').draught.fillna(method = 'ffill')
    df = df.dropna(subset = ['latitude', 'longitude'])
    return df


#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict.pop('msg_type')

with LocalCluster(
    n_workers=1,
    threads_per_worker=1
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .map_partitions(static_merge, meta = meta_dict)
        .to_parquet(
            os.path.join(filepath, 'ais_bulkers_merged'),
            append = False,
            overwrite = True,
            engine = 'fastparquet')
    )

#%%
