"""
Select unique IDs in AIS data for cleaning out incorrect ones.
Input(s): ais_bulkers_indexed_sorted.parquet
Output(s): ais_ids.csv
Runtime: 
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np
from dask.distributed import Client, LocalCluster


datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS/imo_match')

#%%
ais_bulkers = dd.read_parquet(
    os.path.join(filepath, 'ais_bulkers_indexed_sorted'),
    columns = ['msg_type', 'imo', 'name', 'length', 'draught'] 
    )
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[0:2]
ais_bulkers['name'] = ais_bulkers['name'].fillna('NA')

#%%
with LocalCluster(
    n_workers=1,
    threads_per_worker=2
) as cluster, Client(cluster) as client:
    ais_bulkers['draught'] = ais_bulkers['draught'].replace(0, np.NaN)
    ais_bulkers.loc[ais_bulkers['msg_type'] == 5].count().compute()

# timestamp          108416431
# msg_type           108416431
# latitude                   0
# longitude                  0
# speed                      0
# course                     0
# heading                    0
# imo                108416431
# name               108383843
# draught            108079127
# length             108416431
# collection_type    108416431

# How many MMSI are there and how many aren't unique?
#%% Look for MMSI with multiple names, IMO's
with LocalCluster(
    n_workers=2,
    threads_per_worker=2
) as cluster, Client(cluster) as client:
    ids = (
        ais_bulkers
        .loc[ais_bulkers['msg_type'] == 5]
        .drop('msg_type', axis='columns')
        .groupby(['mmsi', 'imo', 'length'], dropna=False)
        .name
        .value_counts()
        .compute()
    )

ids.to_csv(os.path.join(datapath, 'ais_ids.csv'))