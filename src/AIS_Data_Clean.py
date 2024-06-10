"""
Merge to fleet register on IMO number (inner join, keeping only imo's in both datasets).
Drop imo's with less than 10 observations.
Use nominal draught and speed to bound instantaneous values.
Fill draught.
Input(s): ais_bulkers_merged_indexed_sorted.parquet, bulkers_WFR.csv
Output(s): ais_bulkers_cleaned.parquet
Runtime: a few mins

TODO: fill NA draught
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da
import pyreadr # for reading RData files

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged_indexed_sorted')) #,
    # columns = ['timestamp', 'latitude', 'longitude', 'speed', 'course', 'draught', 'msg_type'])
# ais_bulkers.dtypes
# ais_bulkers.partitions[1].head()
# ais_bulkers = ais_bulkers.partitions[0:10]

wfr_bulkers = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))["bulkers_df"]
# wfr_bulkers.columns[wfr_bulkers.columns.str.contains('imo', case=False)]
wfr_bulkers = wfr_bulkers[['IMO.Number', 'Draught..m.', 'Speed..knots.']]
wfr_bulkers = wfr_bulkers.rename(columns={'IMO.Number': 'imo', 'Draught..m.': 'draught_wfr', 'Speed..knots.': 'speed_wfr'})
wfr_bulkers = wfr_bulkers.dropna(subset = ['imo'])
wfr_bulkers = wfr_bulkers.set_index('imo')
# wfr_bulkers.head()


#%% Observation counts to drop those with too few
# count number of observations per IMO
imo_counts = ais_bulkers.groupby('imo').size().compute()


#%%
def join_wfr(df, wfr_df, imo_counts):
    """
        Join nominal draught and speed and use to bound instantaneous values.
    """
    # Drop imo's with less than 10 observations, similar to IMO (p.54)
    df = df[df.index.isin(imo_counts[imo_counts >= 10].index)]
    # Keep only imo's that are in the wfr (inner join)
    df = df.join(wfr_df, how='inner', on=['imo'])
    # Bound speed
    df['speed_wfr'] = df['speed_wfr'].fillna(14.2) # number chosen as average of wfr speed ratings
    df['speed_bound'] = df['speed_wfr'] * 1.5 # follow IMO procedure of bounding by 1.5 times nominal speed (p.54)
    df['speed'] = df[['speed', 'speed_bound']].min(axis=1)
    df.drop(columns=['speed_bound'], inplace=True)
    # Fill draught and bound
    df['draught'] = df['draught'].groupby('imo').ffill()
    df['draught'] = df['draught'].groupby('imo').bfill()
    df['draught'] = df[['draught', 'draught_wfr']].min(axis=1, skipna=False) # bound by wfr draught as per IMO (p.54), if not NA

    df = df.sort_values(['imo', 'timestamp'])
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['draught_wfr'] = 'float'
meta_dict['speed_wfr'] = 'float'

ais_bulkers = ais_bulkers.map_partitions(
    join_wfr,
    wfr_bulkers,
    imo_counts,
    meta = meta_dict,
    align_dataframes = False)

with LocalCluster(
    n_workers=2,
    # processes=True,
    threads_per_worker=2
    # memory_limit='2GB',
    # ip='tcp://localhost:9895',
) as cluster, Client(cluster) as client:
    ais_bulkers.to_parquet(
        os.path.join(filepath, 'ais_bulkers_cleaned'),
        append = False,
        overwrite = True,
        engine = 'fastparquet')
    

# Check
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_cleaned'))
# ais_bulkers[['timestamp', 'draught', 'draught_wfr', 'speed', 'speed_wfr']].partitions[1].head(500) #5000).query('speed > 25')
ais_bulkers.head(60)
# ais_bulkers.partitions[1].head(60)
# ais_bulkers[ais_bulkers['speed_wfr'] != 14.2].head(30, npartitions = -1)
ais_bulkers[ais_bulkers['draught'].isna() & ~ais_bulkers['draught_wfr'].isna()].head(60, npartitions=-1)

n_imos = ais_bulkers.index.nunique().compute()
n_imos
# 11907 unique imo's left
# compared to 13139 in WFR (not all may have been active)