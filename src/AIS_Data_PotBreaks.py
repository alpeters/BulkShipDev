"""
Calculate differences between observations to identify potential breaks (change in ship identity)
Input(s): ais_bulkers_sorted_indexed.parquet
Output(s): ais_bulkers_intervals.parquet (only used here), ais_bulkers_potbreaks.csv
Runtime: 8m(ish)
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np
from dask.distributed import Client, LocalCluster
import dask.array as da

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed_sorted'),
    columns = ['timestamp', 'latitude', 'longitude', 'speed'])
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[3]

#%%
# ais_bulkers = ais_bulkers.loc[ais_bulkers.speed < 25].dropna(subset = 'speed')


#%%
# Time and distance steps between observations
# --------------------------------------------
#%% Functions
# One step difference: map.partition of grouped differencing haversine
def pd_diff_haversine(df, groups):
    df_lag = df.groupby(groups).shift(1)
    timediff = (df.timestamp - df_lag.timestamp)/np.timedelta64(1, 'h')
    lat = np.radians(df.latitude)
    lng = np.radians(df.longitude)
    lat_lag = np.radians(df_lag.latitude)
    lng_lag = np.radians(df_lag.longitude)
    lat_diff = lat - lat_lag
    lng_diff = lng - lng_lag  
    d = (np.sin(lat_diff * 0.5) ** 2
         + np.cos(lat_lag) * np.cos(lat) * np.sin(lng_diff * 0.5) ** 2)
    dist = 2 * 6371.0088 * 0.539956803 * np.arcsin(np.sqrt(d))
    impliedspeed = dist / timediff

    df_lead = df.groupby(groups).shift(-1)
    lat_lead = np.radians(df_lead.latitude)
    lng_lead = np.radians(df_lead.longitude)
    lng_diff = lng_lead - lng
    impliedcourse = (np.degrees(np.arctan2(
        np.cos(lat_lead)*np.sin(lng_diff),
        np.cos(lat)*np.sin(lat_lead) - np.sin(lat)*np.cos(lat_lead)*np.cos(lng_diff)))
        + 360) % 360

    return df.assign(distance = dist,
                     time_interval = timediff,
                     implied_speed = impliedspeed,
                     implied_course = impliedcourse)

# Drop observations between large jumps
def pd_split_jumps(df):
    df['jump'] = df['implied_speed'].gt(25) & df['distance'].gt(140)
    df['path'] = (
        df['jump']
        .astype(int)
        .groupby('mmsi')
        .cumsum())
        # .apply(lambda x: x%2 == 0))
        
    df = df.drop(['jump'], axis = 'columns')
    # df = (
    #     df
    #     .loc[df['first_path']]
    #     .drop(['first_path', 'jump'], axis = 'columns'))
    return df


# Combine functions: Difference, drop jumps, then difference again
def pd_clean_diff_haversine(df):
    df = pd_diff_haversine(df, 'mmsi')
    # df = pd_split_jumps(df)
    # df = pd_diff_haversine(df, 'mmsi')
    return df

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['distance'] = 'float'
meta_dict['time_interval'] = 'float'
meta_dict['implied_speed'] = 'float32'
meta_dict['implied_course'] = 'float32'

# meta_dict['path'] = 'int' #'bool'
# ais_bulkers = ais_bulkers.map_partitions(pd_clean_diff_haversine, meta = meta_dict)
ais_bulkers = ais_bulkers.map_partitions(pd_diff_haversine, 'mmsi', meta = meta_dict)
# ais_bulkers = ais_bulkers.partitions[0:1]

#%% Compute and save
with LocalCluster(
    n_workers=2,
    # processes=True,
    threads_per_worker=2
    # memory_limit='2GB',
    # ip='tcp://localhost:9895',
) as cluster, Client(cluster) as client:
    ais_bulkers.to_parquet(
        os.path.join(filepath, 'ais_bulkers_intervals'),
        append = False,
        engine = 'fastparquet')
# 1m44s diff_haversine
# 1m47s clean_diff_haversine

#%% Check
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_intervals'),
                              columns=['timestamp', 'speed', 'implied_speed', 'distance', 'imo', 'latitude', 'longitude'])
ais_bulkers.head()


#%% Create df of 
jumps = ais_bulkers.loc[ais_bulkers['implied_speed'].gt(25) & ais_bulkers['distance'].gt(140)].compute()
jumps['type'] = 'jump'
first = ais_bulkers.groupby('mmsi').first().compute()
first['type'] = 'first'
last = ais_bulkers.groupby('mmsi').last().compute()
last['type'] = 'last'

#%% bind rows of jumps, first, last
breaks = pd.concat([jumps, first, last], axis=0).reset_index().sort_values(['mmsi', 'timestamp'])
breaks.loc[breaks.mmsi != 200000000].head(30)
# %%
breaks.to_csv(os.path.join(datapath, 'ais_bulkers_potbreaks.csv'), index=False)