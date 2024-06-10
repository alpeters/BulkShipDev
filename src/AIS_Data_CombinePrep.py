"""
Calculate differences between observations to identify potential breaks (change in ship identity)
Input(s): ais_bulkers_sorted_indexed.parquet, ais_corrected_imo.csv
Output(s): ais_bulkers_sepsegments.parquet
Runtime: ~4min
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
    columns = ['imo', 'timestamp', 'msg_type', 'latitude', 'longitude', 'speed', 'course', 'draught'])
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[0:4]

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
    # impliedcourse = (np.degrees(np.arctan2(
    #     np.cos(lat_lead)*np.sin(lng_diff),
    #     np.cos(lat)*np.sin(lat_lead) - np.sin(lat)*np.cos(lat_lead)*np.cos(lng_diff)))
    #     + 360) % 360

    return df.assign(distance = dist,
                    #  time_interval = timediff,
                     implied_speed = impliedspeed)
                    #  implied_course = impliedcourse)

#%%
def correct_imo(df, corrected_df):
    """
        Join corrected (valid truncated) IMO numbers
    """
    df = (
        df
        .set_index(['imo'], append = True)
        .join(corrected_df, how='left', on=['mmsi', 'imo'])
        .reset_index(level = ['imo'])
        .dropna(subset = 'imo_corrected')
        )
    df = df.sort_values(['mmsi', 'timestamp'])
    return df

# Apply functions to all partitions
def process_partitions(df):
    df_dynamic = df.loc[df['msg_type'] != 5]
    df_static = df.loc[df['msg_type'] == 5]
    df_dynamic = pd_diff_haversine(df_dynamic, 'mmsi')
    df_dynamic['segment'] = (df_dynamic['implied_speed'].gt(25) & df_dynamic['distance'].gt(140)).cumsum()
    df_dynamic.drop(columns = ['distance', 'implied_speed'], inplace = True)
    df_static = correct_imo(df_static, ais_corrected_imo)
    df_static['segment'] = np.nan
    df = pd.concat([df_dynamic, df_static], axis=0).sort_values(['mmsi', 'timestamp'])
    return df

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
# meta_dict['distance'] = 'float'
# meta_dict['time_interval'] = 'float'
# meta_dict['implied_speed'] = 'float32'
# meta_dict['implied_course'] = 'float32'
meta_dict['segment'] = 'int'
meta_dict['imo_corrected'] = 'int'

#%%
ais_corrected_imo = pd.read_csv(os.path.join(datapath, 'ais_corrected_imo.csv'),
    usecols = ['mmsi', 'imo', 'imo_corrected'],
    index_col = ['mmsi', 'imo'])

# meta_dict['path'] = 'int' #'bool'
# ais_bulkers = ais_bulkers.map_partitions(pd_clean_diff_haversine, meta = meta_dict)
ais_bulkers = ais_bulkers.map_partitions(process_partitions, meta = meta_dict)
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
        os.path.join(filepath, 'ais_bulkers_sepsegments'),
        append = False,
        engine = 'fastparquet')
# 1m44s diff_haversine
# 1m47s clean_diff_haversine

#%% Check
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_sepsegments')) #,
                            #   columns=['timestamp', 'speed', 'implied_speed', 'distance', 'latitude', 'longitude', 'imo_corrected', 'draught'])
ais_bulkers.head(30)[['msg_type', 'timestamp', 'segment', 'imo', 'imo_corrected']]
ais_bulkers.loc[205041000].head(60)[['msg_type', 'timestamp', 'segment', 'imo', 'imo_corrected']]
# ais_bulkers = ais_bulkers.partitions[0:2]


#%% Create df of potential breaks
# jumps = ais_bulkers.loc[ais_bulkers['implied_speed'].gt(25) & ais_bulkers['distance'].gt(140)].compute()
# jumps['type'] = 'jump'
# first = ais_bulkers.groupby('mmsi').first().compute()
# first['type'] = 'first'
# last = ais_bulkers.groupby('mmsi').last().compute()
# last['type'] = 'last'

# #%% bind rows of jumps, first, last
# breaks = pd.concat([jumps, first, last], axis=0).reset_index().sort_values(['mmsi', 'timestamp'])
# breaks.loc[breaks.mmsi != 200000000].head(30)
# # %%
# breaks.to_csv(os.path.join(datapath, 'ais_bulkers_potbreaks.csv'), index=False)