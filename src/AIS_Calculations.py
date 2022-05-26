"""
Perform calculations on sorted, dynamic AIS data for bulk ships.
Calcs include distance, time interval, and yearly ship aggregations.
Input(s): ais_bulkers_mmsi_timestamp.parquet
Output(s): ais_bulkers_calcs.parquet, AIS_yearly_stats.csv
Runtime: 3 minutes? (longest block is 1m44)
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np

datapath = 'data'
filepath = os.path.join(datapath, 'AIS')

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_mmsi_timestamp'))
ais_bulkers.dtypes

# #%% Check if all observations survived set_index and sort
# original_obs = ais_bulkers.count().compute()
# new_obs = dd.read_parquet(os.path.join(filepath, 'ais_bulkers')).count().compute()
# if any(original_obs != new_obs[new_obs.index != 'mmsi']):
#     print('Observations lost in index and sort operations!')

# %% Which message types have draught data?
# ais_bulkers.groupby(['msg_type']).count().compute()
# Draught only occurs in msg_type 5

#%% Retain only dynamic messages
ais_bulkers = ais_bulkers.loc[ais_bulkers['msg_type'].isin([1,2,3,27])]
ais_bulkers = ais_bulkers.drop(['msg_type', 'draught'], axis = 'columns')

#%%
# Time and distance steps between observations
# --------------------------------------------
#%% One step difference: map.partition of grouped differencing haversine
def pd_diff_haversine(df):
    df_diff = df.groupby('mmsi').shift(1)
    timediff = (df.timestamp - df_diff.timestamp)/np.timedelta64(1, 'h')
    lat = np.radians(df.latitude)
    lng = np.radians(df.longitude)
    lat_lag = np.radians(df_diff.latitude)
    lng_lag = np.radians(df_diff.longitude)
    lat_diff = lat - lat_lag
    lng_diff = lng - lng_lag
    d = (np.sin(lat_diff * 0.5) ** 2
         + np.cos(lat_lag) * np.cos(lat) * np.sin(lng_diff * 0.5) ** 2)
    dist = 2 * 6371.0088 * 0.539956803 * np.arcsin(np.sqrt(d))
    impliedspeed = dist / timediff
    return df.assign(distance = dist,
                     time_interval = timediff,
                     implied_speed = impliedspeed)

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['distance'] = 'float'
meta_dict['time_interval'] = 'float'
meta_dict['implied_speed'] = 'float'
ais_bulkers = ais_bulkers.map_partitions(pd_diff_haversine, meta = meta_dict)

#%% Compute and save
ais_bulkers.to_parquet(
        os.path.join(filepath, 'ais_bulkers_calcs'),
        append = False,
        overwrite = True,
        engine = 'fastparquet')
# 1m44s

#%%
# Aggregate statistics
# --------------------
#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
ais_bulkers.head()

#%%
ais_bulkers['year'] = ais_bulkers.timestamp.apply(
    lambda x: x.year,
    meta = ('x', 'int16'))

#%%
yearly_ISgt30 = (
    ais_bulkers.loc[ais_bulkers.implied_speed > 30]
    .groupby(['mmsi', 'year'])
    .implied_speed
    .count()
    .rename(('IS', 'gt30'))
    .compute()
)

#%%
yearly_TIgt12 = (
    ais_bulkers.loc[ais_bulkers.time_interval > 12]
    .groupby(['mmsi', 'year'])
    .time_interval
    .count()
    .rename(('TI', 'gt12'))
    .compute()
)

#%%
yearly_TIgt48 = (
    ais_bulkers.loc[ais_bulkers.time_interval > 48]
    .groupby(['mmsi', 'year'])
    .time_interval
    .count()
    .rename(('TI', 'gt48'))
    .compute()
)
#%%
any = dd.Aggregation(
    name='any',
    chunk=lambda s: s.any(),
    agg=lambda s0: s0.any()) 

#%%
yearly_stats_base = (
    ais_bulkers
    .groupby(['mmsi', 'year'])
    .agg({
        'timestamp': ['count', 'min', 'max'],
        'time_interval': ['mean', 'max'],
        'distance': ['mean', 'max', 'sum'],
        'implied_speed': ['mean', 'max'],
        'speed': ['mean', 'max']})
    .compute())

#%%
yearly_stats = (
    yearly_stats_base
    .merge(
        yearly_ISgt30,
        how = 'left',
        on = ['mmsi', 'year'])
    .merge(
        yearly_TIgt12,
        how = 'left',
        on = ['mmsi', 'year'])
    .merge(
        yearly_TIgt48,
        how = 'left',
        on = ['mmsi', 'year'])
)
#%%
yearly_stats_flat = yearly_stats
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_yearly_stats.csv'))


# #%%
# # Cleaned aggregate statistics
# # -----------------------------

# yearly_stats_clean.to_csv(os.path.join(datapath, 'AIS_yearly_stats_clean.csv'))
# # %%
# %%
