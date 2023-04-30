"""
Remove observations that clearly jump between ships (based on implied_speed).
Perform calculations on sorted, dynamic AIS data for bulk ships.
Calcs include distance, time interval, draught change, and yearly ship aggregations.
Input(s): ais_bulkers_merged.parquet
Output(s): ais_bulkers_calcs.parquet, AIS_yearly_stats.csv
Runtime: 4m? only at 60percent memory
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
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged'),
    columns = ['timestamp', 'latitude', 'longitude', 'speed', 'course', 'draught'])
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[0]

#%%
ais_bulkers = ais_bulkers.loc[ais_bulkers.speed < 25].dropna(subset = 'speed')


#%%
# Time and distance steps between observations
# --------------------------------------------
#%% Functions
# One step difference: map.partition of grouped differencing haversine
def pd_diff_haversine(df, groups):
    df_lag = df.groupby(groups).shift(1)
    timediff = (df.timestamp - df_lag.timestamp)/np.timedelta64(1, 'h')
    draughtdiff = (df.draught - df_lag.draught)
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
                     implied_course = impliedcourse,
                     draught_change = draughtdiff)

# Drop observations between small single point jumps
def pd_drop_skips(df):
    df['dir_diff'] = (df.implied_course - df.course) % 360
    df['dir_diff_min'] = np.minimum(df.dir_diff, 360 - df.dir_diff)

    df = (
        df
        .loc[~((df.dir_diff_min > 170) & (df.speed > 5)) &
            ~((df.dir_diff_min > 120) & (df.implied_speed > 25))]
        .drop(['dir_diff', 'dir_diff_min'], axis = 'columns'))
    return df

# Drop observations between large jumps
def pd_split_jumps(df):
    df['jump'] = df['implied_speed'].gt(25) & df['distance'].gt(140)
    df['path'] = (
        df['jump']
        .astype(int)
        .groupby('mmsi')
        .cumsum()
        .apply(lambda x: x%2 == 0))
        
    df = df.drop(['jump'], axis = 'columns')
    # df = (
    #     df
    #     .loc[df['first_path']]
    #     .drop(['first_path', 'jump'], axis = 'columns'))
    return df

def pd_select_path(df):
    df = (
        df
        .groupby(['mmsi', 'path'])
        .apply(lambda grp: grp.assign(obs = grp['timestamp'].count()))
        .groupby('mmsi')
        .apply(lambda grp: grp.assign(max_obs = grp['obs'].max()))
    )
    df = df.loc[df.obs == df.max_obs].drop(['max_obs', 'obs'], axis = 'columns')
    return df

# def pd_select_path1(df):
#     counts = (
#         df
#         .groupby(['mmsi', 'path'], group_keys = False)
#         .timestamp
#         .count()
#     )
#     maxids = counts.loc[counts.groupby('mmsi').idxmax()].rename('count')
#     df = (
#         df
#         .set_index('path', append = True)
#         .join(maxids, how='inner', on=['mmsi', 'path'])
#         .reset_index(level = ['path'])
#     )
#     return maxids, df

# Combine functions: Difference, drop jumps, then difference again
def pd_clean_diff_haversine(df):
    df = pd_diff_haversine(df, 'mmsi')
    df = pd_drop_skips(df)
    df = pd_diff_haversine(df, 'mmsi')
    df = pd_split_jumps(df)
    df = pd_select_path(df)
    df = pd_diff_haversine(df, 'mmsi')
    return df

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['distance'] = 'float'
meta_dict['time_interval'] = 'float'
meta_dict['implied_speed'] = 'float32'
meta_dict['implied_course'] = 'float32'
meta_dict['draught_change'] = 'float32'
meta_dict['path'] = 'bool'
ais_bulkers = ais_bulkers.map_partitions(pd_clean_diff_haversine, meta = meta_dict)
# ais_bulkers = ais_bulkers.map_partitions(pd_diff_haversine, 'mmsi', meta = meta_dict)
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
        os.path.join(filepath, 'ais_bulkers_calcs'),
        append = False,
        overwrite = True,
        engine = 'fastparquet')
# 1m44s diff_haversine
# 1m47s clean_diff_haversine

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
ais_bulkers['IS_distance'] = ais_bulkers['implied_speed'] * ais_bulkers['distance']
ais_bulkers['speed_distance'] = ais_bulkers['speed'] * ais_bulkers['distance']

#%%
# yearly_ISgt24 = (
#     ais_bulkers.loc[ais_bulkers.implied_speed > 24]
#     .groupby(['mmsi', 'year'])
#     .distance
#     .agg(['count', 'sum'])
#     .compute()
# )

#%%
# yearly_ISgt24 = yearly_ISgt24.rename(columns = {
#     'count': ('IS', 'gt24'),
#     'sum': ('IS', 'gt24Distance')})


#%%
# yearly_TIgt12 = (
#     ais_bulkers.loc[ais_bulkers.time_interval > 12]
#     .groupby(['mmsi', 'year'])
#     .time_interval
#     .count()
#     .rename(('TI', 'gt12'))
#     .compute()
# )

#%%
# yearly_TIgt48 = (
#     ais_bulkers.loc[ais_bulkers.time_interval > 48]
#     .groupby(['mmsi', 'year'])
#     .time_interval
#     .count()
#     .rename(('TI', 'gt48'))
#     .compute()
# )
#%%
any = dd.Aggregation(
    name='any',
    chunk=lambda s: s.any(),
    agg=lambda s0: s0.any()) 

#%%
# total_stats_base = (
#     ais_bulkers
#     .groupby(['mmsi', 'path'])
#     .agg({
#         'timestamp': ['count', 'min', 'max'],
#         'time_interval': ['mean', 'max'],
#         'distance': ['mean', 'max', 'sum'],
#         'implied_speed': ['mean', 'max'],
#         'speed': ['mean', 'max'],
#         'IS_distance': ['sum'],
#         'speed_distance': ['sum']})
#     ).compute()

# total_stats_flat = total_stats_base
# total_stats_flat.columns = ['_'.join(col) for col in total_stats_flat.columns.values]
# total_stats_flat.to_csv(os.path.join(datapath, 'AIS_total_stats.csv'))


#%%
yearly_stats_base = (
    ais_bulkers
    .groupby(['mmsi', 'year'])
    .agg({
        'timestamp': ['count', 'min', 'max'],
        'time_interval': ['mean', 'max'],
        'distance': ['mean', 'max', 'sum'],
        'implied_speed': ['mean', 'max'],
        'speed': ['mean', 'max'],
        'IS_distance': ['sum'],
        'speed_distance': ['sum']})
    ).compute()

#%%
# yearly_stats = (
#     yearly_stats_base
#     .merge(
#         yearly_ISgt24,
#         how = 'left',
#         on = ['mmsi', 'year'])
#     .merge(
#         yearly_TIgt12,
#         how = 'left',
#         on = ['mmsi', 'year'])
#     .merge(
#         yearly_TIgt48,
#         how = 'left',
#         on = ['mmsi', 'year'])
#     )


#%%
yearly_stats_flat = yearly_stats_base
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]

#%% Distance weighted average implied speed
yearly_stats_flat['weighted_IS_mean'] = yearly_stats_flat['IS_distance_sum'] / yearly_stats_flat['distance_sum']
yearly_stats_flat['weighted_speed_mean'] = yearly_stats_flat['speed_distance_sum'] / yearly_stats_flat['distance_sum']


#%%
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_yearly_stats.csv'))

#%%