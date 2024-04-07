"""
Combine dynamic and static message by ship-hour.
Input(s): ais_bulkers_indexed_sorted.parquet, ais_bulkers_static.parquet
Output(s): ais_bulkers_merged.parquet
Runtime: 3m
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os, time
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')

# ['timestamp', 'mmsi', 'msg_type', 'latitude', 'longitude', 'speed', 'course', 'heading', 'imo', 'name', 'draught', 'length', 'collection_type']
#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed_sorted'),
    columns = ['timestamp', 'msg_type', 'latitude', 'longitude', 'speed', 'course'])
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[0]

#%%
ais_bulkers_static = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_static'),
    columns = ['draught', 'laden', 'hour'])
ais_bulkers_static.dtypes
# ais_bulkers_static = ais_bulkers_static.partitions[0]

ais_bulkers = ais_bulkers.loc[df['msg_type'].isin([1,2,3,27])]
ais_bulkers['hour'] = ais_bulkers['timestamp'].dt.floor('H')
ais_bulkers['course'] = ais_bulkers.course.replace(360, np.NaN)
ais_bulkers = ais_bulkers.merge(ais_bulkers_static, how='outer', on=['mmsi', 'hour'])

#%%
def static_merge(df, df_static):
    df['draught'] = df.groupby('mmsi').draught.fillna(method = 'ffill')
    df['laden'] = df.groupby('mmsi').laden.fillna(method = 'ffill')
    # # ais_bulkers['heading'] = ais_bulkers.heading.replace(511, np.NaN)
    df = df.drop(['msg_type', 'hour'], axis = 'columns').dropna(subset = ['timestamp'])
    return df

# #%%
# def static_merge(df):
#     df['hour'] = df['timestamp'].dt.floor('H')
#     dynamic = (
#         df.loc[df['msg_type'].isin([1,2,3,27])]
#         .drop(['imo', 'name', 'length', 'draught', 'collection_type'], axis = 'columns')
#     )
#     static = (
#         df.loc[df['msg_type'] == 5]
#         .drop(['msg_type','latitude', 'longitude',  'speed', 'heading', 'course', 'collection_type'], axis = 'columns')
#         .rename(columns = {'timestamp':'ts_static'})
#     )
#     df = dynamic.merge(static, how='outer', on=['mmsi', 'hour'])
#     df['timestamp'] = df['timestamp'].fillna(df['ts_static'])
#     df = (
#         df
#         .drop(['hour', 'ts_static'], axis = 'columns')
#         .sort_values(['mmsi', 'timestamp'])
#     )
#     # df['draught'] = df.draught.replace(0, np.NaN) # do earlier
#     # df['draught'] = df.groupby('mmsi').draught.fillna(method = 'ffill')
#     df = df.dropna(subset = ['latitude', 'longitude'])
#     df['heading'] = df.heading.replace(511, np.NaN)
#     return df


#%%
meta_dict = {
    'timestamp': 'datetime64[ns, UTC]',
    # 'msg_type': 'int8',
    'latitude': 'float32',
    'longitude': 'float32',
    'speed': 'float32',
    'course': 'float32',
    'draught': 'float32',
    'laden': 'float32'
    }

with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .map_partitions(static_merge, ais_bulkers_static, meta = meta_dict)
        .to_parquet(
            os.path.join(filepath, 'ais_bulkers_merged'),
            append = False,
            overwrite = True,
            engine = 'fastparquet')
    )

#########################
# Summary stats and plots
#########################
#%% Calculate summary stats
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged'),
    columns = ['timestamp', 'latitude', 'longitude', 'speed', 'course', 'draught'])

ais_bulkers['year'] = ais_bulkers.timestamp.apply(
    lambda x: x.year,
    meta = ('x', 'int16'))

yearly_stats_base = (
    ais_bulkers
    .groupby(['mmsi', 'year'])
    .agg({
        'timestamp': ['count', 'min', 'max'],
        'speed': ['mean', 'max'],
        'draught': ['count', 'mean', 'max']})
    ).compute()

yearly_stats_flat = yearly_stats_base
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]

yearly_stats_other = (
    ais_bulkers
    .loc[ais_bulkers.speed > 25, ['year', 'timestamp']]
    .groupby(['mmsi', 'year'])
    .agg(['count'])
    .compute()
)
yearly_stats_other.columns = ['_'.join(col) + '_highspeed' for col in yearly_stats_other.columns.values]

yearly_stats_flat = yearly_stats_flat.join(yearly_stats_other, how = 'outer')

yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_raw_yearly_stats.csv'))

##### Plots
#%% load yearly_stats.csv
yearly_stats = pd.read_csv(os.path.join(datapath, 'AIS_raw_yearly_stats.csv'))

#%%
yearly_stats_flat['frac_highspeed'] = yearly_stats_flat['timestamp_count_highspeed'] / yearly_stats_flat['timestamp_count']

#%% Overall
yearly_stats_flat.describe(percentiles = [])

#%% By year
yearly_stats_flat.groupby('year').describe(percentiles = [])

# %%
