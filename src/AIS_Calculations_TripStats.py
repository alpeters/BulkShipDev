"""
Calculate trip level stats
Input(s): ais_bulkers_trips.parquet
Output(s): AIS_trip_stats.csv
Runtime:
"""

#%%
import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = './src/data'

#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips'))
ais_bulkers.head()
# ais_bulkers = ais_bulkers.partitions[0]
#%%
ais_bulkers['IS_distance'] = ais_bulkers['implied_speed'] * ais_bulkers['distance']
ais_bulkers['speed_distance'] = ais_bulkers['speed'] * ais_bulkers['distance']


#%% https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))))

trip_stats = (
    ais_bulkers
    .groupby(['mmsi', 'trip'])
    .agg({
        'timestamp': ['first'],
        'distance': ['sum'],
        'time_interval': ['sum'],
        'draught': [nunique, 'mean'],
        'speed': ['mean'],
        'implied_speed': ['mean', 'max'],
        'IS_distance': ['sum'],
        'speed_distance': ['sum'],
        'origin': ['first'],
        'EU': ['max']
        }))

#%% https://stackoverflow.com/questions/63342589/dask-dataframe-groupby-most-frequent-value-of-column-in-aggregate
def chunk(s):
    return s.value_counts()

def agg(s):
    s = s._selected_obj
    return s.groupby(level=list(range(s.index.nlevels))).sum()


def finalize(s):
    level = list(range(s.index.nlevels - 1))
    return (
        s.groupby(level=level)
        .apply(lambda s: s.reset_index(level=level, drop=True).idxmax())
    )

mode = dd.Aggregation('mode', chunk, agg, finalize)

# Calculate draught_mode separately from other stats because of meta data error
draught_mode = (
    ais_bulkers
    .groupby(['mmsi', 'trip'])
    .agg({'draught': mode})
    .rename(columns = {'draught':'draught_mode'}))

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    trip_stats = trip_stats.compute()
    draught_mode = draught_mode.compute()

#%%
trip_stats_flat = trip_stats
trip_stats_flat.columns = ['_'.join(col) for col in trip_stats_flat.columns.values]

#%%
trip_stats_flat = trip_stats_flat.merge(
    draught_mode,
    how = 'left',
    left_on = ['mmsi', 'trip'],
    right_on = ['mmsi', 'trip'])

#%% Distance weighted average implied speed
trip_stats_flat['weighted_IS_mean'] = trip_stats_flat['IS_distance_sum'] / trip_stats_flat['distance_sum']
trip_stats_flat['weighted_speed_mean'] = trip_stats_flat['speed_distance_sum'] / trip_stats_flat['distance_sum']

#%% Destination using lead
trip_stats_flat = trip_stats_flat.rename(columns = {'origin_first': 'origin'})
trip_stats_flat['destination'] = trip_stats_flat.groupby('mmsi').origin.shift(-1)

#%%
trip_stats_flat.to_csv(os.path.join(datapath, 'AIS_trip_stats.csv'))

#%%