"""
Merge portcall locations (EU or not) to AIS trips 
to identify trips into and out of EU
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, AIS_..._EU_yearly_stats.csv
Runtime: 5m
"""

#%%
import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''
filename = 'portcalls_' + callvariant + '_EU'

#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_' + callvariant))
ais_bulkers.head()

# %%
portcalls = pd.read_csv(
    os.path.join(datapath, 'pot' + filename + '.csv'),
    usecols = ['mmsi', 'pot_trip', 'EU', 'ISO_3digit'],
    dtype = {'mmsi' : 'int32',
             'pot_trip': 'int16',
             'EU': 'int8',
             'ISO_3digit': 'str'}
    )
portcalls = (
    portcalls
    .set_index('mmsi')
    .sort_values(['mmsi', 'pot_trip'])
    )
portcalls['pot_in_port'] = True

#%% merge portcalls to AIS data
with LocalCluster(
    n_workers=1,
    threads_per_worker=4
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .merge(portcalls,
            how = 'left',
            on = ['mmsi', 'pot_trip', 'pot_in_port'])
        .to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'), 
            append = False, 
            overwrite = True)
    )
# 2m8s
#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'))
ais_bulkers.head()
# ais_bulkers = ais_bulkers.partitions[9]


#%% Calculate new trip numbers
ais_bulkers['in_port'] = ~ais_bulkers['EU'].isnull()

def update_trip(df):
    df['trip'] = df.groupby('mmsi').in_port.cumsum()
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['trip'] = 'int'

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .map_partitions(update_trip, meta = meta_dict)
        .rename(columns = {'ISO_3digit': 'origin'})
        .to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_trips'), 
            append = False, 
            overwrite = True,
            engine = 'pyarrow') # getting strange overflow error with fastparquet
    )
# 1m29s
#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips'))
ais_bulkers.head()

#%%
portcalls = (
    ais_bulkers
    .loc[ais_bulkers['in_port'] == True,
        ['trip', 'latitude', 'longitude', 'EU', 'origin']]
    .reset_index(drop = False)
    .compute())

portcalls.to_csv(os.path.join(datapath, filename + '.csv'))
#%%
# Add in trip 0 for each ship (these don't appear in portcalls because first portcall assigned to trip 1)
# Assume trip 0 was not from EU port (but may be to one)
trip0 = pd.DataFrame({'mmsi': portcalls.mmsi.unique(),
                      'trip': 0,
                      'EU': False,
                      'origin': np.NAN})
portcalls = pd.concat([portcalls, trip0])
portcalls.sort_values(by = ['mmsi', 'trip'], inplace = True)
# EU trips include travel from previous portcalls
portcalls['EU'] = portcalls.EU == 1 # assumes NA are not in EU
portcalls['prev'] = portcalls.groupby('mmsi').EU.shift(-1, fill_value = False)
EU_trips = portcalls[portcalls.EU | portcalls.prev]
EU_trips = EU_trips[['mmsi', 'trip']].set_index('mmsi')

#%%
# Filter dask dataframe to contain only these combinations
ais_bulkers_EU = ais_bulkers.merge(EU_trips,
    how = 'right',
    on = ['mmsi', 'trip'])

#%% Travel work
ais_bulkers_EU['work_IS'] = ais_bulkers_EU['implied_speed']**2 * ais_bulkers_EU['distance']
ais_bulkers_EU['work'] = ais_bulkers_EU['speed']**2 * ais_bulkers_EU['distance']

# Aggregate distance, etc. by year
#%%
ais_bulkers_EU['year'] = ais_bulkers_EU.timestamp.apply(
    lambda x: x.year,
    meta = ('x', 'int16'))

#%% https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),)

#%%
yearly_stats = (
    ais_bulkers_EU
    .groupby(['mmsi', 'year'])
    .agg({
        'distance': ['sum'],
        'work': ['sum'],
        'work_IS': ['sum'],
        'trip': nunique
        })
    .compute())

#%%
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))
# 43s

#%%