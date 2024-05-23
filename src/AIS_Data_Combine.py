"""
Combine dynamic and static message by ship-hour.
Input(s): ais_bulkers_indexed_sorted.parquet
Output(s): ais_bulkers_merged.parquet
Runtime: 

TODO: Change name of IMO merged here
***May want to drop data if time difference is too large!
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
def fill_static(part):
    """
        Fill static data (draught, imo) into dynamic messages
    """
    part = part.loc[part['msg_type'].isin([1,2,3,27,5])].copy()
    part['draught'] = part.draught.replace(0, np.NaN) # treat zeros as missing values
    # part['hour'] = part['timestamp'].dt.floor('H')
    # part.set_index(['hour'], append = True, inplace = True)
    # part[['draught', 'imo']] = part.groupby(level=['mmsi', 'hour'])[['draught', 'imo']].transform('first') # drops data from static messages that don't have a dynamic message in same hour
    # part[['draught', 'imo']] = part.groupby(level=['mmsi', 'hour'])[['draught', 'imo']].fillna(method='ffill', limit=1) # too slow!!
    # part.reset_index(level='hour', drop=True, inplace=True)
    part[['draught', 'imo']] = part[['draught', 'imo']].groupby(level=['mmsi']).shift() # shifts msg_type 5 data to next message
    # part['time_diff'] = part.timestamp.diff().dt.total_seconds() # check if time difference is too large
    # part.loc[part['time_diff']>(24*60*60), ['draught', 'imo']] = np.nan # drop data if time difference is too large
    # part = part.loc[part['msg_type'] != 5, part.columns.drop('time_diff')]
    part = part.loc[part['msg_type'] != 5]
    return part


# ['timestamp', 'mmsi', 'msg_type', 'latitude', 'longitude', 'speed', 'course', 'heading', 'imo', 'name', 'draught', 'length', 'collection_type']

ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed_sorted'),
    columns = ['timestamp', 'msg_type', 'latitude', 'longitude', 'speed', 'course', 'draught', 'imo'])
# ais_bulkers = ais_bulkers.partitions[0:1]

# ais_bulkers.groupby('msg_type').count().compute() 
# 115002699 non-missing static messages

meta_dict = ais_bulkers.dtypes.to_dict()

with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .map_partitions(fill_static, meta=meta_dict)
        .to_parquet(
            os.path.join(filepath, 'ais_bulkers_merged'),
            append = False,
            overwrite = True,
            engine = 'fastparquet')
    )


#%% Check
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged'),
    columns = ['timestamp', 'msg_type', 'latitude', 'longitude', 'speed', 'course', 'draught', 'imo'])
ais_bulkers.head()

# nobs = ais_bulkers.count().compute()
# Dropping time_diff over 24h:
# draught      104717744
# imo          105165578
# Not dropping any:
# draught      104783429
# imo          105233596
# Not much difference.
# Losing around 10 million static observations, but these must be at end of sample.


# #########################
# # Summary stats and plots
# #########################
# #%% Calculate summary stats
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged'),
#     columns = ['timestamp', 'latitude', 'longitude', 'speed', 'course', 'draught', 'imo'])

# ais_bulkers['year'] = ais_bulkers.timestamp.apply(
#     lambda x: x.year,
#     meta = ('x', 'int16'))

# yearly_stats_base = (
#     ais_bulkers
#     .groupby(['mmsi', 'year'])
#     .agg({
#         'timestamp': ['count', 'min', 'max'],
#         'speed': ['mean', 'max'],
#         'draught': ['count', 'mean', 'max']})
#     ).compute()

# yearly_stats_flat = yearly_stats_base
# yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]

# yearly_stats_other = (
#     ais_bulkers
#     .loc[ais_bulkers.speed > 25, ['year', 'timestamp']]
#     .groupby(['mmsi', 'year'])
#     .agg(['count'])
#     .compute()
# )
# yearly_stats_other.columns = ['_'.join(col) + '_highspeed' for col in yearly_stats_other.columns.values]

# yearly_stats_flat = yearly_stats_flat.join(yearly_stats_other, how = 'outer')

# yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_raw_yearly_stats.csv'))

# ##### Plots
# #%% load yearly_stats.csv
# yearly_stats = pd.read_csv(os.path.join(datapath, 'AIS_raw_yearly_stats.csv'))

# #%%
# yearly_stats_flat['frac_highspeed'] = yearly_stats_flat['timestamp_count_highspeed'] / yearly_stats_flat['timestamp_count']

# #%% Overall
# yearly_stats_flat.describe(percentiles = [])

# #%% By year
# yearly_stats_flat.groupby('year').describe(percentiles = [])

# # %%
