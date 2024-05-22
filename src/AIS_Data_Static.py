"""
Filter out explicitly bad ID matches from static AIS and calculate ship draught percentiles.
Input(s): ais_bulkers_indexed_sorted.parquet, ais_notbad_ids.csv
Output(s): ais_bulkers_static.parquet, draught_quantiles.csv
Runtime: 431s CPU time
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
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed_sorted'),
    columns = ['timestamp', 'msg_type', 'imo', 'name', 'draught', 'length'])
ais_bulkers.dtypes
# ais_bulkers = ais_bulkers.partitions[0:2]

#%%
ais_notbad_ids = pd.read_csv(os.path.join(datapath, 'ais_notbad_ids.csv'),
    usecols = ['mmsi', 'imo', 'name', 'length'],
    index_col = ['mmsi', 'imo', 'name', 'length'])
ais_corrected_imo = pd.read_csv(os.path.join(datapath, 'ais_corrected_imo.csv'),
    usecols = ['mmsi', 'imo', 'imo_corrected'],
    index_col = ['mmsi', 'imo'])

# #%%
# def filter_calc_static(df, filter_df):
#     df = (
#         df
#         .loc[df['msg_type'] == 5]
#         .drop('msg_type', axis = 'columns')
#         )
#     df['draught'] = df.draught.replace(0, np.NaN)
#     df = (
#         df
#         # .dropna(subset = 'draught') # Keep these, because might have IMO number information
#         # .set_index(['imo', 'name', 'length'], append = True)
#         # .join(filter_df, how='inner', on=['mmsi', 'imo', 'name', 'length'])
#         # .reset_index(level = ['imo', 'name', 'length'])
#         .set_index(['imo'], append = True)
#         .join(filter_df, how='left', on=['mmsi', 'imo'])
#         .reset_index(level = ['imo'])
#         )
#     # df['draught_max'] = df.groupby('mmsi').draught.quantile(0.99) # don't do any modifications for now
#     # df['laden'] = (df.draught / df.draught_max) > 0.75 # unused
#     df['hour'] = df['timestamp'].dt.floor('H')
#     df = df[['timestamp', 'imo', 'name', 'draught', 'length', 'imo_corrected', 'hour']]
#     df = df.sort_values(['mmsi', 'hour'])
#     return df


# #%%
# meta_dict = ais_bulkers.dtypes.to_dict()
# meta_dict.pop('msg_type')
# # meta_dict['laden'] = 'bool'
# meta_dict['imo_corrected'] = 'Int64'
# meta_dict['hour'] = 'datetime64[ns, UTC]'

# #%%
# with LocalCluster(
#     n_workers=2,
#     threads_per_worker=2
# ) as cluster, Client(cluster) as client:
#     (
#         ais_bulkers
#             .map_partitions(filter_calc_static,
#                 # ais_notbad_ids,
#                 ais_corrected_imo,
#                 meta = meta_dict,
#                 transform_divisions = False,
#                 align_dataframes = False)
#             .to_parquet(
#             os.path.join(filepath, 'ais_bulkers_static'),
#             append = False,
#             overwrite = True,
#             engine = 'fastparquet')
#     )

# # #%%
# ais_bulkers_static = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_static'))
# ais_bulkers_static.dtypes
# ais_bulkers_static.head()
# ais_bulkers_static = ais_bulkers_static.partitions[0]


# Check how messy IMO numbers are
## Quantify contiguity of IMO reports
#%%
def static_contig_imo(df, corrected_df):
    """
        1. Select only static messages
        2. Try to correct (truncate) invalid IMO numbers
        3. Drop any that are still invalid
        4. Count number of switches and define instances as:
            a contiguous set of a given IMO number,
            e.g. 1, 1, 1, 2, 1, 2 has two instances of each
    """
    df = (
        df
        .loc[df['msg_type'] == 5]
        .drop('msg_type', axis = 'columns')
        )
    df = (
        df
        .set_index(['imo'], append = True)
        .join(corrected_df, how='left', on=['mmsi', 'imo'])
        .reset_index(level = ['imo'])
        .dropna(subset = 'imo_corrected')
        )
    df['imo_instance'] = df.groupby('mmsi').imo_corrected.diff().ne(0).cumsum()
    df = df[['timestamp', 'imo', 'name', 'draught', 'length', 'imo_corrected', 'imo_instance']]
    df = df.sort_values(['mmsi', 'timestamp'])
    return df



#%%
# test = ais_bulkers.partitions[0].compute()
# testout = static_contig_imo(test, ais_corrected_imo)
# testout.head()
# meta_dict_test = testout.dtypes.to_dict()

#%%
# meta_dict = {
#     'mmsi': 'Int64',  # index
#     'imo_corrected': 'float64',
#     'imo_instance': 'Int64',
#     # 'n_obs': 'Int64'
# }

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict.pop('msg_type')
meta_dict['imo_corrected'] = 'int'
meta_dict['imo_instance'] = 'int'

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=2
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
            .map_partitions(
                static_contig_imo,
                ais_corrected_imo,
                meta = meta_dict,
                # meta = meta_dict_test,
                transform_divisions = False,
                align_dataframes = False
                )
            .to_parquet(
            os.path.join(filepath, 'ais_bulkers_static_contig'),
            append = False,
            overwrite = True,
            engine = 'fastparquet')
    )

#%%
ais_bulkers_static = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_static_contig'))
ais_bulkers_static.dtypes
ais_bulkers_static.head()

# Simple count
# n_obs = ais_bulkers_static.groupby(['mmsi', 'imo_corrected', 'imo_instance']).size().compute()
#%% Get start and end timestamp as well as number of observations in each instance
n_obs = (
    ais_bulkers_static
    .groupby(['mmsi', 'imo_corrected', 'imo_instance'])
    .agg(n_obs = ('timestamp', 'size'),
         first_obs = ('timestamp', 'min'),
         last_obs = ('timestamp', 'max'))
    .sort_values(['mmsi', 'first_obs'])
).compute()

n_obs.head(30)
n_obs.loc[n_obs.index.get_level_values('mmsi') != 200000000].head(30)

# Simple count. Variable will equal one if no flipflopping
# contig_obs = n_obs.groupby(['mmsi', 'imo_corrected']).size().rename('n_instances')
contig_obs = (
    n_obs
    .groupby(['mmsi', 'imo_corrected'])
    .agg(n_instances = ('n_obs', 'size'),
         first_obs = ('first_obs', 'min'),
         last_obs = ('last_obs', 'max'))
    .sort_values(['mmsi', 'first_obs'])
)

contig_obs.head(30)
contig_obs.loc[contig_obs.index.get_level_values('mmsi') != 200000000].head(30)


# How many are always the same IMO number?
contig_obs.groupby('mmsi').size().value_counts()
# 9895 have only one IMO number
# 3440 have two IMO numbers
# 1046 have three...

# How many don't flip flop?
contig_obs['n_instances'].groupby('mmsi').apply(lambda x: all(x == 1)).value_counts()
# 10391 do not have flip flopping IMO numbers, while 4329 do 

contig_obs.to_csv(os.path.join(datapath, 'contig_obs.csv'))