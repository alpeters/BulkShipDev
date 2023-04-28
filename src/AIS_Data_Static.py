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
# ais_bulkers = ais_bulkers.partitions[0]

#%%
ais_notbad_ids = pd.read_csv(os.path.join(datapath, 'ais_notbad_ids.csv'),
    usecols = ['mmsi', 'imo', 'name', 'length'],
    index_col = ['mmsi', 'imo', 'name', 'length'])

#%%
def filter_calc_static(df, filter_df):
    df = (
        df
        .loc[df['msg_type'] == 5]
        .drop('msg_type', axis = 'columns')
        )
    df['draught'] = df.draught.replace(0, np.NaN)
    df = (
        df
        .dropna(subset = 'draught')
        .set_index(['imo', 'name', 'length'], append = True)
        .join(filter_df, how='inner', on=['mmsi', 'imo', 'name', 'length'])
        .reset_index(level = ['imo', 'name', 'length'])
        )
    df['draught_max'] = df.groupby('mmsi').draught.quantile(0.99)
    df['laden'] = (df.draught / df.draught_max) > 0.75
    df['hour'] = df['timestamp'].dt.floor('H')
    df = df[['timestamp', 'imo', 'name', 'draught', 'length', 'laden', 'hour']]
    df = df.sort_values(['mmsi', 'hour'])
    return df

#%%
# filter_static(ais_bulkers.compute(), ais_notbad_ids)

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict.pop('msg_type')
meta_dict['laden'] = 'bool'
meta_dict['hour'] = 'datetime64[ns, UTC]'

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=2
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
            .map_partitions(filter_calc_static,
                ais_notbad_ids,
                meta = meta_dict,
                transform_divisions = False,
                align_dataframes = False)
            .to_parquet(
            os.path.join(filepath, 'ais_bulkers_static'),
            append = False,
            overwrite = True,
            engine = 'fastparquet')
    )

#%%
ais_bulkers_static = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_static'))
ais_bulkers_static.dtypes
ais_bulkers_static.head()
# ais_bulkers_static = ais_bulkers_static.partitions[0]

# #%% Check how NAs and non unique match in pandas
# df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K0', 'NaN', 'NaN'],
#                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']}).set_index('key')

# other = pd.DataFrame({'key': ['K0', 'NaN', 'NaN'],
#                       'B': ['B0', 'B1', 'B2']}).set_index('key')

# df.join(other, how = 'inner', on='key')
# # All NA match with all NA and nonunique get duplicated. Good.


#%% Explore draught quantiles

def draught_quantiles(df):
    df = (
        df
        .draught
        .replace(0, np.NaN)
        .dropna()
        .groupby('mmsi')
        .quantile(q = [0, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 1])
        )
    df.index.rename(['mmsi', 'quantile'], inplace = True)
    return df

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=2
) as cluster, Client(cluster) as client:
    draught_quantiles = ais_bulkers_static.map_partitions(
        draught_quantiles).compute()

draught_quantiles.to_csv(os.path.join(datapath, 'draught_quantiles.csv'))

# draught_quantiles.unstack()


# #%% Plot draught histograms for a random sample of ships
# import matplotlib.pyplot as plt
# import random

# mmsi = ais_bulkers_static.index.unique().compute()
# sample_mmsi = random.sample(list(mmsi), 20)

# sample = ais_bulkers_static.loc[sample_mmsi].replace(0, np.NaN).dropna().compute()
# sample['draught_max'] = sample.groupby('mmsi').draught.quantile(0.99)
# sample['draught_min'] = sample.groupby('mmsi').draught.quantile(0.01)
# sample['draught_norm2'] = (sample.draught - sample.draught_min) / (sample.draught_max - sample.draught_min)
# sample['draught_norm1'] = sample.draught / sample.draught_max
# sample.index.get_level_values(0).nunique()

# # sample.reset_index().pivot(columns='mmsi', values='draught_norm2').plot(kind='hist', subplots=True, rwidth=1, bins = 44, align='mid', range = [0, 1.1], legend = None)
# sample.reset_index().pivot(columns='mmsi', values='draught_norm1').plot(kind='hist', subplots=True, rwidth=1, bins = 44, align='mid', range = [0, 1.1], legend = None)
# plt.show()
