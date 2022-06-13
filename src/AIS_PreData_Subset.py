"""
Subset AIS parquet file by ship type to have manageable files for indexing
Input(s): ais_raw.parquet
Output(s): ais_bulkers.parquet, ais_containerships.parquet
"""

import dask.dataframe as dd
import pandas as pd
import os, time

from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=8)

filepath = 'src/data/AIS'
# ais = pd.read_parquet(os.path.join(filepath, 'aisparquet/part.0.parquet'))
ais = dd.read_parquet(os.path.join(filepath, 'ais_raw'))

# Bulkers
bulkers_mmsi = pd.read_csv('src/data/bulkers_mmsi.csv')
ais[ais.mmsi.isin(list(bulkers_mmsi['MMSI']))].to_parquet(os.path.join(filepath, 'ais_bulkers'))

# Containerships
containerships_mmsi = pd.read_csv('src/data/containerships_mmsi.csv')
ais[ais.mmsi.isin(list(containerships_mmsi['MMSI']))].to_parquet(os.path.join(filepath, 'ais_containerships'))
