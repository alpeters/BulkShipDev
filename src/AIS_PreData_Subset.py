"""
Subset AIS parquet file by ship type to have manageable files for indexing
Input(s): ais_raw.parquet
Output(s): ais_bulkers.parquet, ais_containerships.parquet
Runtime: about 10m
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import os, time

from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=8)

filepath = 'src/data/AIS'
# ais = pd.read_parquet(os.path.join(filepath, 'aisparquet/part.0.parquet'))
# ais = dd.read_parquet(os.path.join(filepath, 'ais_raw'))
ais = dd.read_parquet(os.path.join('/media/apeters/Extreme SSD', 'ais_raw'))

# Retrieve all messages from any MMSI ever associated with a bulker IMO number
# Bulkers
bulkers_imo = pd.read_csv('src/data/bulkers_imo.csv')
bulkers_mmsi = ais[ais['imo'].isin(list(bulkers_imo['IMO.Number']))].mmsi.unique().compute()
bulkers_mmsi.to_csv('src/data/bulkers_mmsi.csv', index=False)
ais[ais.mmsi.isin(list(bulkers_mmsi))].to_parquet(os.path.join(filepath, 'ais_bulkers'))

# test = dd.read_parquet(os.path.join(filepath, 'ais_bulkers'))
# 14720 MMSI selected

# Some weird incompatibility with python 3.10 on CC, so need to:
# 1. drop size zero partitions
# 2. remove timezone
ais = dd.read_parquet(os.path.join(filepath, 'ais_bulkers'))
ais = ais.partitions[np.invert(np.isin(np.arange(ais.npartitions), [1157, 1362, 1448, 1657, 1711, 1728]))]
ais['timestamp'] = ais['timestamp'].dt.tz_localize(None)
ais.to_parquet(os.path.join(filepath, 'ais_bulkers_tz'), overwrite = True)

# Containerships
# containerships_mmsi = pd.read_csv('src/data/containerships_mmsi.csv')
# ais[ais.mmsi.isin(list(containerships_mmsi['MMSI']))].to_parquet(os.path.join(filepath, 'ais_containerships'))
