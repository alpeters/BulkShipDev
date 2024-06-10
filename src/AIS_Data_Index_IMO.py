"""
Set index by imo so future operations can be partition-wise
Input(s): ais_bulkers_merged.parquet
Output(s): ais_bulkers_merged_indexed.parquet
Runtime: 
    CC: 207s
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import os, time

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=1)

# filepath = 'AIS'
# filepath = '/scratch/petersal/ShippingEmissions/src/data/AIS'
filepath = 'src/data/AIS'

ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged'))
# ais_bulkers = ais_bulkers.partitions[0:2]

unique_imo = ais_bulkers['imo'].unique().compute()
unique_imo = unique_imo.sort_values(ignore_index = True)
# Need to set divisions because automatic algorithm seems to give floats
breakpoints_idx = np.arange(0, len(unique_imo), 250)
breakpoints = list(unique_imo.loc[breakpoints_idx]) + list(unique_imo.tail(1))

# Set index so data is sorted by imo
print("Setting index to imo")
start = time.time()
print("Starting at: ", start)
outpath = os.path.join(filepath, 'ais_bulkers_merged_indexed')
if not os.path.exists(outpath):
	os.mkdir(outpath)
(
	ais_bulkers.set_index('imo', shuffle = 'disk', divisions = breakpoints)
	.to_parquet(os.path.join(filepath, 'ais_bulkers_merged_indexed'))
)
end = time.time()
print(f"Elapsed time: {(end - start)}")


#%% Check
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged_indexed'))
# ais_bulkers.index.value_counts().compute()