"""
Set index by mmsi so future operations can be partition-wise
Input(s): ais_bulkers.parquet
Output(s): ais_bulkers_indexed.parquet
Runtime: 
	Local: 495s with minimum variables, just barely works with 15GB RAM + 25GB Swap
	CC: 16m45 with full variables
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import os, time

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=1)

filepath = '/scratch/petersal/ShippingEmissions/src/data/AIS'
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers'))
unique_mmsi = ais_bulkers['mmsi'].unique().compute()
unique_mmsi.sort_values(inplace = True, ignore_index = True)
breakpoints = list(np.linspace(unique_mmsi[0], unique_mmsi.iat[-1], 40).astype('int32'))
# Need to set divisions because automatic algorithm seems to give floats
breaks_to_keep = [0, 2, 4, 8, 9, 11, 12, 15, 18, 19, 24, 30, 39]
breakpoints = [breakpoints[i] for i in breaks_to_keep]


# Set index so data is sorted by mmsi
print("Setting index to mmsi")
start = time.time()
print("Starting at: ", start)
outpath = os.path.join(filepath, 'ais_bulkers_indexed')
if not os.path.exists(outpath):
	os.mkdir(outpath)
(
	ais_bulkers.set_index('mmsi', shuffle = 'disk', divisions = breakpoints)
	.to_parquet(os.path.join(filepath, 'ais_bulkers_indexed'),
            append = False,
            overwrite = True)
)
end = time.time()
print(f"Elapsed time: {(end - start)}")