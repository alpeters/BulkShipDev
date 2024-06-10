"""
Sort each partition by imo, timestamp
Input(s): ais_bulkers_imo_indexed.parquet
Output(s): ais_bulkers_imo_indexed_sorted.parquet
Runtime:
    CC: 129s
"""

import dask.dataframe as dd
import os, time

start = time.time()
# filepath = './data/AIS'
# filepath = '/scratch/petersal/ShippingEmissions/src/data/AIS'
filepath = 'AIS'
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_merged_indexed'))

ais_bulkers.map_partitions(
    lambda df : df.sort_values(['imo', 'timestamp']),
    transform_divisions = False,
    align_dataframes = False,
    meta = ais_bulkers
).to_parquet(
    os.path.join(filepath, 'ais_bulkers_merged_indexed_sorted'))
end = time.time()
print("Elapsed time = ", end - start)
