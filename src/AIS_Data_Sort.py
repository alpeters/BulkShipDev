"""
Sort each partition by mmsi, timestamp
Input(s): ais_bulkers_indexed.parquet
Output(s): ais_bulkers_indexed_sorted.parquet
Runtime:
    Local: 242s with minimum variables
    CC: 408s
"""

import dask.dataframe as dd
import os, time

start = time.time()
# filepath = './data/AIS'
# filepath = '/scratch/petersal/ShippingEmissions/src/data/AIS'
filepath = 'AIS'
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed'))

ais_bulkers.map_partitions(
    lambda df : df.sort_values(['mmsi', 'timestamp']),
    transform_divisions = False,
    align_dataframes = False,
    meta = ais_bulkers
).to_parquet(
    os.path.join(filepath, 'ais_bulkers_indexed_sorted'), 
    append = False, 
    overwrite = True)
end = time.time()
print("Elapsed time = ", end - start)
