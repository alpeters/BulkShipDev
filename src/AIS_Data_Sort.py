"""
Sort each partition by mmsi, timestamp
Input(s): ais_bulkers_indexed.parquet
Output(s): ais_bulkers_indexed_sorted.parquet
"""

import dask.dataframe as dd
import os, time

start = time.time()
filepath = './data/AIS'
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_indexed'))

ais_bulkers.map_partitions(
    lambda df : df.sort_values(['mmsi', 'created_at']),
    transform_divisions = False,
    align_dataframes = False,
    meta = ais_bulkers
).to_parquet(
    os.path.join(filepath, 'ais_bulkers_indexed_sorted'), 
    append = False, 
    overwrite = True)
end = time.time()
print("Elapsed time = ", end - start)
# 242s
# Could probably speed up if don't allow it to change partitions (it seems to have done so as there are fewer)
# Setting transform_divisions and align_dataframes did not change speed.
