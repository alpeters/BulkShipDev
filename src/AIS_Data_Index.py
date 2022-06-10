# Set index by mmsi so future operations can be partition-wise

import dask.dataframe as dd
import pandas as pd
import numpy as np
import os, time

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=1)

filepath = 'src/data/AIS'
# ais = pd.read_parquet(os.path.join(filepath, 'aisparquet/part.0.parquet'))
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers'))
# dd.compute(ais_bulkers.shape)
unique_mmsi = ais_bulkers['mmsi'].unique().compute()
unique_mmsi.sort_values(inplace = True, ignore_index = True)
breakpoints = list(np.linspace(unique_mmsi[0], unique_mmsi.iat[-1], 40).astype('int32'))
# Need to set divisions because automatic algorithm seems to give floats
breaks_to_keep = [0, 2, 4, 8, 9, 11, 12, 15, 18, 19, 24, 30, 39]
breakpoints = [ even_breakpoints[i] for i in breaks_to_keep]


# Set index so data is sorted by mmsi
print("Setting index to mmsi")
start = time.time()
print("Starting at: ", start)
(
ais_bulkers.set_index('mmsi', shuffle = 'disk', divisions = breakpoints)
.to_parquet(os.path.join(filepath, 'ais_bulkers_mmsi'),
            append = False,
            overwrite = True)
)
end = time.time()
print(f"Elapsed time: {(end - start)}")
# 495s, just barely worked with 15GB RAM + 25GB Swap