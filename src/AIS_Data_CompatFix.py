import dask.dataframe as dd
import pandas as pd
import numpy as np
import os

filepath = 'AIS'

ais = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_tz'))
ais = ais.partitions[np.invert(np.isin(np.arange(ais.npartitions), [1157, 1362, 1448, 1657, 1711, 1728]))]
ais.to_parquet(os.path.join(filepath, 'ais_bulkers_tz_fix'), overwrite = True)