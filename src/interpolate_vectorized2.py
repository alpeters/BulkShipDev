#%%
import os, time
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from datetime import timedelta
from scipy.spatial.transform import Rotation as R, Slerp
import pandas as pd
from dask.distributed import Client, LocalCluster
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# %%
datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS')

ais_bulkers_pd = pd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs', 'part.0.parquet'))

df = ais_bulkers_pd.loc[ais_bulkers_pd.index == 205041000]
#%%
start_time = time.time()

# ais_bulkers_pd['time_interval_shift'] = ais_bulkers_pd.groupby('mmsi').time_interval.shift(-1, fill_value=np.nan)

ais_bulkers_pd['latitude_rad'] = np.radians(ais_bulkers_pd['latitude'])
ais_bulkers_pd['longitude_rad'] = np.radians(ais_bulkers_pd['longitude'])

#%%
ais_bulkers_pd['col1'] = np.cos(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad)
#%%
ais_bulkers_pd['col2'] = np.sin(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad)
#%%
ais_bulkers_pd['col3'] = np.sin(ais_bulkers_pd.latitude_rad)

#%%
rotation = Rotation.from_rotvec(ais_bulkers_pd[['col1', 'col2', 'col3']].values)

#%%
slerp = Slerp(np.arange(0, len(rotation)), rotation)

#########################

steps = int(round(ais_bulkers_pd.groupby('mmsi').time_interval[1:]))

#%% Convert steps to integers
steps = steps.astype(int)


#%%
interp_vec = np.arange(0, len(rotation))
#########################
#%%
slerp_vec = slerp(interp_vec).as_rotvec()

#%%
ais_bulkers_pd['lon_interp'] = np.degrees(np.arctan2(slerp_vec[:, 1], slerp_vec[:, 0]))

#%%
ais_bulkers_pd['lat_interp'] = np.degrees(
    np.arctan2(
        slerp_vec[:, 2],
        np.sqrt(slerp_vec[:, 0]**2 + slerp_vec[:, 1]**2)))



