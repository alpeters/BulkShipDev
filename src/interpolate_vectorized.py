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
import matplotlib.pyplot as plt

# %%
datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS')

ais_bulkers_pd = pd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs', 'part.0.parquet'))

#%%
start_time = time.time()
ais_bulkers_pd['rot'] = ais_bulkers_pd.groupby('mmsi').apply

ais_bulkers_pd['time_interval_shift'] = ais_bulkers_pd.groupby('mmsi').time_interval.shift(-1, fill_value=np.nan)

ais_bulkers_pd['latitude_shift'] = ais_bulkers_pd.groupby('mmsi').latitude.shift(-1, fill_value=np.nan)

ais_bulkers_pd['longitude_shift'] = ais_bulkers_pd.groupby('mmsi').longitude.shift(-1, fill_value=np.nan)

ais_bulkers_pd['latitude_rad'] = np.radians(ais_bulkers_pd['latitude'])
ais_bulkers_pd['longitude_rad'] = np.radians(ais_bulkers_pd['longitude'])
ais_bulkers_pd['latitude_shift_rad'] = np.radians(ais_bulkers_pd['latitude_shift'])
ais_bulkers_pd['longitude_shift_rad'] = np.radians(ais_bulkers_pd['longitude_shift'])
#%%
ais_bulkers_pd['col1'] = np.cos(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad)
#%%
ais_bulkers_pd['col2'] = np.sin(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad)
#%%
ais_bulkers_pd['col3'] = np.sin(ais_bulkers_pd.latitude_rad)

#%%
ais_bulkers_pd[['col1', 'col2', 'col3']].values

#%%
test = R.from_rotvec(ais_bulkers_pd[['col1', 'col2', 'col3']].values)

#%%


#%%
len(ais_bulkers_pd)

# #%% combine three length N series into an Nx3 array
# np.column_stack(
#     np.cos(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad), np.sin(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad), np.sin(ais_bulkers_pd.latitude_rad))

# #%%
# ais_bulkers_pd['rot'] = R.from_rotvec(np.column_stack((
#     np.cos(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad), np.sin(ais_bulkers_pd.longitude_rad) * np.cos(ais_bulkers_pd.latitude_rad), np.sin(ais_bulkers_pd.latitude_rad))))

#%%
ais_bulkers_pd['rot_shift'] = ais_bulkers_pd.groupby('mmsi').rot.shift(-1, fill_value=np.nan)


#%%
ais_bulkers_pd.head()

#%% return first element of rot column
ais_bulkers_pd.rot.iloc[0].as_rotvec()
#%% select first element of ais_bulkers_pd.rot
ais_bulkers_pd.rot.iloc[0]


#%%
ais_bulkers_pd.rotations.apply(lambda x: Slerp([0,1], R.from_rotvec(x)))


#%%
ais_bulkers_pd['slerp'] = Slerp([0, 1], R.from_rotvec(np.column_stack((ais_bulkers_pd.rot, ais_bulkers_pd.rot_shift))))


