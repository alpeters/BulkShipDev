
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
row1 = ais_bulkers_pd.iloc[0]
row2 = ais_bulkers_pd.iloc[1]

#%%
def dostuff(row):
    lat, lon = np.radians(row['latitude']), np.radians(row['longitude'])

    col1 = np.cos(lon) * np.cos(lat)
    col2 = np.sin(lon) * np.cos(lat)
    col3 = np.sin(lat)
    vec = np.array([col1,col2,col3])
    # rot = R.from_rotvec(np.array(vec)).as_rotvec()
    return vec
#%%
slerp = Slerp([0, 1], R.from_rotvec([dostuff(row1), dostuff(row2)]))

#%%
num_points = 3
step = 1/(num_points-1)
t_values = np.linspace(step, 1-step, num_points-2)
print(t_values)
#%%
rot_values = slerp(t_values)

#%%
rotvec_values = rot_values.as_rotvec()

#%%
lon_lat_values = np.degrees(np.column_stack((
    np.arctan2(rotvec_values[:, 1], rotvec_values[:, 0]),
    np.arctan2(rotvec_values[:, 2], np.sqrt(rotvec_values[:, 0]**2 + rotvec_values[:, 1]**2))
)))

#%%
time_diff = (row2['timestamp'] - row1['timestamp']).total_seconds() / 3600
timestamps = [row1['timestamp'] + timedelta(hours=t*time_diff) for t in t_values]

interpolated_rows = [{
    'latitude': lat,
    'longitude': lon,
    #'year': row1['year'],
    'timestamp': timestamp,
    'speed': row2['implied_speed'],
    'interpolated': True
} for (lon, lat), timestamp in zip(lon_lat_values, timestamps)]


# %%
