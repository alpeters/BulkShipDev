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

group = ais_bulkers_pd.loc[ais_bulkers_pd.index == 205041000].copy()


# def interpolate_missing_hours(group):
# Calculate slerp location interpolation function
lat_rad = np.radians(group['latitude'])
long_rad = np.radians(group['longitude'])
col1 = np.cos(long_rad) * np.cos(lat_rad)
col2 = np.sin(long_rad) * np.cos(lat_rad)
col3 = np.sin(lat_rad)
rotation = Rotation.from_rotvec(np.column_stack([col1,col2,col3]))
slerp = Slerp(np.arange(0, len(rotation)), rotation)

#%% Create row number column of type int
group['data_counter'] = np.arange(len(group))

#%% Calculate step size for timestamp interpolation
# group['interp_steps'] = round(group.time_interval).clip(lower=1)
group['timestamp_hour'] = group.timestamp.dt.floor('H')
group['interp_steps'] = group.timestamp_hour.diff().dt.total_seconds().div(3600).fillna(1)
#%%
group['interp_step'] = (group.time_interval.clip(lower=1)/group.interp_steps).shift(-1).fillna(1)

#%% Create interpolated rows
group.reset_index(inplace=True)
#%%
group.set_index('timestamp_hour', inplace=True)
#%%
group = group.resample('H').asfreq()
group.mmsi.ffill(inplace=True)
group.reset_index(inplace=True)
group.set_index('mmsi', inplace=True)
group.timestamp.ffill(inplace=True)
group.time_interval.bfill(inplace=True)
group.interp_step.ffill(inplace=True)
group['interp_steps'] = group.interp_steps.bfill().astype(int)

#%% Interpolate timestamps
group['interp_counter'] = (np.ceil((group.timestamp_hour - group.timestamp).dt.total_seconds() / 3600).astype(int))

#%% Check that non-interpolated timestamps won't be modified
# group['timestamp_test'] = group.timestamp + pd.to_timedelta(group.interp_step*group.interp_counter, unit='H')
# test = group.loc[~np.isnan(group.data_counter)]
# all(test.timestamp == test.timestamp_test)

#%% Check interpolated timestamp steps are correct
# group['timestamp_test_diff'] = group.timestamp_test.diff().dt.total_seconds().div(3600).shift(-1)
# test = group[['timestamp_test_diff', 'interp_step']].loc[~np.isnan(group.data_counter) & ~(group.interp_step == 1)]
# any(test.timestamp_test_diff - test.interp_step > 0.001)

#%%
group['timestamp'] = group.timestamp + pd.to_timedelta(group.interp_step*group.interp_counter, unit='H')

#%% Interpolate coordinates
group['interp_coord_index'] = group.data_counter.ffill() + group.interp_counter/group.interp_steps
slerp_vec = slerp(group.interp_coord_index).as_rotvec()
group['lat_interp'] = np.degrees(
    np.arctan2(
        slerp_vec[:, 2],
        np.sqrt(slerp_vec[:, 0]**2 + slerp_vec[:, 1]**2)))
group['lon_interp'] = np.degrees(np.arctan2(slerp_vec[:, 1], slerp_vec[:, 0]))


#%% Plot interpolated coordinates on original coordinates to check
plotdf = group.iloc[0:1000]
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf_interp = gpd.GeoDataFrame(plotdf, geometry=gpd.points_from_xy(plotdf.lon_interp, plotdf.lat_interp))
gdf_data = gpd.GeoDataFrame(plotdf, geometry=gpd.points_from_xy(plotdf.longitude, plotdf.latitude))
fig, ax = plt.subplots()
ax.set_xlim([gdf_interp.total_bounds[0]-10, gdf_interp.total_bounds[2]+10])
ax.set_ylim([gdf_interp.total_bounds[1]-10, gdf_interp.total_bounds[3]+10])
world.boundary.plot(ax=ax)
gdf_interp.plot(ax=ax, color='red', markersize=1)
gdf_data.plot(ax=ax, color='blue', markersize=0.5)
plt.show()


# %%
