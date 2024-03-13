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

def interpolate_missing_hours(group):
    # Calculate slerp location interpolation function
    lat_rad = np.radians(group['latitude'])
    long_rad = np.radians(group['longitude'])
    col1 = np.cos(long_rad) * np.cos(lat_rad)
    col2 = np.sin(long_rad) * np.cos(lat_rad)
    col3 = np.sin(lat_rad)
    rotation = R.from_rotvec(np.column_stack([col1,col2,col3]))
    slerp = Slerp(np.arange(0, len(rotation)), rotation)

    # Create row number column of type int
    group['data_counter'] = np.arange(len(group))

    # Calculate step size for timestamp interpolation
    # group['interp_steps'] = round(group.time_interval).clip(lower=1)
    group['timestamp_hour'] = group.timestamp.dt.floor('H')
    group['interp_steps'] = group.timestamp_hour.diff().dt.total_seconds().div(3600).fillna(1)
    group['interp_step'] = (group.time_interval.clip(lower=1)/group.interp_steps).shift(-1).fillna(1)

    # Create interpolated rows
    group.reset_index(inplace=True)
    group.set_index('timestamp_hour', inplace=True)
    group = group.resample('H').asfreq()
    group.mmsi.ffill(inplace=True)
    group.reset_index(inplace=True)
    group.set_index('mmsi', inplace=True)
    group.index = group.index.astype(int)
    group.timestamp.ffill(inplace=True)
    group.time_interval.bfill(inplace=True)
    group.interp_step.ffill(inplace=True)
    group['interp_steps'] = group.interp_steps.bfill().astype(int)
    group['path'] = group.path.astype(bool)
    # Interpolate timestamps
    group['interp_counter'] = (np.ceil((group.timestamp_hour - group.timestamp).dt.total_seconds() / 3600).astype(int))
    group['timestamp'] = group.timestamp + pd.to_timedelta(group.interp_step*group.interp_counter, unit='H')

    # Interpolate coordinates
    group['interp_coord_index'] = group.data_counter.ffill() + group.interp_counter/group.interp_steps
    slerp_vec = slerp(group.interp_coord_index).as_rotvec()
    group['latitude'] = np.degrees(
        np.arctan2(
            slerp_vec[:, 2],
            np.sqrt(slerp_vec[:, 0]**2 + slerp_vec[:, 1]**2)))
    group['longitude'] = np.degrees(np.arctan2(slerp_vec[:, 1], slerp_vec[:, 0]))
    group['interpolated'] = group.data_counter.isna()
    group = group.drop(columns=['data_counter', 'timestamp_hour', 'interp_steps', 'interp_step', 'interp_counter', 'interp_coord_index'])
    
    return group

########################
#%%
ais_bulkers_pd = pd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs', 'part.0.parquet'))

group = ais_bulkers_pd.loc[ais_bulkers_pd.index == 205041000].copy()
print(group.dtypes)
#%%
group_interpolated = interpolate_missing_hours(group)
print(group_interpolated.dtypes)
#%%
plot_range = np.arange(1000, 1110)
plotdf_interp = group_interpolated.iloc[plot_range]
plotdf = group.iloc[plot_range]

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf_interp = gpd.GeoDataFrame(plotdf, geometry=gpd.points_from_xy(plotdf_interp.longitude, plotdf_interp.latitude))
gdf_data = gpd.GeoDataFrame(plotdf, geometry=gpd.points_from_xy(plotdf.longitude, plotdf.latitude))
fig, ax = plt.subplots()
ax.set_xlim([gdf_interp.total_bounds[0]-5, gdf_interp.total_bounds[2]+5])
ax.set_ylim([gdf_interp.total_bounds[1]-5, gdf_interp.total_bounds[3]+5])
world.boundary.plot(ax=ax)
gdf_interp.plot(ax=ax, color='red', markersize=1)
gdf_data.plot(ax=ax, color='blue', markersize=0.5)
plt.show()

# %% Test on dataframe with multiple groups
start_time = time.time()
df_interpolated = ais_bulkers_pd.groupby('mmsi', group_keys=False).apply(interpolate_missing_hours)
print(f"Time to interpolate: {time.time() - start_time}")

#%% Test on multiple partitions
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_test'))

def process_partition(df):
    df = (
        df
        .groupby('mmsi', group_keys=False)
        .apply(interpolate_missing_hours)
    )
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['interpolated'] = 'bool'


start_time = time.time()
with LocalCluster(
    n_workers=2,
    # processes=True,
    threads_per_worker=3
    # memory_limit='2GB',
    # ip='tcp://localhost:9895',
) as cluster, Client(cluster) as client:
    ais_bulkers.map_partitions(process_partition, meta=meta_dict).to_parquet(
        os.path.join(filepath, 'ais_bulkers_interp'),
        append=False,
        overwrite=True,
        engine='fastparquet'
    )
print(f"Time: {time.time() - start_time}")

# Load ais_bulkers_interp
ais_bulkers_interp = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp'))
ais_bulkers_interp.head()
