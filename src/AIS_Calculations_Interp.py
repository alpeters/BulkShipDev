"""
Hourly interpolation and detect trip phases from cleaned dynamic AIS data.
Input(s): ais_bulkers_calcs.parquet
Output(s): ais_bulkers_interp.parquet
Runtime: 23m local
"""

running_on = 'local'  # 'local' or 'hpc'

#%%
import os, time
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from scipy.spatial.transform import Rotation as R, Slerp

#%% Functions

# %%

def interpolate_missing_hours(group):
    """ 
    Interpolate missing observations in the time series at roughly hourly intervals.

    Args:
        group (pd.DataFrame): group of dataframe (grouped by imo) with columns latitude, longitude, timestamp

    Returns:
        pd.DataFrame: input group with additional interpolated rows and column indicating whether the row is interpolated
    """
    group['interpolated'] = False

    # Create hourly timestamp column
    group['timestamp_hour'] = group.timestamp.dt.floor('H')
    # Remove duplicate hours (gives error on resample if not removed)
    group = group.loc[~group['timestamp_hour'].duplicated(keep='first')].copy()

    # Calculate slerp location interpolation function
    lat_rad = np.radians(group['latitude'])
    long_rad = np.radians(group['longitude'])
    col1 = np.cos(long_rad) * np.cos(lat_rad)
    col2 = np.sin(long_rad) * np.cos(lat_rad)
    col3 = np.sin(lat_rad)
    rotation = R.from_rotvec(np.column_stack([col1,col2,col3]))
    if len(rotation) < 2:
        return group.drop(columns=['timestamp_hour'])
    slerp = Slerp(np.arange(0, len(rotation)), rotation)

    # Create row number column of type int
    group['data_counter'] = np.arange(len(group))
    # Calculate step size for timestamp interpolation 
    group['interp_steps'] = group.timestamp_hour.diff().dt.total_seconds().div(3600).fillna(1)
    group['interp_step'] = (group.time_interval.clip(lower=1)/group.interp_steps).shift(-1).fillna(1)
    # Create interpolated rows
    group.reset_index(inplace=True)
    group.set_index('timestamp_hour', inplace=True)
    group = group.resample('H').asfreq()
    group.imo.ffill(inplace=True)
    # group.path.ffill(inplace=True)
    group.reset_index(inplace=True)
    group.set_index('imo', inplace=True)
    group.index = group.index.astype(int)
    group.timestamp.ffill(inplace=True)
    group.time_interval.bfill(inplace=True)
    group.interp_step.ffill(inplace=True)
    group['interp_steps'] = group.interp_steps.bfill().astype(int)
    # group['path'] = group.path.astype(bool)
    group['interpolated'] = group['interpolated'].astype(bool)
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
    group.loc[group['data_counter'].isna(), 'interpolated'] = True
    group = group.drop(columns=['data_counter', 'timestamp_hour', 'interp_steps', 'interp_step', 'interp_counter', 'interp_coord_index'])
    
    return group

def pd_diff_haversine(df):
    """
    Calculates the distance (haversine) and time interval with respect to the previous row
    
    Args:
        df (pd.DataFrame): input dataframe with columns latitude, longitude, timestamp

    Returns:
        pd.DataFrame: input dataframe with additional columns distance, time_interval
    """
    df_lag = df.shift(1)
    timediff = (df.timestamp - df_lag.timestamp)/np.timedelta64(1, 'h')
    haversine_formula = 2 * 6371.0088 * 0.539956803  # precompute constant

    lat_diff, lng_diff = np.radians(df.latitude - df_lag.latitude), np.radians(df.longitude - df_lag.longitude)
    d = (np.sin(lat_diff * 0.5) ** 2 + np.cos(np.radians(df_lag.latitude)) * np.cos(np.radians(df.latitude)) * np.sin(lng_diff * 0.5) ** 2)
    dist = haversine_formula * np.arcsin(np.sqrt(d))

    return df.assign(distance=dist, time_interval=timediff)


def impute_speed(df):
    """ Impute speed for interpolated observations based on distance/time_interval. """
    speed = df['distance'] / df['time_interval']
    df['speed'] = np.where(df['interpolated'], speed, df['speed'])
    return df


def infill_draught_partition(df):
    """ Forward fill draught and then backward fill in case the first row is missing."""
    df['draught_interpolated'] = df['draught'].isna()
    df['draught'] = df['draught'].ffill()
    df['draught'] = df['draught'].bfill()
    # df['draught'].bfill(inplace=True) if df['draught'].isna().iloc[0] else df['draught'].ffill(inplace=True) # this will lead to inconsistent filling strategy
    return df


def assign_phase(df):
    """ Assign phase to each observation based on speed and distance to coast. """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(buffered_coastline.crs)
    # gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')
    gdf = gpd.sjoin(buffered_coastline, gdf, how="right", predicate='contains')
    gdf['phase'] = pd.cut(gdf['speed'], [-np.inf, 3, 5, np.inf], right=True, include_lowest=True, labels=['Anchored', 'Manoeuvring', 'Sea'])
    gdf.loc[(gdf['phase'] == 'Manoeuvring') & (gdf['index_left'].isna()), 'phase'] = 'Sea'
    gdf = gdf.drop_duplicates('timestamp') # in case of overlapping geometry causing double matches
    return gdf[df.columns.tolist() + ['phase']]

def process_group(group):
    """ 
    Performs interpolation and phase assignment group-wise.
    
    Args:
        group (pd.DataFrame): group of ais dataframe (grouped by imo)

    Returns:
        pd.DataFrame: input group with additional interpolated rows and columns 'interpolated' and 'phase'
    """
    group = interpolate_missing_hours(group)
    group = infill_draught_partition(group)
    group = pd_diff_haversine(group)
    group = impute_speed(group)
    group = assign_phase(group)
    return group

def process_partition(df):
    print(f"Processing partition with first imo {df.index[0]}")
    # df = df.loc[205041000]
    print(f"Time: {time.time() - start_time}")
    df = (
        df
        .groupby('imo', group_keys=False)
        .apply(process_group)
    )
    return df

############### Load data and run ################

# %%
datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS')

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'),
                              columns = ['timestamp', 'latitude', 'longitude', 'speed', 'implied_speed', 'draught', 'distance', 'time_interval'])
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs')).get_partition(0)
# ais_bulkers = ais_bulkers.partitions[0:5]

#%%
# Load the buffered mapfile (can be done by python or use QGIS directly)
buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_split_buffered_0_8333_degrees_fixed.gpkg')).drop(columns=['featurecla'])

#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['interpolated'] = 'bool'
meta_dict['draught_interpolated'] = 'bool'
meta_dict['phase'] = 'string'

#%%
if running_on == 'hpc':
    # Create a SLURM cluster object
    cluster = SLURMCluster(
        account='def-kasahara-ab',
        cores=1,  # This matches --ntasks-per-node in the job script
        memory='100GB', # Total memory
        walltime='1:00:00'
        #job_extra=['module load proj/9.0.1',
                #'source ~/carbon/bin/activate']
    )
    cluster.scale(jobs=3) # This matches --nodes in the SLURM script
elif running_on == 'local':
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=3
    )


# Connect Dask to the cluster
client = Client(cluster)

# Check Dask dashboard 
# with open('dashboard_url.txt', 'w') as f:
#     f.write(client.dashboard_link)

start_time = time.time()
ais_bulkers.map_partitions(process_partition, meta=meta_dict).to_parquet(
    os.path.join(filepath, 'ais_bulkers_interp'),
    append=False,
    overwrite=True,
    engine='fastparquet'
    )
print(f"Time: {time.time() - start_time}")

# Shut down the cluster
client.close()
cluster.close()


# Check
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp'))
ais_bulkers.head()