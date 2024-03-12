"""
Hourly interpolation and detect trip phases from cleaned dynamic AIS data.
Input(s): ais_bulkers_calcs.parquet
Output(s): ais_bulkers_interp.parquet
Runtime:
"""

running_on = 'local'  # 'local' or 'hpc'

#%%
import os, time
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from datetime import timedelta
from dask.distributed import Client, LocalCluster
# from dask.distributed import Lock
from scipy.spatial.transform import Rotation as R, Slerp
import pandas as pd

#%% Functions

# %%

def interpolate_missing_hours(group):
    """ 
    Interpolate missing observations in the time series at roughly hourly intervals.

    Args:
        group (pd.DataFrame): group of dataframe (grouped by mmsi) with columns latitude, longitude, timestamp

    Returns:
        pd.DataFrame: input group with additional interpolated rows and column indicating whether the row is interpolated
    """
    group['interpolated'] = False
    
    # Calculate slerp location interpolation function
    lat_rad = np.radians(group['latitude'])
    long_rad = np.radians(group['longitude'])
    col1 = np.cos(long_rad) * np.cos(lat_rad)
    col2 = np.sin(long_rad) * np.cos(lat_rad)
    col3 = np.sin(lat_rad)
    rotation = R.from_rotvec(np.column_stack([col1,col2,col3]))
    if len(rotation) < 2:
        return group
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
    df['draught'] = df['draught'].ffill()
    df['draught'] = df['draught'].bfill()
    # df['draught'].bfill(inplace=True) if df['draught'].isna().iloc[0] else df['draught'].ffill(inplace=True) # this will lead to inconsistent filling strategy
    return df


def assign_phase_sjoin(df):
    """ Assign phase to each observation based on speed and distance to coast. """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(buffered_coastline.crs)
    # gdf = gpd.sjoin(buffered_coastline, gdf, how="right", predicate='contains')
    gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')
    # gdf.loc[(gdf['phase'] == 'Manoeuvring') & (gdf['fid'].isna()), 'phase'] = 'Sea'
    gdf['phase'] = pd.cut(gdf['speed'], [-np.inf, 3, 5, np.inf], right=True, include_lowest=True, labels=['Anchored', 'Manoeuvring', 'Sea'])
    gdf.loc[(gdf['phase'] == 'Manoeuvring') & (gdf['index_right'].isna()), 'phase'] = 'Sea'
    return gdf[df.columns.tolist() + ['phase']]

def assign_phase_within(df):
    """ Assign phase to each observation based on speed and distance to coast. """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(buffered_coastline.crs)
    points_within = pd.Series(False, index=gdf.index)
    for i in range(0, len(buffered_coastline['geometry'])):
        points_within = points_within | gdf.geometry.within(buffered_coastline['geometry'][i])
    gdf['within'] = points_within
    gdf['phase'] = pd.cut(gdf['speed'], [-np.inf, 3, 5, np.inf], right=True, include_lowest=True, labels=['Anchored', 'Manoeuvring', 'Sea'])
    gdf.loc[(gdf['phase'] == 'Manoeuvring') & (~gdf['within']), 'phase'] = 'Sea'
    return gdf[df.columns.tolist() + ['phase']]

def process_group(group):
    """ 
    Performs interpolation and phase assignment group-wise.
    
    Args:
        group (pd.DataFrame): group of ais dataframe (grouped by mmsi)

    Returns:
        pd.DataFrame: input group with additional interpolated rows and columns 'interpolated' and 'phase'
    """
    group = interpolate_missing_hours(group)
    group = infill_draught_partition(group)
    group = pd_diff_haversine(group)
    group = impute_speed(group)
    # group = assign_phase_sjoin(group)
    group = assign_phase_within(group)
    return group

def process_partition(df):
    print(f"Processing partition with first mmsi {df.index[0]}")
    print(f"Time: {time.time() - start_time}")
    df = (
        df
        .groupby('mmsi', group_keys=False)
        .apply(process_group)
    )
    return df

############### Load data and run ################

# %%
datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS')


#%% Create test file with first rows of each partition
# (
#     dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
#     # .partitions[0:8]
#     .map_partitions(lambda df: df.head(2000)).to_parquet(
#     os.path.join(filepath, 'ais_bulkers_calcs_testheads'),
#     append=False,
#     overwrite=True,
#     engine='fastparquet')
# )
#%%
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs')).partitions[0]
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_testheads'))
ais_bulkers.head()
#%%
# Load the buffered reprojected shapefile (can be done by python or use QGIS directly)
# buffered_coastline = gpd.read_file(os.path.join(datapath, 'buffered_reprojected_coastline', 'buffered_reprojected_coastline.shp'))
# .drop(columns=['featurecla', 'scalerank', 'min_zoom'])
# buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_buffered_0_1_degrees.gpkg')).drop(columns=['featurecla', 'scalerank', 'min_zoom'])
buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_split_buffered_0_8333_degrees_fixed.gpkg')).drop(columns=['featurecla'])
# buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_split_buffered_0_8333_degrees_unfixed.gpkg')).drop(columns=['featurecla'])
#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['interpolated'] = 'bool'
meta_dict['phase'] = 'string'
# meta_dict['within'] = 'bool'
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
    # This matches --nodes in the SLURM script
    cluster.scale(jobs=3) 
elif running_on == 'local':
    cluster = LocalCluster(
        n_workers=2,
        # processes=True,
        threads_per_worker=3
        # memory_limit='2GB',
        # ip='tcp://localhost:9895',
    )


# Connect Dask to the cluster
client = Client(cluster)

# Check Dask dashboard 
# with open('dashboard_url.txt', 'w') as f:
#     f.write(client.dashboard_link)

# suffix = '_nodistcriteria'
# suffix = '_sjoin_fixed'
suffix = '_within_fixed'
# suffix=''


np.random.seed(3209)
partitions = np.random.choice(ais_bulkers.npartitions, 6, replace=False)
partitions = list(np.sort(partitions))
print(partitions)

start_time = time.time()
ais_bulkers.partitions[partitions].map_partitions(process_partition, meta=meta_dict).to_parquet(
    # os.path.join(filepath, 'ais_bulkers_interp'),
    os.path.join(filepath, 'ais_bulkers_interp'+suffix),
    append=False,
    overwrite=True,
    engine='fastparquet'
    )
print(f"Time: {time.time() - start_time}")

# Shut down the cluster
client.close()
cluster.close()

#%% Check results
# Load ais_bulkers_interp
import dask.dataframe as dd
import geopandas as gpd
import matplotlib.pyplot as plt
datapath = 'src/data/'
filepath = os.path.join(datapath, 'AIS')
suffix = '_nodistcriteria'
buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_split_buffered_0_8333_degrees_fixed.gpkg')).drop(columns=['featurecla'])
ais_bulkers_interp = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp'+suffix))
ais_bulkers_interp.head()
test_day = ais_bulkers_interp.loc[205041000].compute()
test_day = test_day.loc[test_day.speed.gt(3) & test_day.speed.lt(5)]
# extract date part of timestamp and compare with a given day
# test_day = test_day[test_day['timestamp'].dt.date == pd.to_datetime('2021-11-17').date()]
# test_day = test_day[test_day['timestamp'].dt.date == pd.to_datetime('2020-12-11').date()]
# test_day = test_day.loc[(test_day['timestamp'].dt.hour < 8)]
# test_day = ais_bulkers_interp.compute()
# print(test_day[['timestamp', 'speed', 'phase', 'within', 'latitude', 'longitude']].head(10))
print(test_day[['timestamp', 'speed', 'phase', 'latitude', 'longitude']].head(10))

test_day_gdf = gpd.GeoDataFrame(test_day, geometry=gpd.points_from_xy(test_day.longitude, test_day.latitude), crs="EPSG:4326")
test_day_gdf = test_day_gdf.to_crs(buffered_coastline.crs)
test_day_gdf.to_file(os.path.join(datapath, 'test_day'+suffix+'.gpkg'), driver='GPKG')
# 4am should be Sea and 6am should be Manoeuvring
# %% Plot test_day_gdf on top of buffered_coastline
fig, ax = plt.subplots()
buffered_coastline.plot(ax=ax)
ax.set_ylim([test_day_gdf['latitude'].min(), test_day_gdf['latitude'].max()])
ax.set_xlim([test_day_gdf['longitude'].min(), test_day_gdf['longitude'].max()])
# test_day_gdf.plot(ax=ax, column='speed', legend=True)
test_day_gdf.plot(ax=ax, column='phase', legend=True)
plt.show()

# %% Print partition lengths
# (
#     dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
#     .map_partitions(lambda df: print(len(df)))
#     .compute()
# )

# %% Compare output files
first = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp_sjoin_fixed'))
second = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp_within_fixed'))
#%% loop through partitions of first and second and compare the phase columns
for i in range(0, first.npartitions):
    print(i)
    first_part = first.partitions[i].compute()
    second_part = second.partitions[i].compute()
    compare = first_part.merge(second_part, on='timestamp', suffixes=('_first', '_second'))
    print(compare.loc[compare.phase_first != compare.phase_second])
                               
                     




# %%
# # #%%
# # Test the assign_phase function on just the very first group of the first partition
# # Create a dataframe with the first group of the first partition
# group = ais_bulkers.get_partition(0)
# group = group.loc[205041000]
# # Load the buffered mapfile (can be done by python or use QGIS directly)
# buffered_coastline = gpd.read_file(os.path.join(datapath, 'land_split_buffered_0_8333_degrees_fixed.gpkg')).drop(columns=['featurecla'])
# #%% Test each function in the process_group function
# def test_fun(group):
#     group = interpolate_missing_hours(group)
#     group = infill_draught_partition(group)
#     group = pd_diff_haversine(group)
#     group = impute_speed(group)
#     gdf = gpd.GeoDataFrame(group, geometry=gpd.points_from_xy(group.longitude, group.latitude), crs="EPSG:4326")
#     gdf = gdf.to_crs(buffered_coastline.crs)
#     gdf = gdf[['timestamp', 'speed', 'geometry']]
#     # gdf = gdf.reset_index()
#     # print(gdf.head())
#     # buffered_coastline = buffered_coastline.reset_index()
#     # print(buffered_coastline.head())
#     # print(pd._geom_predicate_query(gdf, buffered_coastline, 'within'))
#     # gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')
#     indices = geom_predicate_query(gdf, buffered_coastline, 'within')
#     # return indices
#     # gdf = gpd.sjoin(buffered_coastline, gdf, how="right", predicate='contains')
#     gdf['phase'] = pd.cut(gdf['speed'], [-np.inf, 3, 5, np.inf], right=True, include_lowest=True, labels=['Anchored', 'Manoeuvring', 'Sea'])
#     gdf.loc[(gdf['phase'] == 'Manoeuvring') & (gdf['index_right'].isna()), 'phase'] = 'Sea'
#     # # print(gdf[['timestamp', 'phase']].head())
#     return gdf[['timestamp', 'phase']]

# #%% Test the assign_phase function
# # meta_dict = {k: v for k, v in meta_dict.items() if k in ['mmsi', 'timestamp']}
# # meta_dict['phase'] = 'string'
# output = group.groupby('mmsi', group_keys=False).apply(test_fun).compute()
# output.head()


# # %%
# import warnings
# def geom_predicate_query(left_df, right_df, predicate):
#     """Compute geometric comparisons and get matching indices.

#     Parameters
#     ----------
#     left_df : GeoDataFrame
#     right_df : GeoDataFrame
#     predicate : string
#         Binary predicate to query.

#     Returns
#     -------
#     DataFrame
#         DataFrame with matching indices in
#         columns named `_key_left` and `_key_right`.
#     """
#     with warnings.catch_warnings():
#         # We don't need to show our own warning here
#         # TODO remove this once the deprecation has been enforced
#         warnings.filterwarnings(
#             "ignore", "Generated spatial index is empty", FutureWarning
#         )

#         original_predicate = predicate

#         if predicate == "within":
#             # within is implemented as the inverse of contains
#             # contains is a faster predicate
#             # see discussion at https://github.com/geopandas/geopandas/pull/1421
#             predicate = "contains"
#             sindex = left_df.sindex
#             input_geoms = right_df.geometry
#         else:
#             # all other predicates are symmetric
#             # keep them the same
#             sindex = right_df.sindex
#             input_geoms = left_df.geometry

#     if sindex:
#         l_idx, r_idx = sindex.query(input_geoms, predicate=predicate, sort=False)
#         indices = pd.DataFrame({"_key_left": l_idx, "_key_right": r_idx})
#     else:
#         # when sindex is empty / has no valid geometries
#         indices = pd.DataFrame(columns=["_key_left", "_key_right"], dtype=float)

#     if original_predicate == "within":
#         # within is implemented as the inverse of contains
#         # flip back the results
#         indices = indices.rename(
#             columns={"_key_left": "_key_right", "_key_right": "_key_left"}
#         )

#     return indices
# # %%
# geom_predicate_query(gdf, buffered_coastline, 'within')
# # %%

# %%