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


#%%
def interpolate_coordinates_slerp(row1, row2, num_points):
    lat1, lon1 = np.radians(row1['latitude']), np.radians(row1['longitude'])
    lat2, lon2 = np.radians(row2['latitude']), np.radians(row2['longitude'])

    rot1 = R.from_rotvec(np.array([np.cos(lon1) * np.cos(lat1), np.sin(lon1) * np.cos(lat1), np.sin(lat1)]))
    rot2 = R.from_rotvec(np.array([np.cos(lon2) * np.cos(lat2), np.sin(lon2) * np.cos(lat2), np.sin(lat2)]))

    slerp = Slerp([0, 1], R.from_rotvec([rot1.as_rotvec(), rot2.as_rotvec()]))

    if num_points <= 2:
        return []

    step = 1/(num_points-1)
    t_values = np.linspace(step, 1-step, num_points-2)
    rot_values = slerp(t_values)
    rotvec_values = rot_values.as_rotvec()

    lon_lat_values = np.degrees(np.column_stack((
        np.arctan2(rotvec_values[:, 1], rotvec_values[:, 0]),
        np.arctan2(rotvec_values[:, 2], np.sqrt(rotvec_values[:, 0]**2 + rotvec_values[:, 1]**2))
    )))

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

    return interpolated_rows

#%%
def interpolate_missing_hours_slerp(df):
    # df = df.assign(timestamp=pd.to_datetime(df['timestamp']), interpolated=False)
    df['interpolated'] = False

    def generate_interpolated_rows():
        yield df.index[0], df.iloc[0]
        for i in range(len(df)-1):
            row1, row2 = df.iloc[i], df.iloc[i+1]
            time_interval_hours = int((row2['timestamp']-row1['timestamp']).total_seconds()/3600)
            if time_interval_hours <= 1:
                yield df.index[i+1], row2
            else:
                interpolated_rows = interpolate_coordinates_slerp(row1, row2, time_interval_hours)
                for row in interpolated_rows:
                    yield df.index[i+1], pd.Series(row)
                yield df.index[i+1], row2

    return pd.DataFrame(
        data=(item[1] for item in generate_interpolated_rows()),
        index=(item[0] for item in generate_interpolated_rows())
    )

#%%
def interpolate_missing_hours_slerp_old(df):


    df = df.assign(timestamp=pd.to_datetime(df['timestamp']), interpolated=False)

    df_interpolated = [(df.index[0], df.iloc[0])]
    for i in range(len(df)-1):
        row1, row2 = df.iloc[i], df.iloc[i+1]
        time_interval_hours = int(round(row2['time_interval']))

        if time_interval_hours <= 1:
            df_interpolated.append((df.index[i+1], row2))
        else:
            interpolated_rows = interpolate_coordinates_slerp(row1, row2, time_interval_hours)
            df_interpolated.extend([(df.index[i+1], pd.Series(row)) for row in interpolated_rows])
            df_interpolated.append((df.index[i+1], row2))

    return pd.DataFrame(
        data=[item[1] for item in df_interpolated],
        index=[item[0] for item in df_interpolated]
    )

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
    group = group.drop(columns=['data_counter', 'timestamp_hour', 'interp_steps', 'interp_step', 'interp_counter', 'interp_coord_index']) 
    
    return group


###########################
# ais_bulkers = pd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_firstpart'))
#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_test'))

#%% Create test file with first rows of each partition
(
    ais_bulkers
    .map_partitions(lambda df: df.head(20000)).to_parquet(
    os.path.join(filepath, 'ais_bulkers_calcs_testheads'),
    append=False,
    overwrite=True,
    engine='fastparquet')
)
#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_testheads'))

ais_bulkers.head()

# %%
with LocalCluster(
    n_workers=2,
    # processes=True,
    threads_per_worker=3
    # memory_limit='2GB',
    # ip='tcp://localhost:9895',
) as cluster, Client(cluster) as client:
    start_time = time.time()
    ais_bulkers.map_partitions(lambda df: df.groupby('mmsi').apply(interpolate_missing_hours_slerp_old)).to_parquet(
        os.path.join(filepath, 'ais_bulkers_interp'),
        append=False,
        overwrite=True,
        engine='fastparquet'
    )
    print(f"Time: {time.time() - start_time}")

#%%
start_time = time.time()
interp = (
    ais_bulkers
    .groupby('mmsi')
    .apply(interpolate_missing_hours)
).to_parquet(os.path.join(filepath, 'ais_bulkers_calcs_testheads_interpolated'), engine='fastparquet')
print(f"Time: {time.time() - start_time}")

# %%
# start_time = time.time()

# interp = (
#     ais_bulkers
#     .groupby('mmsi')
#     .apply(interpolate_missing_hours_slerp_old)
# ).compute()
# print(f"Time: {time.time() - start_time}")


# #%% Plot the rows of ais_bulkers as points on a map using geopandas
# from shapely.geometry import Point

# # Assume df is your DataFrame and it has 'longitude' and 'latitude' columns
# geometry = [Point(xy) for xy in zip(interp['longitude'], interp['latitude'])]

# # Convert the DataFrame to a GeoDataFrame
# gdf = gpd.GeoDataFrame(interp, geometry=geometry)
# #%%
# # Plot the GeoDataFrame, with colors based on the interpolated column, include legend, smaller point size
# gdf.plot(column='interpolated', cmap='viridis', legend = True, markersize=1)
# # gdf.plot()





