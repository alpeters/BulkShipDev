import os, time
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from dask.distributed import Lock
from scipy.spatial.transform import Rotation as R, Slerp



# Create a SLURM cluster object
# cluster = SLURMCluster(
#     account='def-kasahara-ab',
#     cores=1,  # This matches --ntasks-per-node in the job script
#     memory='100GB', # Total memory
#     walltime='1:00:00'
#     #job_extra=['module load proj/9.0.1',
#                #'source ~/carbon/bin/activate']
# )

# # This matches --nodes in the SLURM script
# cluster.scale(jobs=3) 

# # Connect Dask to the cluster
# client = Client(cluster)

# # Check Dask dashboard 
# with open('dashboard_url.txt', 'w') as f:
#     f.write(client.dashboard_link)


# %%
datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')

# %%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs_testheads'))

# %%

# Load the buffered reprojected shapefile (can be done by python or use QGIS directly)
# buffered_coastline = gpd.read_file(os.path.join(filepath, 'buffered_reprojected_coastline.shp'))
buffered_coastline = gpd.read_file(os.path.join(datapath, 'buffered_reprojected_coastline', 'buffered_reprojected_coastline.shp'))



# %%
# Define function to interpolate coordinates using Slerp (Spherical Linear Interpolation)
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


def interpolate_missing_hours_slerp(df):


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


def pd_diff_haversine(df):
    df_lag = df.shift(1)
    timediff = (df.timestamp - df_lag.timestamp)/np.timedelta64(1, 'h')
    haversine_formula = 2 * 6371.0088 * 0.539956803  # precompute constant

    lat_diff, lng_diff = np.radians(df.latitude - df_lag.latitude), np.radians(df.longitude - df_lag.longitude)
    d = (np.sin(lat_diff * 0.5) ** 2 + np.cos(np.radians(df_lag.latitude)) * np.cos(np.radians(df.latitude)) * np.sin(lng_diff * 0.5) ** 2)
    dist = haversine_formula * np.arcsin(np.sqrt(d))

    return df.assign(distance=dist, time_interval=timediff)


def infill_draught_partition(df):
    df['draught'].bfill(inplace=True) if df['draught'].isna().iloc[0] else df['draught'].ffill(inplace=True)
    return df


def process_partition_geo(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    gdf = gdf.set_crs(buffered_coastline.crs)
    gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')
    phase_conditions = [
        (gdf['speed'] <= 3, 'Anchored'),
        ((gdf['speed'] >= 4) & (gdf['speed'] <= 5), 'Manoeuvring'),
        (gdf['speed'] > 5, 'Sea')
    ]
    for condition, phase in phase_conditions:
        gdf.loc[condition, 'phase'] = phase

    return gdf[df.columns.tolist() + ['phase']]


def process_group(group):
    group = interpolate_missing_hours_slerp(group)
    # group = infill_draught_partition(group)
    # group = pd_diff_haversine(group)
    # group = process_partition_geo(group)
    return group

def process_partition(df):
    df = (
        df
        .groupby('mmsi')
        .apply(process_group)
        .reset_index(level=0, drop=True)
    )
    df.index.name = 'mmsi'
    return df


meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['interpolated'] = 'bool'
# meta_dict['phase'] = 'string'


start_time = time.time()
with LocalCluster(
    n_workers=2,
    # processes=True,
    threads_per_worker=3
    # memory_limit='2GB',
    # ip='tcp://localhost:9895',
) as cluster, Client(cluster) as client:
    ais_bulkers.map_partitions(process_partition, meta=meta_dict).to_parquet(
        os.path.join(filepath, 'ais_bulkers_interp_original'),
        append=False,
        overwrite=True,
        engine='fastparquet'
    )
print(f"Time: {time.time() - start_time}")


# # Shut down the cluster
# client.close()
# cluster.close()