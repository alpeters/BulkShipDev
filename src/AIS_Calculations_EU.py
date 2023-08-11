"""
Merge portcall locations (EU or not) to AIS trips 
to identify trips into and out of EU
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, AIS_..._EU_yearly_stats.csv
Runtime: 5m
"""

#%%
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
import dask_geopandas
import os, time
import shapely
from shapely.wkt import loads
from shapely.geometry import Point
from dask.distributed import Client, LocalCluster
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial
import pyproj
import re
import numpy as np
import folium
from scipy.spatial.transform import Rotation as R, Slerp
from datetime import timedelta
from dask import delayed, compute
from math import sin, cos, asin, atan2, sqrt, radians, degrees

datapath = 'src/data'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''
filename = 'portcalls_' + callvariant + '_EU'

#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_' + callvariant))
ais_bulkers.head()

# %%
portcalls = pd.read_csv(
    os.path.join(datapath, 'pot' + filename + '.csv'),
    usecols = ['mmsi', 'pot_trip', 'EU', 'ISO_3digit'],
    dtype = {'mmsi' : 'int32',
             'pot_trip': 'int16',
             'EU': 'int8',
             'ISO_3digit': 'str'}
    )
portcalls = (
    portcalls
    .set_index('mmsi')
    .sort_values(['mmsi', 'pot_trip'])
    )
portcalls['pot_in_port'] = True

#%% merge portcalls to AIS data
with LocalCluster(
    n_workers=1,
    threads_per_worker=4
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .merge(portcalls,
            how = 'left',
            on = ['mmsi', 'pot_trip', 'pot_in_port'])
        .to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'), 
            append = False, 
            overwrite = True)
    )
# 2m8s
#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'))
ais_bulkers.head()
# ais_bulkers = ais_bulkers.partitions[9]


#%% Calculate new trip numbers
ais_bulkers['in_port'] = ~ais_bulkers['EU'].isnull()

def update_trip(df):
    df['trip'] = df.groupby('mmsi').in_port.cumsum()
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['trip'] = 'int'

#%%
with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    (
        ais_bulkers
        .map_partitions(update_trip, meta = meta_dict)
        .rename(columns = {'ISO_3digit': 'origin'})
        .to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_trips'), 
            append = False, 
            overwrite = True,
            engine = 'pyarrow') # getting strange overflow error with fastparquet
    )
# 1m29s
#%%
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips'))
ais_bulkers.head()

#%%
portcalls = (
    ais_bulkers
    .loc[ais_bulkers['in_port'] == True,
        ['trip', 'latitude', 'longitude', 'EU', 'origin']]
    .reset_index(drop = False)
    .compute())

portcalls.to_csv(os.path.join(datapath, filename + '.csv'))
#%%
# Add in trip 0 for each ship (these don't appear in portcalls because first portcall assigned to trip 1)
# Assume trip 0 was not from EU port (but may be to one)
trip0 = pd.DataFrame({'mmsi': portcalls.mmsi.unique(),
                      'trip': 0,
                      'EU': False,
                      'origin': np.NAN})
portcalls = pd.concat([portcalls, trip0])
portcalls.sort_values(by = ['mmsi', 'trip'], inplace = True)
# EU trips include travel from previous portcalls
portcalls['EU'] = portcalls.EU == 1 # assumes NA are not in EU
portcalls['prev'] = portcalls.groupby('mmsi').EU.shift(-1, fill_value = False)
EU_trips = portcalls[portcalls.EU | portcalls.prev]
EU_trips = EU_trips[['mmsi', 'trip']].set_index('mmsi')

#%%
# Filter dask dataframe to contain only these combinations
ais_bulkers_EU = ais_bulkers.merge(EU_trips,
    how = 'right',
    on = ['mmsi', 'trip'])

#%% Travel work
ais_bulkers_EU['work_IS'] = ais_bulkers_EU['implied_speed']**2 * ais_bulkers_EU['distance']
ais_bulkers_EU['work'] = ais_bulkers_EU['speed']**2 * ais_bulkers_EU['distance']

# Aggregate distance, etc. by year
#%%
ais_bulkers_EU['year'] = ais_bulkers_EU.timestamp.apply(
    lambda x: x.year,
    meta = ('x', 'int16'))

# Load the buffered reprojected shapefile (can be done by python or use QGIS directly)
# filepath should be changed
buffered_coastline = gpd.read_file('/Users/oliver/Desktop/Carbon Emission Project/buffered_reprojected_coastline.shp')

# Check the crs of the shapefile
print(buffered_coastline.crs)


# Define function to interpolate coordinates using Slerp (Spherical Linear Interpolation)
def interpolate_coordinates_slerp(row1, row2, num_points):
    lat1, lon1 = np.radians(row1['latitude']), np.radians(row1['longitude'])
    lat2, lon2 = np.radians(row2['latitude']), np.radians(row2['longitude'])

    x1, y1, z1 = np.cos(lon1) * np.cos(lat1), np.sin(lon1) * np.cos(lat1), np.sin(lat1)
    x2, y2, z2 = np.cos(lon2) * np.cos(lat2), np.sin(lon2) * np.cos(lat2), np.sin(lat2)

    rot1 = R.from_rotvec([x1, y1, z1])
    rot2 = R.from_rotvec([x2, y2, z2])

    slerp = Slerp([0, 1], R.from_rotvec([rot1.as_rotvec(), rot2.as_rotvec()]))

    interpolated_rows = []
    if num_points > 2:
        for t in np.linspace(1/(num_points-1), 1-1/(num_points-1), num_points-2): # Avoid the last timestamp to be duplciated 
            rot = slerp(t)
            x, y, z = rot.as_rotvec()

            lon, lat = np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

            new_row = row1.copy()
            new_row['latitude'] = lat
            new_row['longitude'] = lon
            new_row['timestamp'] += timedelta(hours=t*(row2['timestamp'] - row1['timestamp']).total_seconds()/3600)
            new_row['speed'] = row2['implied_speed']
            new_row['interpolated'] = True  # indicate this row is interpolated
            interpolated_rows.append(new_row)

    return interpolated_rows


def interpolate_missing_hours_slerp(df):
    
    if df.empty:
        print("DataFrame is empty")
        print("Index values:", df.index.values) # Print index values for further inspection
        return df

    df = df.assign(timestamp=pd.to_datetime(df['timestamp']))
    df = df.assign(interpolated=False)  # Set 'interpolated' to False for all original rows

    df_interpolated = [df.iloc[0].copy()]  # add the first row as is

    for i in range(len(df)-1):  # start from the first row
        row1, row2 = df.iloc[i].copy(), df.iloc[i+1].copy()
        time_interval_hours = int(round(row2['time_interval']))  # use the 'time_interval' from the second row

        # Skip interpolation if time_interval is less than or equal to 1 hour
        if time_interval_hours <= 1:
            df_interpolated.append(row2)
        else:
            interpolated_rows = interpolate_coordinates_slerp(row1, row2, time_interval_hours)
            df_interpolated.extend(interpolated_rows)  # add the interpolated rows
            df_interpolated.append(row2)  # add the 'end' timestamp here

    return pd.DataFrame(df_interpolated)


def pd_diff_haversine(df):
    df_lag = df.shift(1)
    timediff = (df.timestamp - df_lag.timestamp)/np.timedelta64(1, 'h')
    lat = np.radians(df.latitude)
    lng = np.radians(df.longitude)
    lat_lag = np.radians(df_lag.latitude)
    lng_lag = np.radians(df_lag.longitude)
    lat_diff = lat - lat_lag
    lng_diff = lng - lng_lag  
    d = (np.sin(lat_diff * 0.5) ** 2
         + np.cos(lat_lag) * np.cos(lat) * np.sin(lng_diff * 0.5) ** 2)
    dist = 2 * 6371.0088 * 0.539956803 * np.arcsin(np.sqrt(d))

    return df.assign(distance = dist,
                     time_interval = timediff) 


def infill_draught_partition(df):
    if df['draught'].isna().iloc[0]: # if the first value is NaN, apply backward fill
        df['draught'] = df['draught'].bfill()
    else:
        df['draught'] = df['draught'].ffill()
    return df


def process_group(group):
    group = interpolate_missing_hours_slerp(group)
    group = infill_draught_partition(group)
    group = pd_diff_haversine(group)
    return group
    
def process_partition_geo(df):
    # Create a new GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Reproject the left geometries to match the CRS of the right geometries
    gdf = gdf.set_crs(buffered_coastline.crs)

    # Determine if each point is within the buffer zone
    gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')

    # Set the phase based on whether the point is within the buffer zone and interpolated status
    gdf['phase'] = 'Sea'
    
    # For interpolated rows, use speed to determine phase
    gdf.loc[(gdf['interpolated'] == True) & (gdf['speed'] <= 3), 'phase'] = 'Anchored'
    gdf.loc[(gdf['interpolated'] == True) & (gdf['speed'] >= 4) & (gdf['speed'] <= 5), 'phase'] = 'Manoeuvring'
    gdf.loc[(gdf['interpolated'] == True) & (gdf['speed'] > 5), 'phase'] = 'Sea'
    
    # For non-interpolated rows, use geographical criteria
    gdf.loc[(gdf['interpolated'] == False) & (gdf['speed'] <= 3), 'phase'] = 'Anchored'
    gdf.loc[(gdf['interpolated'] == False) & (gdf['speed'] >= 4) & (gdf['speed'] <= 5) & (gdf.index_right.notna()), 'phase'] = 'Manoeuvring'
    gdf.loc[(gdf['interpolated'] == False) & (gdf['speed'] > 5), 'phase'] = 'Sea'
    
    # Drop the unneeded columns
    gdf = gdf[df.columns.tolist() + ['phase']]

    return gdf

def process_partition(df):
    result_list = []
    for _, group in df.groupby(df.index):
        processed_group = process_group(group)
        result_list.append(processed_group)
    processed_partition = pd.concat(result_list)
    return process_partition_geo(processed_partition)  # Apply geo function here

meta_dict = ais_bulkers.dtypes.to_dict()
# Include the new columns' data type
meta_dict['interpolated'] = 'bool'
meta_dict['phase'] = 'string' 

ais_bulkers = ais_bulkers.map_partitions(process_partition, meta=meta_dict)
ais_bulkers.dtypes

# Now we have parquet files with new column called 'phase'
# Get ready to join with wfr_bulkers_calcs to get hourly fuel consumption
# List of columns to keep
selected_columns = ['mmsi', 'ME_W_ref', 'W_component', 
                    'Dwt', 'ME_SFC_base', 'AE_SFC_base',
                    'Boiler_SFC_base', 'Draught..m.', 
                    'Service.Speed..knots.'] 

# Read the csv file into a Dask DataFrame with only the selected columns
wfr_bulkers = dd.read_csv('/Users/oliver/Desktop/Data/bulkers_WFR_calcs.csv', usecols=selected_columns)

# Join the Dask DataFrames
joined_df = ais_bulkers_EU.merge(wfr_bulkers, on='mmsi', how='inner')

# drop rows where hourly speed or draught is missing 
joined_df = joined_df.dropna(subset=['speed', 'draught'])


def calculate_FC_ME(ME_W_ref, W_component, draught, speed, ME_SFC_base):
    load = W_component * draught**0.66 * speed**3 
    return ME_W_ref * W_component * draught**0.66 * speed**3 * ME_SFC_base * (0.455 * load**2 - 0.710 * load + 1.280)

def calculate_FC_AE(AE_W, AE_SFC_base):
    return AE_W * AE_SFC_base

def calculate_Boiler_AE(Boiler_W, Boiler_SFC_base):
    return Boiler_W * Boiler_SFC_base

def calculate_hourly_power(df):
    # calculate hourly main engine power
    df['ME_W'] = df['ME_W_ref'] * df['W_component'] * (df['draught']**0.66) * (df['speed']**3)
    
    # set initial auxiliary engine power and boiler power to zero
    df['AE_W'] = 0
    df['Boiler_W'] = 0
    
    # conditions and power values
    conditions = [
        (df['Dwt'] <= 9999),
        ((df['Dwt'] > 9999) & (df['Dwt'] <= 34999)),
        ((df['Dwt'] > 34999) & (df['Dwt'] <= 59999)),
        ((df['Dwt'] > 59999) & (df['Dwt'] <= 99999)),
        ((df['Dwt'] > 99999) & (df['Dwt'] <= 199999)),
        (df['Dwt'] >= 200000)
    ]
    
    # Auxiliary engine power and boiler power values for each condition
    # Order: 'Anchored', 'Manoeuvring', 'Sea'
    phases = ['Anchored', 'Manoeuvring', 'Sea']
    ae_values = [(180, 500, 190), (180, 500, 190), (250, 680, 260), (400, 1100, 410), (400, 1100, 410), (400, 1100, 410)]
    boiler_values = [(70, 60, 0), (70, 60, 0), (130, 120, 0), (260, 240, 0), (260, 240, 0), (260, 240, 0)]
    
    # if main engine power is between 150 and 500
    # auxiliary engine is set to 5% of the main engine power
    # assign the boiler power based on phase and ship size
    df.loc[(df['ME_W'] > 150) & (df['ME_W'] <= 500), 'AE_W'] = df['ME_W'] * 0.05
    
    for condition, ae_value, boiler_value in zip(conditions, ae_values, boiler_values):
        for i, phase in enumerate(phases):
            df.loc[(df['ME_W'] > 150) & (df['ME_W'] <= 500) & condition & (df['phase'] == phase), 'Boiler_W'] = boiler_value[i]

    # if main engine power > 500, assign both power values based on phase and ship size
    for condition, ae_value, boiler_value in zip(conditions, ae_values, boiler_values):
        for i, phase in enumerate(phases):
            df.loc[(df['ME_W'] > 500) & condition & (df['phase'] == phase), 'AE_W'] = ae_value[i]
            df.loc[(df['ME_W'] > 500) & condition & (df['phase'] == phase), 'Boiler_W'] = boiler_value[i]

    return df

meta = joined_df._meta.assign(ME_W=pd.Series(dtype=float),
                              AE_W=pd.Series(dtype=float), 
                              Boiler_W=pd.Series(dtype=float))

joined_df = joined_df.map_partitions(calculate_hourly_power, meta=meta)

joined_df['FC_ME'] = calculate_FC_ME(joined_df['ME_W_ref'], 
                                     joined_df['W_component'], 
                                     joined_df['draught'], 
                                     joined_df['speed'], 
                                     joined_df['ME_SFC_base'])


joined_df['FC_AE'] = calculate_FC_AE(joined_df['AE_W'],
                                     joined_df['AE_SFC_base'])

joined_df['FC_Boiler'] = calculate_Boiler_AE(joined_df['Boiler_W'], 
                                             joined_df['Boiler_SFC_base'])

joined_df['FC'] = joined_df['FC_ME'] + joined_df['FC_AE'] + joined_df['FC_Boiler']

joined_df['t_m_times_v_n'] = joined_df['draught']**0.66 * joined_df['speed']**3
joined_df['t_over_t_ref_with_m'] = joined_df['draught']**0.66 / joined_df['Draught..m.']**0.66
joined_df['t_over_t_ref_without_m'] = joined_df['draught'] / joined_df['Draught..m.']
joined_df['v_over_v_ref_with_n'] = joined_df['speed']**3 / joined_df['Service.Speed..knots.']**3
joined_df['v_over_v_ref_without_n'] = joined_df['speed'] / joined_df['Service.Speed..knots.']

# Take a look at the joined_df
joined_df.head()

#%% https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),)

#%%
yearly_stats = (
    joined_df
    .groupby(['mmsi', 'year'])
    .agg({
        'distance': ['sum'],
        'work': ['sum'],
        'work_IS': ['sum'],
        'trip': nunique,
        'W_component': ['first'],
        'ME_W_ref': ['first'],
        't_m_times_v_n': ['sum'],
        't_over_t_ref_with_m': ['sum'],
        't_over_t_ref_without_m': ['sum'],
        'v_over_v_ref_with_n': ['sum'],
        'v_over_v_ref_without_n': ['sum'],
        'FC': ['sum']
        })
    .compute())

#%%
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))
# 43s

#%%
