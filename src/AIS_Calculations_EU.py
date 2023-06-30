"""
Merge portcall locations (EU or not) to AIS trips 
to identify trips into and out of EU
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, AIS_..._EU_yearly_stats.csv
Runtime: 5m
"""

#%%
import sys, os
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
import dask_geopandas
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

def process_partition(df):
    # Create a new GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Reproject the left geometries to match the CRS of the right geometries
    gdf = gdf.set_crs(buffered_coastline.crs)

    # Determine if each point is within the buffer zone
    gdf = gpd.sjoin(gdf, buffered_coastline, how="left", predicate='within')

    # Set the phase based on whether the point is within the buffer zone
    gdf['phase'] = 'Sea'
    gdf.loc[(gdf['speed'] <= 3), 'phase'] = 'Anchored'
    gdf.loc[(gdf['speed'] >= 4) & (gdf['speed'] <= 5) & (gdf.index_right.notna()), 'phase'] = 'Manoeuvring'
    gdf.loc[(gdf['speed'] > 5), 'phase'] = 'Sea'

    # Drop the unneeded columns
    gdf = gdf[df.columns.tolist() + ['phase']]

    return gdf

# Create a dictionary of column data types
meta_dict = ais_bulkers_EU.dtypes.to_dict()
meta_dict['phase'] = 'string'

# Call map_partitions with the new meta argument
ais_bulkers_EU = ais_bulkers_EU.map_partitions(process_partition, meta=meta_dict)
ais_bulkers_EU.dtypes

# Now we have parquet files with new column called 'phase'
# Get ready to join with wfr_bulkers_calcs to get hourly fuel consumption
# List of columns to keep
selected_columns = ['mmsi', 'ME_W_ref', 'W_component', 'Dwt', 'SFC_base', 'Service.Speed..knots.'] 

# Read the csv file into a Dask DataFrame with only the selected columns
# filepath to be changed
wfr_bulkers = dd.read_csv('/Users/oliver/Desktop/Data/bulkers_WFR_calcs.csv', usecols=selected_columns)

# Join the Dask DataFrames
joined_df = ais_bulkers_EU.merge(wfr_bulkers, on='mmsi', how='inner')

# drop rows where hourly speed or draught is missing 
joined_df = joined_df.dropna(subset=['speed', 'draught'])


def calculate_FC_ME(ME_W_ref, W_component, draught, speed, SFC_base, service_speed):
    load = speed / service_speed
    return ME_W_ref * W_component * draught**0.66 * speed**3 * SFC_base * (0.455 * load**2 - 0.710 * load + 1.280)

def calculate_FC_AE(AE_W_ref, W_component, draught, speed, SFC_base):
    return AE_W_ref * W_component * draught**0.66 * speed**3 * SFC_base

def calculate_Boiler_AE(Boiler_W_ref, W_component, draught, speed, SFC_base):
    return Boiler_W_ref * W_component * draught**0.66 * speed**3 * SFC_base

def assign_values(df):
    ae_values = {
        'Anchored': [180, 180, 250, 400, 400, 400],
        'Manoeuvring': [500, 500, 680, 1100, 1100, 1100],
        'Sea': [190, 190, 260, 410, 410, 410]
    }
    
    boiler_values = {
        'Anchored': [70, 70, 130, 260, 260, 260],
        'Manoeuvring': [60, 60, 120, 240, 240, 240],
        'Sea': [0, 0, 0, 0, 0, 0]
    }
    
    for phase in ['Anchored', 'Manoeuvring', 'Sea']:
        phase_df = df[df['phase'] == phase]
        conditions = [
            phase_df['Dwt'] < 10000,
            (phase_df['Dwt'] >= 10000) & (phase_df['Dwt'] < 35000),
            (phase_df['Dwt'] >= 35000) & (phase_df['Dwt'] < 60000),
            (phase_df['Dwt'] >= 60000) & (phase_df['Dwt'] < 100000),
            (phase_df['Dwt'] >= 100000) & (phase_df['Dwt'] < 200000),
            phase_df['Dwt'] >= 200000
        ]
        df.loc[df['phase'] == phase, 'AE_W_ref'] = np.select(conditions, ae_values[phase], default=0)
        df.loc[df['phase'] == phase, 'Boiler_W_ref'] = np.select(conditions, boiler_values[phase], default=0)
    
    return df


meta = joined_df._meta.assign(AE_W_ref=pd.Series(dtype=float), Boiler_W_ref=pd.Series(dtype=float))
joined_df = joined_df.map_partitions(assign_values, meta=meta)

joined_df['FC_ME'] = calculate_FC_ME(joined_df['ME_W_ref'], 
                                     joined_df['W_component'], 
                                     joined_df['draught'], 
                                     joined_df['speed'], 
                                     joined_df['SFC_base'], 
                                     joined_df['Service.Speed..knots.'])


joined_df['FC_AE'] = calculate_FC_AE(joined_df['AE_W_ref'], 
                                     joined_df['W_component'], 
                                     joined_df['draught'], 
                                     joined_df['speed'], 
                                     joined_df['SFC_base'])

joined_df['FC_Boiler'] = calculate_Boiler_AE(joined_df['Boiler_W_ref'], 
                                             joined_df['W_component'], 
                                             joined_df['draught'], 
                                             joined_df['speed'], 
                                             joined_df['SFC_base'])

joined_df['FC'] = joined_df['FC_ME'] + joined_df['FC_AE'] + joined_df['FC_Boiler']
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
        'FC': ['sum']
        })
    .compute())

#%%
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))
# 43s

#%%
