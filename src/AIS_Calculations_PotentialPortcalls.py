"""
Detect potential portcalls from cleaned dynamic AIS data.
Callvariant 'speed' uses a maximum speed threshold over a given time window to label potential portcalls.
Input(s): ais_bulkers_interp.parquet
Output(s): ais_bulkers_potportcalls_'callvariant'.parquet, potportcalls_'callvariant'.shp 
Runtime: 7m
"""

#%%
import dask.dataframe as dd
import pandas as pd
import os
import geopandas as gpd
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')
callvariant = 'speed' #'heading'

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_interp'))
ais_bulkers.head()

#%%
# ais_bulkers = ais_bulkers.partitions[0:2]
# df = ais_bulkers.partitions[0].compute()

# Trip detection
def pd_detect_trips_speed(df, time_window = '12H', speed_threshold = 1):
    df = df.copy() # was getting warning about writing to slice
    # Need to swap index from mmsi to timestamp for time rolling operation
    df['mmsi'] = df.index.get_level_values('mmsi')
    df.set_index('timestamp', inplace = True)

    ## Potential in-port detection
    df['pot_in_port'] = (
        df
        .groupby('mmsi')
        .speed
        .transform(lambda x: x.rolling(
            window = time_window,
            min_periods = 1)
            .apply(lambda x1: all(x1 < speed_threshold), raw = True)
            .fillna(False)
            .astype(int) # Convert to integer to avoid unpredictable behaviour differencing booleans
            .diff(1) # Use positive 1 to flag first observation after in port for 12H
            .gt(0) # Flag only in-port detection, not exit
            ))

    df['pot_trip'] = df.groupby('mmsi').pot_in_port.cumsum()
    df.insert(0, 'timestamp', df.index.get_level_values('timestamp'))
    df.set_index('mmsi', inplace = True)
    return df
#%%
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['pot_in_port'] = 'bool'
meta_dict['pot_trip'] = 'int'
ais_bulkers = ais_bulkers.map_partitions(pd_detect_trips_speed, meta = meta_dict)


#%% Compute and save
with LocalCluster(
    n_workers=2,
    threads_per_worker=3
) as cluster, Client(cluster) as client:
    ais_bulkers.to_parquet(
        os.path.join(filepath, 'ais_bulkers_potportcalls_' + callvariant),
        append = False,
        overwrite = True,
        engine = 'fastparquet')
# 5m10s (speed2)

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_potportcalls_' + callvariant))
ais_bulkers.head()

# Port call locations
#%% Use in port event (assign location from first observation once ship has been in port for 12H)
portcalls_df = (
    ais_bulkers
    .loc[ais_bulkers['pot_in_port'] == True,
        ['pot_trip', 'latitude', 'longitude']]
    .compute())


# portcalls_df['timestamp'] = portcalls_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
#%%
## Save as shapefile because can't figure out how to load csv to QGIS with python
# portcalls_df = pd.read_csv(os.path.join(datapath, 'portcalls.csv'))
portcalls_gdf = gpd.GeoDataFrame(
    portcalls_df,
    geometry=gpd.points_from_xy(
        portcalls_df.longitude,
        portcalls_df.latitude))
# 24s

#%%
filename = 'potportcalls_' + callvariant
outfilepath = os.path.join(datapath, filename)
if not os.path.exists(outfilepath):
    os.mkdir(outfilepath)
portcalls_gdf.to_file(
    os.path.join(outfilepath, filename + '.shp'),
    driver ='ESRI Shapefile'
    )
# 55s
# %%
