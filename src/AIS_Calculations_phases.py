"""
Detect trip phases from cleaned dynamic AIS data.
Input(s): ais_bulkers_calcs.parquet
Output(s):
Runtime:
"""

# %%
import dask.dataframe as dd
import pandas as pd
import os
import geopandas as gpd
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
filepath = os.path.join(datapath, 'AIS')
callvariant = 'speed'  # 'heading'

# %%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
pd.set_option('display.max_columns', 24)
ais_bulkers.head()


# %%
# speed detection: 3-5 knots by default
# Using implied speed as reference
def pd_detect_speed_implied(df, speed_threshold_low=3, speed_threshold_high=5):
    df['implied_speed_3_to_5'] = df['implied_speed'].apply(lambda x:
                                                           True if speed_threshold_low <= x <= speed_threshold_high
                                                           else False)

    df_implied_speed_3_to_5 = df[df['implied_speed_3_to_5'] == True]
    return df_implied_speed_3_to_5


# Using AIS reported speed as reference
def pd_detect_speed(df, speed_threshold_low=3, speed_threshold_high=5):
    df['speed_3_to_5'] = df['speed'].apply(lambda x:
                                           True if speed_threshold_low < x <= speed_threshold_high else False)

    df_speed_3_to_5 = df[df['speed_3_to_5'] == True]
    return df_speed_3_to_5


# %%
# Temporary Test
ais_bulkers = ais_bulkers.partitions[0:2]
df = ais_bulkers.partitions[0].compute()
print(df.iloc[1])
selected_df = pd_detect_speed(df)
print(selected_df)
# %%
# meta_dict = ais_bulkers.dtypes.to_dict()
# meta_dict['pot_in_port'] = 'bool'
# meta_dict['pot_trip'] = 'int'
# ais_bulkers = ais_bulkers.map_partitions(pd_detect_trips_speed, meta=meta_dict)
#
# # %% Compute and save
# with LocalCluster(
#         n_workers=2,
#         threads_per_worker=3
# ) as cluster, Client(cluster) as client:
#     ais_bulkers.to_parquet(
#         os.path.join(filepath, 'ais_bulkers_potportcalls_' + callvariant),
#         append=False,
#         overwrite=True,
#         engine='fastparquet')
# # 5m10s (speed2)
#
# # %%
# ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_potportcalls_' + callvariant))
# ais_bulkers.head()
#
# # Port call locations
# # %% Use in port event (assign location from first observation once ship has been in port for 12H)
# portcalls_df = (
#     ais_bulkers
#     .loc[ais_bulkers['pot_in_port'] == True,
#     ['pot_trip', 'latitude', 'longitude']]
#     .compute())
#
# # portcalls_df['timestamp'] = portcalls_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
# # %%
# ## Save as shapefile because can't figure out how to load csv to QGIS with python
# # portcalls_df = pd.read_csv(os.path.join(datapath, 'portcalls.csv'))
# portcalls_gdf = gpd.GeoDataFrame(
#     portcalls_df,
#     geometry=gpd.points_from_xy(
#         portcalls_df.longitude,
#         portcalls_df.latitude))
# # 24s
#
# # %%
# filename = 'potportcalls_' + callvariant
# outfilepath = os.path.join(datapath, filename)
# if not os.path.exists(outfilepath):
#     os.mkdir(outfilepath)
# portcalls_gdf.to_file(
#     os.path.join(outfilepath, filename + '.shp'),
#     driver='ESRI Shapefile'
# )
# 55s
# %%

# Load the buffered reprojected shapefile (can be done by python or use QGIS directly)
buffered_coastline = gpd.read_file('/Users/oliver/Desktop/Carbon Emission Project/buffered_reprojected_coastline.shp')

# Check the crs of the shapefile
print(buffered_coastline.crs)

def set_operational_phase(geometry):
    return geometry.within(buffered_coastline.unary_union)

gdf['operational_phase'] = gdf['geometry'].map_partitions(set_operational_phase, meta=('geometry', 'bool'))
gdf['operational_phase'] = gdf['operational_phase'].mask(gdf['operational_phase'], 'manoeuvring').fillna('sea')


ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
ais_bulkers['geometry'] = dd.from_pandas(gpd.points_from_xy(ais_bulkers.longitude, ais_bulkers.latitude), npartitions=ais_bulkers.npartitions)
gdf = dask_geopandas.from_dask_dataframe(ais_bulkers)
gdf = gdf.set_crs(buffered_coastline.crs)
gdf['operational_phase'] = 'sea'
gdf['operational_phase'] = gdf['geometry'].map_partitions(set_operational_phase, meta=('geometry', 'bool'))
gdf['operational_phase'] = gdf['operational_phase'].mask(gdf['operational_phase'], 'manoeuvring').fillna('sea')