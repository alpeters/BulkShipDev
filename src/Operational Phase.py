"""
Assign operational phase to hourly AIS data
Input(s): any ais parquet file
Output(s): ais_bulkers_phase.parquet
Runtime:
"""


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


# In[2]:


datapath = '/Users/oliver/Desktop/data'
filepath = os.path.join(datapath, 'AIS')

# Load the buffered reprojected shapefile (can be done by python or use QGIS directly)
buffered_coastline = gpd.read_file('/Users/oliver/Desktop/Carbon Emission Project/buffered_reprojected_coastline.shp')

# Check the crs of the shapefile
print(buffered_coastline.crs)

ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))

ais_bulkers['year'] = ais_bulkers.timestamp.apply(
    lambda x: x.year,
    meta = ('x', 'int16'))

pd.set_option('display.max_columns', 24)

ais_bulkers.head()


# In[3]:


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
meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['phase'] = 'string'

# Call map_partitions with the new meta argument
ais_bulkers = ais_bulkers.map_partitions(process_partition, meta=meta_dict)
ais_bulkers.dtypes


ais_bulkers.head()

