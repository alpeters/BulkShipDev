#%%
# from turtle import heading
import dask.dataframe as dd
import pandas as pd
import os
import plotly.express as px
import geopandas as gpd
import numpy as np

#%%
filepath = 'data/AIS'

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
ais_bulkers.head()

#%%
df = ais_bulkers.partitions[0].compute()

#%% Filter infeasible speeds
# df1 = df1[df1.speed < 25]
# df = df[df.implied_speed < 30]

# Trip detection

#%% Hourly average speed below threshold for 24 hours
df['mmsi'] = df.index.get_level_values('mmsi')
df.set_index('timestamp', inplace = True)

#%% Work with a small sample for now
mmsi_selection = [False] * len(df.mmsi.unique())
for i in [0, 3, 10]:
    mmsi_selection[i] = True

df = df.loc[df['mmsi'].isin(df.mmsi.unique()[mmsi_selection])].copy()

## Method 1: Average speed threshold
# #%%
# speed_threshold = 4.3
# df['port_exit'] = (
#     df
#     .groupby('mmsi')
#     .implied_speed
#     .transform(lambda x: x.rolling(
#         window = '24H',
#         min_periods = 1)
#         .apply(lambda y:
#             all(y < speed_threshold))
#         .diff()
#         .lt(0)))

## Method 2: Instantaneous speed threshold
# #%%
# speed_threshold = 1.0
# df['port_exit'] = (
#     df
#     .groupby('mmsi')
#     .speed
#     .transform(lambda x: x.rolling(
#         window = '24H',
#         min_periods = 1)
#         .apply(lambda y:
#             all(y < speed_threshold))
#         .diff()
#         .lt(0)))

## Method 3: Std heading threshold and instantaneous speed threshold
#%%
heading_threshold = 1
speed_threshold = 1
df['dir_change'] = (
    df
    .groupby('mmsi')
    .heading
    .transform(lambda x: x.rolling(
        window = '12H',
        min_periods = 1)
        .std()
        # .lt(heading_threshold)
        # .astype(int)
        # .diff(-1)
        # .gt(0)
        ))

df['int'] = df.dir_change.lt(heading_threshold).astype(int).diff(1).gt(0)
df['port_exit'] = df['int'] & (df['speed'] < speed_threshold)

#%%
df['trip'] = (
    df
    .groupby('mmsi')
    .port_exit
    .cumsum()
    )

#%% Trip duration
df = df.reset_index(drop = False)
def timeextent(group):
    group['duration'] = group['timestamp'].max() - group['timestamp'].min()
    return group
df = df.groupby(['mmsi', 'trip']).apply(timeextent)

# Plot some ship trajectories
## Plot speed
#%%
# for mmsi in df.mmsi.unique():
mmsi = df.mmsi.unique()[0]
# df_plot = df.loc[df['mmsi'] == mmsi][df['implied_speed'] < 30]
df_plot = df.loc[df['mmsi'] == mmsi][df['speed'] < 30]
gdf = gpd.GeoDataFrame(
    df_plot, geometry=gpd.points_from_xy(df_plot.longitude, df_plot.latitude))

fig = px.scatter_geo(
    gdf,
    lat='latitude',
    lon='longitude',
    hover_name='timestamp',
    # color='implied_speed',
    color='speed',
    title=str(mmsi))
fig.show()

## World Ports
#%%
wpi = gpd.read_file("./data/world_port_index/WPI.shp")
gdf_wpi = gpd.GeoDataFrame(
    wpi, geometry = gpd.points_from_xy(wpi.LONGITUDE, wpi.LATITUDE))

#%%
fig = px.scatter_geo(
    gdf_wpi,
    lat='LATITUDE',
    lon='LONGITUDE',
    hover_name='PORT_NAME',
    color='COUNTRY',
    title="Ports"
)
fig.show()

## Plot trips
#%%
# for mmsi in df.mmsi.unique():
mmsi = df.mmsi.unique()[0]
df_plot = df.loc[df['mmsi'] == mmsi]
gdf = gpd.GeoDataFrame(
    df_plot, geometry=gpd.points_from_xy(df_plot.longitude, df_plot.latitude))

fig = px.scatter_geo(
    gdf,
    lat='latitude',
    lon='longitude',
    hover_name='timestamp',
    color='trip',
    title=str(mmsi))
fig.add_scattergeo(
    lat=gdf_wpi['LATITUDE'],
    lon=gdf_wpi['LONGITUDE'],
    opacity = 0.3,
    marker={'symbol': 'cross'}
)
fig.show()


#%% Short Trips
df_plot = df[df['duration'].lt(pd.Timedelta("2 days"))]
for mmsi in df.mmsi.unique():
    df_plot = df_plot.loc[df['mmsi'] == mmsi]
    gdf = gpd.GeoDataFrame(
        df_plot, geometry=gpd.points_from_xy(df_plot.longitude, df_plot.latitude))

    fig = px.scatter_geo(
        gdf,
        lat='latitude',
        lon='longitude',
        hover_name='timestamp',
        color='speed',
        symbol='trip',
        title=str(mmsi))
    fig.add_scattergeo(
        lat=gdf_wpi['LATITUDE'],
        lon=gdf_wpi['LONGITUDE'],
        opacity = 0.3,
        marker={'symbol': 'cross'}
    )
    fig.show()

#%% Port call location
# Using minimum implied_speed in a given trip
# portcalls = df.loc[df.groupby(['mmsi', 'trip']).implied_speed.idxmin()]
# portcalls = df.loc[df.groupby(['mmsi', 'trip']).speed.idxmin()]

# portcalls = df.loc[df.groupby().B.idxmin()]


# Using port exit detection
portcalls = (
    df.loc[df['port_exit'] == True]
)

#%%
for mmsi in df.mmsi.unique():
    df_plot = df.loc[df['mmsi'] == mmsi]
    gdf = gpd.GeoDataFrame(
        df_plot, geometry=gpd.points_from_xy(df_plot.longitude, df_plot.latitude))

    fig = px.scatter_geo(
        gdf,
        lat='latitude',
        lon='longitude',
        hover_name='timestamp',
        color='trip',
        title=str(mmsi))
    fig.add_scattergeo(
        lat=portcalls.loc[(mmsi,), 'latitude'],
        lon=portcalls.loc[(mmsi,), 'longitude'],
        # lat=portcalls.loc[portcalls['mmsi'] == mmsi, 'latitude'],
        # lon=portcalls.loc[portcalls['mmsi'] == mmsi, 'longitude'],
        opacity = 0.5,
        marker={'symbol': 'x',
                'size': 20,
                'color': 'black'}
    )
    fig.show()

# %%
fig = px.scatter_geo(
        portcalls.loc[df.mmsi.unique()[0]].reset_index(),
        # portcalls.loc[portcalls['mmsi'] == df.mmsi.unique()[0]],
        lat='latitude',
        lon='longitude',
        # hover_name='timestamp',
        color='trip',
        title='Port Calls')

fig.update_traces(marker={'symbol': 'x'})

fig.show()
# %% Save sample to csv to verify in QGIS
sample_mmsi = df.mmsi.unique()[2]
portcalls.loc[portcalls['mmsi'] == sample_mmsi].to_csv('data/' + str(sample_mmsi) + '_portcalls.csv')
df.loc[df['mmsi'] == sample_mmsi].to_csv('data/' + str(sample_mmsi) + '_trajectory.csv')
# Checked first 3 visually. Typically too sensitive, but this is better than the opposite.
# Found one example where there was likely a port call but 12 hours between observations.
# Found another trip to Russia but no clear landing.
# %%
portcalls.to_csv('data/portcalls.csv')

# %%
