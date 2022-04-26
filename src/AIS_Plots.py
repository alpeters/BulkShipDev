#%%
import dask.dataframe as dd
import pandas as pd
import os
import plotly.express as px
import geopandas as gpd
import numpy as np

#%%
filepath = 'src/data/AIS'
plotpath = 'plots/AIS_Plots'

#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_sample'))
ais_bulkers.head()

#%%
df = ais_bulkers.partitions[0].compute()

#%% Filter infeasible speeds
# df1 = df1[df1.speed < 25]
# df = df[df.implied_speed < 30]

#%% Hourly average speed below threshold for 24 hours
df['mmsi'] = df.index.get_level_values('mmsi')
df.set_index('timestamp', inplace = True)

#%% Work with a small sample for now
mmsi_selection = [False] * len(df.mmsi.unique())
for i in [0, 3, 10]:
    mmsi_selection[i] = True

df = df.loc[df['mmsi'].isin(df.mmsi.unique()[mmsi_selection])].copy()

#%%
speed_threshold = 4.3
df['port_exit'] = (
    df
    .groupby('mmsi')
    .implied_speed
    .transform(lambda x: x.rolling(
        window = '24H',
        min_periods = 1)
        .apply(lambda y:
            all(y < speed_threshold))
        .diff()
        .lt(0)))

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
for mmsi in df.mmsi.unique():
    df_plot = df.loc[df['mmsi'] == mmsi][df['implied_speed'] < 30]
    gdf = gpd.GeoDataFrame(
        df_plot, geometry=gpd.points_from_xy(df_plot.longitude, df_plot.latitude))

    fig = px.scatter_geo(
        gdf,
        lat='latitude',
        lon='longitude',
        hover_name='timestamp',
        color='implied_speed',
        title=str(mmsi))
    fig.write_image(os.path.join(plotpath, str(mmsi) + '_implied_speed.png'))
    # fig.show()

## World Ports
#%%
wpi = gpd.read_file("./data/WPI/WPI.shp")
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
fig.update_layout(showlegend=False)
# fig.write_image(os.path.join(plotpath, 'WPI.png'))
fig.show()

## Plot trips
#%%
# df_plot = df[df['duration'].lt(pd.Timedelta("2 days"))]
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
        lat=gdf_wpi['LATITUDE'],
        lon=gdf_wpi['LONGITUDE'],
        opacity = 0.3,
        marker={'symbol': 'cross'}
    )
    fig.write_image(os.path.join(plotpath, str(mmsi) + '_trips.png'))
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

# %%
