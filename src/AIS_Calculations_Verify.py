"""
Check bulk ship AIS data for errors such as:
    - missing observations,
    - null values
Input(s): ais_bulkers_calcs.parquet, AIS_yearly_stats.csv
Output(s): none
"""

#%%
import dask.dataframe as dd
import pandas as pd
import seaborn as sns
import os, time
import numpy as np
import random2

datapath = 'data'
filepath = os.path.join(datapath, 'AIS')

#%%
yearly_stats = (
    pd.read_csv(
        os.path.join(datapath, 'AIS_yearly_stats.csv'),
        parse_dates = ['timestamp_min', 'timestamp_max'])
    .set_index(['mmsi', 'year'])
)

#%%
# Aggregate values
# -----------------

#%% Count
g = sns.displot(
    yearly_stats,
    x = 'timestamp_count',
    hue = 'year',
    palette = 'tab10'
)
# g.set_titles("Count")
g.set_axis_labels("Annual Observations", "")

#%% Delay before first observation
yearly_stats['delay_first'] = pd.to_datetime(
        yearly_stats.index.get_level_values('year'),
        format = '%Y',
        utc = True)
yearly_stats['delay_first'] = yearly_stats['timestamp_min'] - yearly_stats['delay_first']
yearly_stats['delay_first'] = (
    yearly_stats['delay_first']
    .apply(lambda x: x.total_seconds()/(60*60*24)))
#%%
g = sns.displot(
    yearly_stats.loc[yearly_stats['delay_first'] > 1/2],
    x = 'delay_first',
    col = 'year',
    palette = 'tab10',
    bins = 365
)
# g.set_titles("Count")
g.set_axis_labels("Annual Observations", "")

#%% Delay after last observation

#%% Max Time Interval
g = sns.displot(
    yearly_stats,
    x = 'time_interval_max',
    hue = 'year',
    palette = 'tab10',
    bins = 365*2
)
# g.set_titles("Count")
g.set_axis_labels("Max Time Interval", "")

#%% Max Time Interval Filtered
g = sns.displot(
    yearly_stats.loc[yearly_stats['time_interval_max'] < 1000],
    x = 'time_interval_max',
    row = 'year',
    palette = 'tab10',
    bins = 365
)
# g.set_titles("Count")
g.set_axis_labels("Max Time Interval", "")

#%% Mean Time Interval
g = sns.displot(
    yearly_stats,
    x = 'time_interval_mean',
    hue = 'year',
    palette = 'tab10',
    bins = 365*2
)
# g.set_titles("Count")
g.set_axis_labels("Mean Time Interval", "")

#%% Mean Time Interval Filtered
g = sns.displot(
    yearly_stats.loc[yearly_stats['time_interval_mean'] > 24],
    x = 'time_interval_mean',
    hue = 'year',
    palette = 'tab10',
    bins = 365*2
)
# g.set_titles("Count")
g.set_axis_labels("Mean Time Interval", "")

#%% Invalid implied speeds in a year
yearly_stats.IS_gt30.gt(0).groupby('year').value_counts()


#%%
# Single ship
# -----------
#%%
ais_bulkers = dd.read_parquet(os.path.join(filepath, 'ais_bulkers_calcs'))
ais_bulkers.head()

### Random ship
#%%
mmsi = random2.sample(yearly_stats.index.get_level_values(0).unique(), 1)[0]
print("Random ship is: ", mmsi)
#%%
random_ship = ais_bulkers.loc[ais_bulkers.index == mmsi].compute()
#%%
sns.histplot(
    random_ship.distance,
    binwidth = 5
)
#%%
sns.histplot(
    random_ship.time_interval,
    binwidth = 2
)
#%%
random_ship.loc[random_ship['distance'] > 100]

#%%
sns.scatterplot(
    data = random_ship,
    x = 'speed',
    y = 'implied_speed'
)

#%%
random_ship[random_ship.implied_speed < 0.5]

# Are invalid coords, speeds a problem?
#%%
def temp():
    lats_in = ais_bulkers.latitude[ais_bulkers.latitude.between(-90, 90)].count()
    lats_tot = ais_bulkers.latitude.count()
    longs_in = ais_bulkers.longitude[ais_bulkers.longitude.between(-180, 180)].count()
    speeds_in = ais_bulkers.speed[ais_bulkers.speed.lt(30)].count()
    impspeed_in = ais_bulkers.speed[ais_bulkers.implied_speed.lt(30)].count()
    return lats_in, lats_tot, longs_in, speeds_in, impspeed_in

out = temp()
dd.compute(out)


