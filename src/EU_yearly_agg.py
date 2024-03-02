"""
Merge portcall locations (EU or not) to AIS trips 
to identify trips into and out of EU &
Aggregate yearly fuel consumption
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, AIS_..._EU_yearly_stats.csv
"""


import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = '/home/oliverxu/data'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''
filename = 'portcalls_' + callvariant + '_EU'


ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_speed'))
ais_bulkers['timestamp'] = ais_bulkers['timestamp'].dt.tz_localize(None)
ais_bulkers['timestamp'] = ais_bulkers['timestamp'].astype('datetime64[us]')
ais_bulkers.head()

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

ais_bulkers.merge(portcalls,
                  how = 'left',
                  on = ['mmsi', 'pot_trip', 'pot_in_port']).to_parquet(
            os.path.join(filepath, 'ais_bulkers_pottrips'), 
            append = False, 
            overwrite = True)

ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'))
ais_bulkers.head()


ais_bulkers['in_port'] = ~ais_bulkers['EU'].isnull()
ais_bulkers['year'] = ais_bulkers.timestamp.apply(lambda x: x.year, meta=('x', 'int16'))

def update_trip(df):
    df['trip'] = df.groupby('mmsi').in_port.cumsum()
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['trip'] = 'int'


ais_bulkers.map_partitions(update_trip, meta=meta_dict).rename(columns = {'ISO_3digit': 'origin'}).to_parquet(
    os.path.join(filepath, 'ais_bulkers_trips'),
    append = False,
    overwrite = True,
    engine = 'pyarrow')

ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips'))
ais_bulkers.head()


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
ais_bulkers_EU['work'] = ais_bulkers_EU['speed']**2 * ais_bulkers_EU['distance']


# Now we have parquet files with new column called 'phase'
# Get ready to join with wfr_bulkers_calcs to get hourly fuel consumption
# List of columns to keep
selected_columns = ['mmsi', 'ME_W_ref', 'W_component', 
                    'Dwt', 'ME_SFC_base', 'AE_SFC_base',
                    'Boiler_SFC_base', 'Draught..m.', 
                    'Service.Speed..knots.'] 

# Read the csv file into a Dask DataFrame with only the selected columns
wfr_bulkers = dd.read_csv(os.path.join(datapath, 'bulkers_WFR_calcs.csv'), usecols=selected_columns)


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

def pd_diff_haversine(df):
    df_lag = df.shift(1)
    lat_diff = np.radians(df['latitude'] - df_lag['latitude'])
    lng_diff = np.radians(df['longitude'] - df_lag['longitude'])
    d = (np.sin(lat_diff * 0.5) ** 2 +
         np.cos(np.radians(df_lag['latitude'])) *
         np.cos(np.radians(df['latitude'])) *
         np.sin(lng_diff * 0.5) ** 2)
    haversine_formula = 2 * 6371.0088 * np.arcsin(np.sqrt(d))
    return haversine_formula

def calculate_longest_distance(group):
    distances = pd_diff_haversine(group)
    # Ignore the first distance which is NaN due to the shift
    return distances.iloc[1:].max()


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

# Calculate percentage of ships that are sitting at the port
port_pct = (
    joined_df[joined_df['phase'] == 'Anchored']
    .groupby(['mmsi', 'year'])
    .size()
    .divide(joined_df.groupby(['mmsi', 'year']).size())
).compute().rename('port_pct')

# Calculate longest distance where 'interpolated' is False
longest_jump = (
    joined_df[joined_df['interpolated'] == False]
    .groupby(['mmsi', 'year'])
    .apply(lambda df: pd_diff_haversine(df).iloc[1:].max(), meta=('distance', 'f8'))
).compute().rename('longest_jump')

# Calculate proportion of missing hourly data for 'operational phase' is 'sea'
miss_pct_sea = (
    joined_df[joined_df['phase'] == 'Sea']
    .groupby(['mmsi', 'year'])
    ['interpolated']
    .apply(lambda s: (s == True).sum() / s.size, meta=('miss_pct_sea', 'f8'))
).compute().rename('miss_pct_sea')

# Add the calculated features to the yearly_stats DataFrame directly
yearly_stats['port_pct'] = port_pct
yearly_stats['longest_jump'] = longest_jump
yearly_stats['miss_pct_sea'] = miss_pct_sea

#%%
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))
