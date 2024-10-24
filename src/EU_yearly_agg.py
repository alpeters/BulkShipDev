"""
Merge portcall locations (EU or not) to AIS trips to identify trips into and out of EU
and aggregate yearly fuel consumption
Power consumption calculations are based on International Maritime Organization's Fourth Greenhouse Gas Study (Faber et al., 2020),
referenced as 'IMO4' in the code.
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, ais_bulkers_trips_EU.parquet, ais_bulkers_trips_EU_power.parquet, AIS_..._EU_yearly_stats.csv
Runtime: 4m48 + 8m59 + 1m43 + 8m? + 1m15
"""

#%%
import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'src/data'
# datapath = '/media/apeters/Extreme SSD/Working'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''
filename = 'portcalls_' + callvariant + '_EU'

#%%###### Portcall and trip assignment ######
# Load data and format for merging with portcalls
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_speed'))
# ais_bulkers = ais_bulkers.partitions[0:5]
# ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_speed')).get_partition(0)
# ais_bulkers['timestamp'] = ais_bulkers['timestamp'].dt.tz_localize(None) #TODO: why is this necessary?
# ais_bulkers['timestamp'] = ais_bulkers['timestamp'].astype('datetime64[ns]')
ais_bulkers.head()

portcalls = pd.read_csv(
    os.path.join(datapath, 'pot' + filename + '.csv'),
    usecols = ['imo', 'pot_trip', 'EU', 'ISO_3digit'],
    dtype = {'imo' : 'int32',
             'pot_trip': 'int16',
             'EU': 'int8',
             'ISO_3digit': 'str'}
    )
portcalls = (
    portcalls
    .set_index('imo')
    .sort_values(['imo', 'pot_trip'])
    )
portcalls['pot_in_port'] = True

# Merge portcalls to AIS data and save to file
ais_bulkers.merge(portcalls,
                  how = 'left',
                  on = ['imo', 'pot_trip', 'pot_in_port']).to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'), 
            append = False, 
            overwrite = True,
            engine = 'fastparquet')

#%% ############## Assign trip numbers ##############
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'))
# ais_bulkers.head()

# Label an observation as 'in port' if it has matched to any portcall
ais_bulkers['in_port'] = ~ais_bulkers['EU'].isnull()
ais_bulkers['year'] = ais_bulkers.timestamp.apply(lambda x: x.year, meta=('x', 'int16'))

# Assign a sequential trip number for observations between portcalls
def update_trip(df):
    df['trip'] = df.groupby('imo').in_port.cumsum()
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['trip'] = 'int'

ais_bulkers.map_partitions(update_trip, meta=meta_dict).rename(columns = {'ISO_3digit': 'origin'}).to_parquet(
    os.path.join(datapath, 'AIS', 'ais_bulkers_trips'),
    append = False,
    overwrite = True,
    engine = 'fastparquet')

#%% ###### Subset EU trips ######
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips'))
ais_bulkers.head()


# Save dataframe of EU portcalls
portcalls = (
    ais_bulkers
    .loc[ais_bulkers['in_port'] == True,
        ['trip', 'latitude', 'longitude', 'EU', 'origin']]
    .reset_index(drop = False)
    .compute())

portcalls.to_csv(os.path.join(datapath, filename + '.csv'))

# portcalls = pd.read_csv(os.path.join(datapath, filename + '.csv'))

# Deal with observations that occur before the first portcall:
# Add in trip 0 for each ship (these don't appear in portcalls because first portcall assigned to trip 1)
# Assume trip 0 was not from EU port (but may be to one)
trip0 = pd.DataFrame({'imo': portcalls.imo.unique(),
                      'trip': 0,
                      'EU': False,
                      'origin': np.NAN})
portcalls = pd.concat([portcalls, trip0])
portcalls.sort_values(by = ['imo', 'trip'], inplace = True)
# EU trips include travel from previous portcalls
portcalls['EU'] = portcalls.EU == 1 # assumes NA are not in EU
portcalls['prev'] = portcalls.groupby('imo').EU.shift(-1, fill_value = False)
EU_trips = portcalls[portcalls.EU | portcalls.prev]
EU_trips = EU_trips[['imo', 'trip']].set_index('imo')

# Create dataframe of just EU trips

def subset_EU_trips(part):
    """ Subsets ais_bulkers to just the EU trips"""
    return part.merge(EU_trips,
               how = 'right',
               on = ['imo', 'trip'])

# ais_bulkers_EU = 
ais_bulkers.map_partitions(subset_EU_trips, meta = ais_bulkers).to_parquet(
    os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU'),
    append = False,
    overwrite = True,
    engine = 'fastparquet')


# ais_bulkers_EU.head()
#%% Load ais_bulkers_trips_EU
ais_bulkers_EU = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU'))
# ais_bulkers_EU = ais_bulkers_EU.partitions[0:5]

###### Fuel consumption calculations (for EU trips) ######
# Merge WFR ship specifications that are required for fuel consumption calculations
selected_columns = ['IMO.Number', 'ME_W_ref', 'W_component', 
                    'Dwt', 'ME_SFC_base', 'AE_SFC_base',
                    'Boiler_SFC_base', 'Draught..m.', 
                    'Service.Speed..knots.'] 
wfr_bulkers = dd.read_csv(os.path.join(datapath, 'bulkers_WFR_calcs.csv'), usecols=selected_columns)
wfr_bulkers = wfr_bulkers.rename(columns = {'IMO.Number': 'imo'}).dropna(subset=['imo'])
wfr_bulkers = dd.concat([wfr_bulkers, wfr_bulkers.assign(imo = wfr_bulkers['imo']*-1)])
wfr_bulkers = wfr_bulkers.set_index('imo').compute()

# Load lookup table for aux and boiler power
W_aux_table = pd.read_csv(os.path.join(datapath, 'AE_Boiler_Power_table.csv'),
                          dtype = {'ME_W_bin': 'float',
                                   'Dwt_bin': 'float',
                                   'phase': 'str',
                                   'AE_W': 'float',
                                   'Boiler_W': 'float'})

def join_wfr(df, wfr):
    # Calculate travel work
    df['work'] = df['speed']**2 * df['distance']
    joined_df = df.merge(wfr, on = 'imo', how='inner')
    joined_df = joined_df.dropna(subset=['speed', 'draught'])
    # TODO: document how many get dropped here
    return joined_df

# Define functions for fuel consumption calculations
def calculate_hourly_power(df, table):
    """
    Calculate hourly power demand for main engine, auxiliary engine, and boiler.
    Auxiliary and boiler powers according to IMO4 p68.
    We employ Table 17, except we don't identify 'at berth' and instead assign 'anchored' values to these.

    Args:
        df (DataFrame): Dask DataFrame with columns 'ME_W_ref', 'W_component', 'draught', 'speed', 'phase', 'Dwt'
    """
    # calculate hourly main engine power (IMO4 Eq 8)
    df['ME_W'] = df['ME_W_ref'] * df['W_component'] * (df['draught']**0.66) * (df['speed']**3)
    # Assign auxiliary engine and boiler power based on IMO rules and lookup table
    dwt_bins = [0, 9999, 34999, 59999, 99999, 199999, np.inf]
    df['ME_W_bin'] = pd.cut(df['ME_W'], bins=[0, 150, 500, np.inf], labels=np.arange(1.0,4.0))
    df['Dwt_bin'] = pd.cut(df['Dwt'], bins=dwt_bins, labels=np.arange(1.0, 7.0))
    # df['ME_W_column_name'] = df['column_name'].astype('category').cat.codes
    df.reset_index(inplace=True)
    df = df.merge(table, on = ['ME_W_bin', 'Dwt_bin', 'phase'], how='left')
    df.set_index('imo', inplace=True)
    # print(df.index.name)
    # print(df.columns)
    df['AE_W'] = df.apply(lambda row: row['ME_W'] * 0.05 if np.isnan(row['AE_W']) else row['AE_W'], axis=1)
    df = df.drop(columns=['ME_W_bin', 'Dwt_bin'])
    
    return df

## test calculate_hourly_power
# test_table = pd.read_csv(os.path.join(datapath, 'AE_Boiler_Power_test_table.csv'))
# calculate_hourly_power(test_table, W_aux_table)

def process_partitions(part, wfr, table):
    part = join_wfr(part, wfr)
    part = calculate_hourly_power(part, table)
    return part

meta_dict = ais_bulkers_EU.dtypes.to_dict()
meta_dict['work'] = 'float'
meta_dict['Dwt'] = 'float'
meta_dict['Draught..m.'] = 'float'
meta_dict['Service.Speed..knots.'] = 'float'
meta_dict['W_component'] = 'float'
meta_dict['ME_W_ref'] = 'float'
meta_dict['ME_SFC_base'] = 'float'
meta_dict['AE_SFC_base'] = 'float'
meta_dict['Boiler_SFC_base'] = 'float'
# meta_dict['imo'] = 'int'
meta_dict['ME_W'] = 'float'
meta_dict['AE_W'] = 'float'
meta_dict['Boiler_W'] = 'float'


(
    ais_bulkers_EU
    .map_partitions(process_partitions,
                    wfr_bulkers,
                    W_aux_table,
                    meta=meta_dict,
                    align_dataframes=False)
    .to_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU_power'),
                append = False,
                overwrite = True,
                engine = 'fastparquet')
)


# joined_df.head()
################################# RUN FROM HERE TO ADD NEW STATS
#%% Hourly fuel consumption for main engine (IMO4 Eqns 8 and 10)
joined_df = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU_power'))

def calculate_FC_ME(ME_W_ref, W_component, draught, speed, ME_SFC_base):
    """
    Calculate hourly main engine fuel consumption.
    Implements SFC empirical equation from IMO4 Eqn 10.
    Take 'load' as instantaneous *power* divided by reference *power*. (IMO4 Eqn 8 divided by W_ref)

    Args:
        ME_W_ref (float): Reference power of the main engine from WFR
        W_component (float): Fixed ship-specific component of power demanded calculted from specs in WFR
        draught (float): Instantaneous ship draught
        speed (float): Instantaneous ship speed
        ME_SFC_base (float): Base specific fuel consumption calculated from WFR

    Returns:
        Float: Instantaneous specific fuel consumption for the main engine
    """
    load = W_component * draught**0.66 * speed**3 
    return ME_W_ref * W_component * draught**0.66 * speed**3 * ME_SFC_base * (0.455 * load**2 - 0.710 * load + 1.280)

joined_df['FC_ME'] = calculate_FC_ME(joined_df['ME_W_ref'], 
                                     joined_df['W_component'], 
                                     joined_df['draught'], 
                                     joined_df['speed'], 
                                     joined_df['ME_SFC_base'])

# Hourly fuel consumption for auxiliary engine and boiler (IMO4 Eqn 11)
joined_df['FC_AE'] = joined_df['AE_W']*joined_df['AE_SFC_base'] 
joined_df['FC_Boiler'] = joined_df['Boiler_W']*joined_df['Boiler_SFC_base']

# Total hourly fuel consumption (sum of main engine, auxiliary engine, and boiler)
joined_df['FC'] = joined_df['FC_ME'] + joined_df['FC_AE'] + joined_df['FC_Boiler']

# Additional quantities to use as predictive variables for fuel consumption
## Instantaneous component of power demanded (t_i^m*v_i^n in IMO4 Eqn 8)
joined_df['t_m_times_v_n'] = joined_df['draught']**0.66 * joined_df['speed']**3
joined_df['t_m_times_v_n_squared'] = joined_df['t_m_times_v_n']**2
joined_df['t_m_times_v_n_cubed'] = joined_df['t_m_times_v_n']**3
## Instantaneous draft relative to the reference draft, with and without exponent ((t_i/t_ref) in IMO4 Eqn 8)
joined_df['t_over_t_ref_with_m'] = joined_df['draught']**0.66 / joined_df['Draught..m.']**0.66
joined_df['t_over_t_ref_without_m'] = joined_df['draught'] / joined_df['Draught..m.']
## Instantaneous speed relative to the reference speed, with and without exponent ((v_i/v_ref) in IMO4 Eqn 8)
joined_df['v_over_v_ref_with_n'] = joined_df['speed']**3 / joined_df['Service.Speed..knots.']**3
joined_df['v_over_v_ref_without_n'] = joined_df['speed'] / joined_df['Service.Speed..knots.']

## Draught, speed terms of Admiralty equation
## aka ship-specific terms (without C) of relative power W_i/W_ref
joined_df['rel_power_ship'] = joined_df['t_over_t_ref_with_m'] * joined_df['v_over_v_ref_with_n']
joined_df['rel_power_ship_squared'] = joined_df['rel_power_ship']**2
joined_df['rel_power_ship_cubed'] = joined_df['rel_power_ship']**3


joined_df.head()

###### Summary statistics aggregated at the ship-year level ######

# Define an aggregation function to get the number of unique values
# This will be used to get the number of unique trips
# refer to https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),)


yearly_stats = (
    joined_df
    .groupby(['imo', 'year'])
    .agg({
        'timestamp': ['count'],
        'distance': ['sum'],
        'work': ['sum'],
        'trip': nunique,
        'interpolated' : ['sum'],
        'W_component': ['first'],
        'ME_W_ref': ['first'],
        'ME_SFC_base': ['first', 'mean'],
        't_m_times_v_n': ['sum'],
        't_m_times_v_n_squared': ['sum'],
        't_m_times_v_n_cubed': ['sum'],
        't_over_t_ref_with_m': ['sum'],
        't_over_t_ref_without_m': ['sum'],
        'v_over_v_ref_with_n': ['sum'],
        'v_over_v_ref_without_n': ['sum'],
        'rel_power_ship': ['sum'],
        'rel_power_ship_squared': ['sum'],
        'rel_power_ship_cubed': ['sum'],
        'FC': ['sum'],
        'FC_ME': ['sum'],
        'speed': ['mean'],
        'draught': ['mean']
        })
    .compute())

# Calculate percentage of observations in which ships at port
yearly_stats['port_frac'] = (
    joined_df[joined_df['phase'] == 'Anchored']
    .groupby(['imo', 'year'])
    .size()
    .divide(joined_df.groupby(['imo', 'year']).size())
).compute().rename('port_frac')

# Calculate longest jump (distance) between observed data points
def observed_distances(df):
    """
    Calculates the distance (haversine) and time interval with respect to the previous row
    
    Args:
        df (pd.DataFrame): input dataframe with columns latitude, longitude, timestamp

    Returns:
        vector?: haversine distances
    """
    df = df[~df['interpolated']]
    lat_lag = df.latitude.shift(1)
    lng_lag = df.longitude.shift(1)
    lat_diff = np.radians(df['latitude'] - lat_lag)
    lng_diff = np.radians(df['longitude'] - lng_lag)
    d = (np.sin(lat_diff * 0.5) ** 2 +
         np.cos(np.radians(lat_lag)) *
         np.cos(np.radians(df['latitude'])) *
         np.sin(lng_diff * 0.5) ** 2)
    df['distance'] = 2 * 6371.0088 * np.arcsin(np.sqrt(d))
    return df

yearly_stats['longest_jump'] = (
    joined_df
    .map_partitions(lambda part: part.groupby(['imo', 'year']).apply(lambda df: observed_distances(df).distance.max()))).compute()

# Calculate total jump distance
def observed_large_distances(df):
    df = df[~df['interpolated']]
    lat_lag = df.latitude.shift(1)
    lng_lag = df.longitude.shift(1)
    lat_diff = np.radians(df['latitude'] - lat_lag)
    lng_diff = np.radians(df['longitude'] - lng_lag)
    d = (np.sin(lat_diff * 0.5) ** 2 +
         np.cos(np.radians(lat_lag)) *
         np.cos(np.radians(df['latitude'])) *
         np.sin(lng_diff * 0.5) ** 2)
    df['distance'] = 2 * 6371.0088 * np.arcsin(np.sqrt(d))
    df = df[df['implied_speed'] > 25]
    return df

yearly_stats['total_jump_distance'] = (
    joined_df
    .map_partitions(lambda part: part.groupby(['imo', 'year']).apply(lambda df: observed_large_distances(df).distance.sum()))).compute()

# Flatten the multi-index columns
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]

# Calculate proportion of missing hourly data for 'operational phase' is 'sea'
interpolated_sea = (
    joined_df[joined_df['phase'] == 'Sea']
    .groupby(['imo', 'year'])
    .interpolated
    .agg(['sum', 'count'])
).compute()
missing_frac_sea = (interpolated_sea['sum'] / interpolated_sea['count']).rename('missing_frac_sea')

# Calculate average speed and variance of speed while at sea
speed_sea = (
    joined_df[joined_df['phase'] == 'Sea']
    .groupby(['imo', 'year'])
    .speed
    .agg(['mean', 'var'])
).compute()
speed_sea.columns = ['speed_sea_' + col for col in speed_sea.columns.values]

yearly_stats_flat = yearly_stats_flat.join(missing_frac_sea, on = ['imo', 'year'])
yearly_stats_flat = yearly_stats_flat.join(speed_sea, on = ['imo', 'year'])
yearly_stats_flat = yearly_stats_flat.rename(columns={
    'port_frac_':'port_frac',
    'longest_jump_':'longest_jump',
    'total_jump_distance_':'total_jump_distance',})
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))

#%%