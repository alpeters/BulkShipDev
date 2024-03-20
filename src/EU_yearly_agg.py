"""
Merge portcall locations (EU or not) to AIS trips to identify trips into and out of EU
and aggregate yearly fuel consumption
Power consumption calculations are based on International Maritime Organization's Fourth Greenhouse Gas Study (Faber et al., 2020),
referenced as 'IMO4' in the code.
Input(s): portcalls_'callvariant'_EU.csv, ais_bulkers_potportcalls_'callvariant'.parquet
Output(s): ais_bulkers_pottrips.parquet, ais_bulkers_trips.parquet, ais_bulkers_trips_EU.parquet, ais_bulkers_trips_EU_power.parquet, AIS_..._EU_yearly_stats.csv
Runtime: 4m48 + 8m59 + 1m43 + 2m17 + 1m15
"""

#%%
import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

# datapath = 'src/data'
datapath = '/media/apeters/Extreme SSD/Working'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''
filename = 'portcalls_' + callvariant + '_EU'

#%%###### Portcall and trip assignment ######
# Load data and format for merging with portcalls
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_speed'))
# ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_potportcalls_speed')).get_partition(0)
ais_bulkers['timestamp'] = ais_bulkers['timestamp'].dt.tz_localize(None) #TODO: why is this necessary?
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

# Merge portcalls to AIS data and save to file
ais_bulkers.merge(portcalls,
                  how = 'left',
                  on = ['mmsi', 'pot_trip', 'pot_in_port']).to_parquet(
            os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'), 
            append = False, 
            overwrite = True)

#%% ############## Assign trip numbers ##############
ais_bulkers = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_pottrips'))
# ais_bulkers.head()

# Label an observation as 'in port' if it has matched to any portcall
ais_bulkers['in_port'] = ~ais_bulkers['EU'].isnull()
ais_bulkers['year'] = ais_bulkers.timestamp.apply(lambda x: x.year, meta=('x', 'int16'))

# Assign a sequential trip number for observations between portcalls
def update_trip(df):
    df['trip'] = df.groupby('mmsi').in_port.cumsum()
    return df

meta_dict = ais_bulkers.dtypes.to_dict()
meta_dict['trip'] = 'int'

ais_bulkers.map_partitions(update_trip, meta=meta_dict).rename(columns = {'ISO_3digit': 'origin'}).to_parquet(
    os.path.join(datapath, 'AIS', 'ais_bulkers_trips'),
    append = False,
    overwrite = True,
    engine = 'pyarrow')

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

# Create dataframe of just EU trips

def subset_EU_trips(part):
    """ Subsets ais_bulkers to just the EU trips"""
    return part.merge(EU_trips,
               how = 'right',
               on = ['mmsi', 'trip'])

# ais_bulkers_EU = 
ais_bulkers.map_partitions(subset_EU_trips, meta = ais_bulkers).to_parquet(
    os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU'),
    append = False,
    overwrite = True,
    engine = 'pyarrow')


# ais_bulkers_EU.head()
#%% Load ais_bulkers_trips_EU
ais_bulkers_EU = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU'))

###### Fuel consumption calculations (for EU trips) ######
# Define functions for fuel consumption calculations
def calculate_hourly_power(df):
    """
    Calculate hourly power demand for main engine, auxiliary engine, and boiler.
    Auxiliary and boiler powers according to IMO4 p68.
    We employ Table 17, except we don't identify 'at berth' and instead assign 'anchored' values to these.

    Args:
        df (DataFrame): Dask DataFrame with columns 'ME_W_ref', 'W_component', 'draught', 'speed', 'phase', 'Dwt'
    """
    # calculate hourly main engine power (IMO4 Eq 8)
    df['ME_W'] = df['ME_W_ref'] * df['W_component'] * (df['draught']**0.66) * (df['speed']**3)
    
    # Assign auxiliary engine power and boiler power values as per IMO4 Table 17
    ## conditions refer to ship size categories as per IMO4 Table 17
    conditions = [
        (df['Dwt'] <= 9999),
        ((df['Dwt'] > 9999) & (df['Dwt'] <= 34999)),
        ((df['Dwt'] > 34999) & (df['Dwt'] <= 59999)),
        ((df['Dwt'] > 59999) & (df['Dwt'] <= 99999)),
        ((df['Dwt'] > 99999) & (df['Dwt'] <= 199999)),
        (df['Dwt'] >= 200000)
    ]
    
    ## Order: 'Anchored', 'Manoeuvring', 'Sea'
    phases = ['Anchored', 'Manoeuvring', 'Sea']
    ae_values = [(180, 500, 190), (180, 500, 190), (250, 680, 260), (400, 1100, 410), (400, 1100, 410), (400, 1100, 410)]
    boiler_values = [(70, 60, 0), (70, 60, 0), (130, 120, 0), (260, 240, 0), (260, 240, 0), (260, 240, 0)]
    
    # 3 Cases based on engine power
    ## Case 1: main engine power below 150, both aux and boiler power are zero
    ## Set auxiliary engine power and boiler power as per case 1 for all cases and then modify below for the other cases
    df['AE_W'] = 0
    df['Boiler_W'] = 0
    
    ## Case 2: main engine power is between 150 and 500
    ### auxiliary engine is set to 5% of the main engine power
    df.loc[(df['ME_W'] > 150) & (df['ME_W'] <= 500), 'AE_W'] = df['ME_W'] * 0.05
    ### assign the boiler power based on phase and ship size
    for condition, ae_value, boiler_value in zip(conditions, ae_values, boiler_values):
        for i, phase in enumerate(phases):
            df.loc[(df['ME_W'] > 150) & (df['ME_W'] <= 500) & condition & (df['phase'] == phase), 'Boiler_W'] = boiler_value[i]

    ## Case 3: main engine power > 500, 
    ### assign both power values based on phase and ship size
    for condition, ae_value, boiler_value in zip(conditions, ae_values, boiler_values):
        for i, phase in enumerate(phases):
            df.loc[(df['ME_W'] > 500) & condition & (df['phase'] == phase), 'AE_W'] = ae_value[i]
            df.loc[(df['ME_W'] > 500) & condition & (df['phase'] == phase), 'Boiler_W'] = boiler_value[i]

    return df

# Calculate travel work
ais_bulkers_EU['work'] = ais_bulkers_EU['speed']**2 * ais_bulkers_EU['distance']

# Merge WFR ship specifications that are required for fuel consumption calculations
# List of columns to keep
selected_columns = ['mmsi', 'ME_W_ref', 'W_component', 
                    'Dwt', 'ME_SFC_base', 'AE_SFC_base',
                    'Boiler_SFC_base', 'Draught..m.', 
                    'Service.Speed..knots.'] 

# Read the csv file into a Dask DataFrame with only the selected columns
wfr_bulkers = dd.read_csv(os.path.join(datapath, 'bulkers_WFR_calcs.csv'), usecols=selected_columns)


# Join the Dask DataFrames
joined_df = ais_bulkers_EU.merge(wfr_bulkers, on='mmsi', how='inner')

# joined_df.head()

# drop rows where hourly speed or draught is missing 
joined_df = joined_df.dropna(subset=['speed', 'draught'])
# TODO: document how many get dropped here

meta = joined_df._meta.assign(ME_W=pd.Series(dtype=float),
                              AE_W=pd.Series(dtype=float), 
                              Boiler_W=pd.Series(dtype=float))

joined_df.map_partitions(calculate_hourly_power, meta=meta).to_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU_power'), append = False, overwrite = True, engine = 'pyarrow')

# joined_df.head()

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
## Instantatneous component of power demanded (t_i^m*v_i^n in IMO4 Eqn 8)
joined_df['t_m_times_v_n'] = joined_df['draught']**0.66 * joined_df['speed']**3
## Instantaneous draft relative to the reference draft, with and without exponent ((t_i/t_ref) in IMO4 Eqn 8)
joined_df['t_over_t_ref_with_m'] = joined_df['draught']**0.66 / joined_df['Draught..m.']**0.66
joined_df['t_over_t_ref_without_m'] = joined_df['draught'] / joined_df['Draught..m.']
## Instantaneous speed relative to the reference speed, with and without exponent ((v_i/v_ref) in IMO4 Eqn 8)
joined_df['v_over_v_ref_with_n'] = joined_df['speed']**3 / joined_df['Service.Speed..knots.']**3
joined_df['v_over_v_ref_without_n'] = joined_df['speed'] / joined_df['Service.Speed..knots.']


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
    .groupby(['mmsi', 'year'])
    .agg({
        'timestamp': ['count'],
        'distance': ['sum'],
        'work': ['sum'],
        'trip': nunique,
        'interpolated' : ['sum'],
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

# Calculate percentage of observations in which ships at port
yearly_stats['port_frac'] = (
    joined_df[joined_df['phase'] == 'Anchored']
    .groupby(['mmsi', 'year'])
    .size()
    .divide(joined_df.groupby(['mmsi', 'year']).size())
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
    .map_partitions(lambda part: part.groupby(['mmsi', 'year']).apply(lambda df: observed_distances(df).distance.max()))).compute()


# Flatten the multi-index columns
yearly_stats_flat = yearly_stats.rename(columns = {"invalid_speed": ("invalid", "speed")})
yearly_stats_flat.columns = ['_'.join(col) for col in yearly_stats_flat.columns.values]

# Calculate proportion of missing hourly data for 'operational phase' is 'sea'
interpolated_sea = (
    joined_df[joined_df['phase'] == 'Sea']
    .groupby(['mmsi', 'year'])
    .interpolated
    .agg(['sum', 'count'])
).compute()
missing_frac_sea = (interpolated_sea['sum'] / interpolated_sea['count']).rename('missing_frac_sea')

yearly_stats_flat = yearly_stats_flat.join(missing_frac_sea, on = ['mmsi', 'year'])
yearly_stats_flat = yearly_stats_flat.rename(columns={
    'port_frac_':'port_frac',
    'longest_jump_':'longest_jump'})
yearly_stats_flat.to_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))

#%%