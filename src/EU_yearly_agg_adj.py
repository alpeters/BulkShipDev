#%%
import sys, os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster

datapath = 'data'
callvariant = 'speed' #'heading'
EUvariant = '_EEZ' #''


adj_pcts = np.array([-20, -15, -10, -5, 5, 10, 15, 20])
# adj_pcts = np.array([-10, 10])
which_speeds = np.array(['sea'])
#%%

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


def calc_sum_stats(datapath, callvariant, EUvariant, adj_pct, which_speed):
    #%% Hourly fuel consumption for main engine (IMO4 Eqns 8 and 10)
    joined_df = dd.read_parquet(os.path.join(datapath, 'AIS', 'ais_bulkers_trips_EU_power'))

    # Adjust speed, dist
    if which_speed == 'all':
        joined_df['speed'] *= (adj_pct/100 + 1)
        joined_df['distance'] *= (adj_pct/100 + 1)
    elif which_speed == 'sea':
        joined_df['speed'] = joined_df['speed'].mask(joined_df['phase'] == 'Sea', joined_df['speed'] * (adj_pct/100 + 1))
        joined_df['distance'] = joined_df['distance'].mask(joined_df['phase'] == 'Sea', joined_df['distance'] * (adj_pct/100 + 1))

    joined_df['work'] = joined_df['speed']**2 * joined_df['distance']

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

    ## Relative power terms for structural
    joined_df['W_rel'] = joined_df['t_m_times_v_n'] * joined_df['W_component']
    joined_df['W_rel_squared'] = joined_df['W_rel']**2
    joined_df['W_rel_linspline'] = joined_df['W_rel_squared'] - 0.78 * joined_df['W_rel']
    joined_df['W_rel_quadspline'] = joined_df['W_rel'] * (joined_df['W_rel'] - 0.78)**2

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
    yearly_stats['longest_jump'] = (
        joined_df
        .map_partitions(lambda part: part.groupby(['imo', 'year']).apply(lambda df: observed_distances(df).distance.max()))).compute()

    # Calculate total jump distance
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
        .agg(['count', 'mean', 'var'])
    ).compute()
    speed_sea.columns = ['speed_sea_' + col for col in speed_sea.columns.values]

    # Calculate spline terms for structural estimation with minimum constrained
    spline_terms = (
        joined_df[joined_df['W_rel'] > 0.78]
        .groupby(['imo', 'year'])
        .agg({
            'W_rel_linspline': ['sum'],
            'W_rel_quadspline': ['sum']
        })
        .compute()
    )
    spline_terms.columns = ['_'.join(col).strip() for col in spline_terms.columns.values]

    yearly_stats_flat = yearly_stats_flat.join(missing_frac_sea, on = ['imo', 'year'])
    yearly_stats_flat = yearly_stats_flat.join(speed_sea, on = ['imo', 'year'])
    yearly_stats_flat = yearly_stats_flat.join(spline_terms, on = ['imo', 'year'])
    # set to zero if there were no observations of W_rel greater than the minimum constraint
    yearly_stats_flat[['W_rel_linspline_sum', 'W_rel_quadspline_sum']] = yearly_stats_flat[['W_rel_linspline_sum', 'W_rel_quadspline_sum']].fillna(0)
    yearly_stats_flat = yearly_stats_flat.rename(columns={
        'port_frac_':'port_frac',
        'longest_jump_':'longest_jump',
        'total_jump_distance_':'total_jump_distance',})
    yearly_stats_flat.to_csv(os.path.join(
        datapath,
        'AIS_' + callvariant + EUvariant + '_EU_yearly_stats_bulkers_adj' + str(adj_pct) + '_' + which_speed + '.csv'
    ))

#%%
if __name__ == '__main__':
    for adj_pct in adj_pcts:
        for which_speed in which_speeds:
            print('Adjusting speed and distance by ' + str(adj_pct) + '% for ' + str(which_speed) + ' speeds')
            calc_sum_stats(datapath, callvariant, EUvariant, adj_pct, which_speed)