"""
Match MRV & WFR data with AIS ship-level aggregated yearly fuel consumption
Input(s): bulkers_WFR.csv, MRV.rda, AIS_..._EU_yearly_stats_bulkers.csv
Output(s): df_ml.csv
"""
#%%
import os 
import pyreadr
import numpy as np
import pandas as pd
import pyreadr

print(os.getcwd())
datapath = 'data'
trackeddatapath = 'tracked_data'
callvariant = 'speed' 
EUvariant = '_EEZ' 
filename = 'portcalls_' + callvariant + '_EU'

adj_pcts = np.array([-20, -10, -15, -5, 5, 10, 15, 20])
which_speeds = np.array(['sea'])
tol = 'abs'

#%%
bulkers_wfr_df = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))['bulkers_df']
mrv_df = pyreadr.read_r(os.path.join(datapath, 'MRV.Rda'))['MRV_df']


#%% Select and rename MRV data to ensure not used for prediction models
mrv_df = mrv_df.loc[:, ['imo.number', 'reporting.period', 'EU.distance', 'total.fc', 'a', 'b', 'c', 'd', 'verifier.country', 'verifier.name', 'ship.type']]
mrv_df = mrv_df.rename(columns={
    'imo.number':'IMO.Number',
    'reporting.period': 'year',
    'a': 'MRV.method.a',
    'b': 'MRV.method.b',
    'c': 'MRV.method.c',
    'd': 'MRV.method.d',
    'EU.distance': 'MRV.EU.distance',
    'verifier.country': 'MRV.verifier.country',
    'verifier.name': 'MRV.verifier.name',
    'ship.type': 'MRV.ship.type'
    })

#%%
bulkers_wfr_df = bulkers_wfr_df.rename(columns={'MMSI': 'mmsi'})
bulkers_wfr_df.dropna(subset=['IMO.Number'], inplace=True)
mrv_df.dropna(subset=['IMO.Number'], inplace=True)
bulkers_wfr_df['IMO.Number'] = bulkers_wfr_df['IMO.Number'].astype(int)
mrv_df['IMO.Number'] = mrv_df['IMO.Number'].astype(int)

#%% Join bulkers_wfr_df with mrv_df on 'IMO.Number' and 'Year'
merged_df = pd.merge(bulkers_wfr_df, mrv_df, how='inner', on='IMO.Number')
merged_df['year'] = merged_df['year'].astype('int64')
merged_df = merged_df.rename(columns={'IMO.Number': 'imo'})
merged_df = pd.concat([merged_df, merged_df.assign(imo = merged_df['imo']*-1)])

# Get matched ship IMOs
distance_matched_df = pd.read_csv(os.path.join(trackeddatapath, "df_ml_" + tol + "_all.csv"), usecols=['imo', 'year'])

for adj_pct in adj_pcts:
    for which_speed in which_speeds:
        ais_eu_df = pd.read_csv(os.path.join(
            datapath,
            'AIS_' + callvariant + EUvariant + '_EU_yearly_stats_bulkers_adj' + str(adj_pct) + '_' + which_speed + '.csv'
        ))

        # Filter the matched ship-years using a filter join
        ais_eu_df = pd.merge(ais_eu_df, distance_matched_df, how='inner', on=['imo', 'year'])

        #%% Merge the resulting df with ais_eu_df on 'MMSI' and 'Year'
        final_df = pd.merge(merged_df, ais_eu_df, how='inner', on=['imo', 'year'])

        #%% Rename fuel consumption columns for clarity 
        final_df = final_df.rename(columns={
            'total.fc': 'report_fc',
            'FC_sum': 'cal_fc',
            'FC_ME_sum': 'cal_fcme'
        })

        #%% Calculate variables
        final_df['age'] = final_df['year'] - final_df['Built.Year']
        final_df['cal_fc'] = final_df['cal_fc'] / 1E6 # scale to same units at report_fc
        final_df['cal_fcme'] = final_df['cal_fcme'] / 1E6 # scale to same units at report_fc
        final_df['residual'] = np.log1p(final_df['report_fc'].values) - np.log1p(final_df['cal_fc'].values)
        # Note: We define residual as reported minus calculated
        final_df['cal_fc_auxbo'] = final_df['cal_fc'] - final_df['cal_fcme']
        final_df['report_fcme'] = final_df['report_fc'] - final_df['cal_fc_auxbo']
        # construct an estimate of the reported main engine fuel consumption by subtracting calculated FC from aux and boiler
        final_df['log_report_fc'] = np.log1p(final_df['report_fc'].values)
        final_df['log_report_fcme'] = np.log1p(final_df['report_fcme'].values)
        final_df['log_cal_fc'] = np.log1p(final_df['cal_fc'].values)
        final_df['log_cal_fcme'] = np.log1p(final_df['cal_fcme'].values)

        #%% Additional criteria on fraction of distance attributed to jumps
        final_df['jump_distance_frac'] = final_df['total_jump_distance'] / final_df['distance_sum']
        final_df['within_tol_jumps'] = final_df['jump_distance_frac'] <= 0.003

        final_df.to_csv(os.path.join(trackeddatapath, 'df_ml_' + tol + '_all_adj' + str(adj_pct) + '_' + which_speed + '.csv'), index=False)