"""
Match MRV & WFR data with AIS ship-level aggregated yearly fuel consumption
Input(s): bulkers_WFR.csv, MRV.rda, AIS_..._EU_yearly_stats.csv
Output(s): df_ml.csv
"""
#%%
import os 
import pyreadr
import numpy as np
import pandas as pd
import pyreadr


datapath = 'src/data'
trackeddatapath = 'src/tracked_data'
callvariant = 'speed' 
EUvariant = '_EEZ' 
filename = 'portcalls_' + callvariant + '_EU'

#%%
bulkers_wfr_df = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))['bulkers_df']
mrv_df = pyreadr.read_r(os.path.join(datapath, 'MRV.Rda'))['MRV_df']
mrv_df = mrv_df.loc[:, ['imo.number', 'reporting.period', 'EU.distance', 'total.fc']]
ais_eu_df = pd.read_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))

#%%
mrv_df.columns = ['IMO.Number' if x == 'imo.number' else 
                  'year' if x == 'reporting.period' else
                  x for x in mrv_df.columns]

bulkers_wfr_df.columns = ['mmsi' if x == 'MMSI' else 
                          x for x in bulkers_wfr_df.columns]

bulkers_wfr_df.dropna(subset=['IMO.Number'], inplace=True)
mrv_df.dropna(subset=['IMO.Number'], inplace=True)

bulkers_wfr_df['IMO.Number'] = bulkers_wfr_df['IMO.Number'].astype(int)
mrv_df['IMO.Number'] = mrv_df['IMO.Number'].astype(int)

#%% Join bulkers_wfr_df with mrv_df on 'IMO.Number' and 'Year'
merged_df = pd.merge(bulkers_wfr_df, mrv_df, how='inner', on='IMO.Number')
merged_df['year'] = merged_df['year'].astype('int64')

#%% Merge the resulting df with ais_eu_df on 'MMSI' and 'Year'
final_df = pd.merge(merged_df, ais_eu_df, how='inner', on=['mmsi', 'year'])

#%% Rename fuel consumption columns for clarity 
final_df = final_df.rename(columns={'total.fc': 'report_fc', 'FC_sum': 'cal_fc'})

#%% Calculate variables
final_df['age'] = final_df['year'] - final_df['Built.Year']
final_df['cal_fc'] = final_df['cal_fc'] / 1E6 # scale to same units at report_fc
final_df['residual'] = np.log1p(final_df['report_fc'].values) - np.log1p(final_df['cal_fc'].values)
# Note: We define residual as reported minus calculated
final_df['log_report_fc'] = np.log1p(final_df['report_fc'].values)
final_df['log_cal_fc'] = np.log1p(final_df['cal_fc'].values)

#%% Identify observations where distance discrepancy is within a certain tolerance
## Absolute
tolerance_abs = 500 
final_df['distance_difference'] = final_df['EU.distance'] - final_df['distance_sum']
final_df['within_tol_abs'] = abs(final_df['distance_difference']) <= tolerance_abs

## Relative
tolerance_rel = 0.1
final_df['distance_difference_rel'] = final_df['distance_difference']/final_df['EU.distance'] # this is negative of typical definition but in keeping with residual definition
final_df['within_tol_rel'] = abs(final_df['distance_difference_rel']) <= tolerance_rel

#%% Label training and testing sets
## Use 2019,2020 for training and 2020 for test
final_df['set'] = ['train' if x in [2019, 2020] else 'test' for x in final_df['year']]

#%% Check
## Overlap of datasets
final_df.groupby('set').value_counts(['within_tol_abs', 'within_tol_rel'])

## Separately
#%%
### Absolute
print('Absolute')
final_df.groupby('set').value_counts(['within_tol_abs'])
#%%
### Relative
print('Relative')
final_df.groupby('set').value_counts(['within_tol_rel'])

#%%
# Save separate datasets for absolute and relative tolerance and training and testing
for tol in ['abs', 'rel']:
    for set in ['train', 'test']:
        (
            final_df[final_df['within_tol_' + tol] & (final_df['set'] == set)]
            .drop(columns = ['within_tol_rel', 'within_tol_abs', 'set'])
            .to_csv(os.path.join(trackeddatapath, 'df_ml_' + tol + '_' + set + '.csv'), index=False)
        )

#%% Check training set outliers
# Absolute
train_abs_df = final_df[final_df['within_tol_abs'] & (final_df['set'] == 'train')].copy()
raw_mean = train_abs_df['residual'].mean()
raw_std = train_abs_df['residual'].std()
outlier_threshold = 3 # number of standard deviations from the mean
#%% set outlier true if residual between lower and upper thresholds, false otherwise
train_abs_df['outlier'] = train_abs_df['residual'].between(
    raw_mean - outlier_threshold * raw_std,
    raw_mean + outlier_threshold * raw_std,
    inclusive='neither')
train_abs_df['outlier'].value_counts()

#%%
train_abs_df.loc[train_abs_df['outlier'], ['mmsi', 'year', 'IMO.Number']].sort_values(['mmsi', 'year'])