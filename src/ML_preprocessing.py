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
import seaborn as sns
import matplotlib.pyplot as plt

datapath = 'src/data'
trackeddatapath = 'src/tracked_data'
callvariant = 'speed' 
EUvariant = '_EEZ' 
filename = 'portcalls_' + callvariant + '_EU'

#%%
bulkers_wfr_df = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))['bulkers_df']
mrv_df = pyreadr.read_r(os.path.join(datapath, 'MRV.Rda'))['MRV_df']
ais_eu_df = pd.read_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))

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
final_df['distance_difference'] = final_df['MRV.EU.distance'] - final_df['distance_sum']
final_df['within_tol_abs'] = abs(final_df['distance_difference']) <= tolerance_abs

## Relative
tolerance_rel = 0.1
final_df['distance_difference_rel'] = final_df['distance_difference']/final_df['MRV.EU.distance'] # this is negative of typical definition but in keeping with residual definition
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

#%% Additional criteria on fraction of distance attributed to jumps
final_df['jump_distance_frac'] = final_df['total_jump_distance'] / final_df['distance_sum']
final_df['within_tol_jumps'] = final_df['jump_distance_frac'] <= 0.003


#%%
# Save separate datasets for absolute and relative tolerance and training and testing
for tol in ['abs', 'rel']:
    for set in ['train', 'test']:
        (
            final_df[final_df['within_tol_jumps'] & final_df['within_tol_' + tol] & (final_df['set'] == set)]
            .drop(columns = ['within_tol_rel', 'within_tol_abs', 'set'])
            .to_csv(os.path.join(trackeddatapath, 'df_ml_' + tol + '_' + set + '.csv'), index=False)
        )

#########################
# Summary stats and plots
#########################

#%% How many MRV observations that we consider to be bulk carriers and what other categories do they fall under in MRV.ship.type?
merged_df['MRV.ship.type'].unique().tolist()
#%% Which categories have we chosen as dry bulk in WFR?
bulkers_wfr_df['Type'].unique().tolist()
#%% Which actually get matched'
types_considered = merged_df['Type'].unique().tolist()

#%% How many active 'bulkers' in WFR?
bulkers_wfr_df['Status'].unique().tolist()
#%%
active_bulkers_df = bulkers_wfr_df[
    bulkers_wfr_df['Demo.Date'].isna() &
    bulkers_wfr_df['Status'].isin(['Active', 'In Service', 'Idle', 'Repairs', 'Storage', 'Laid Up']) &
    bulkers_wfr_df['Built.Year'].notna() &
    bulkers_wfr_df['Type'].isin(types_considered)
    ]

#%% How many considered bulkers in WFR?
obs_counts = merged_df.value_counts(['year']).sort_index().to_frame()
obs_counts['fraction_active'] = obs_counts['count']/active_bulkers_df.shape[0]
obs_counts

#%%
merged_df.value_counts(['year', 'MRV.ship.type', 'Type'])

#%% How many observations in each year after merges?
merged_df.value_counts('year')
#%%
final_df.value_counts('year')

#%% How many within tolerance?
final_df.value_counts(['year', 'within_tol_abs', 'within_tol_jumps'])
#%%
final_df.value_counts(['year', 'within_tol_rel', 'within_tol_jumps'])

#%% How many left?
final_df.loc[final_df['within_tol_abs'] & final_df['within_tol_jumps']].value_counts('year')

#%% MRV counts
mrv_df.groupby(['year'])


#%% Check training set features
final_df.columns[final_df.columns.str.contains('a', case=True)]
        
#%% Check training set outliers
# Absolute
train_abs_df = final_df[final_df['within_tol_abs'] & (final_df['set'] == 'train')].copy()
raw_mean = train_abs_df['residual'].mean()
raw_std = train_abs_df['residual'].std()
outlier_threshold = 3 # number of standard deviations from the mean
#%% set outlier true if residual between lower and upper thresholds, false otherwise
train_abs_df['outlier'] = ~train_abs_df['residual'].between(
    raw_mean - outlier_threshold * raw_std,
    raw_mean + outlier_threshold * raw_std,
    inclusive='neither')
train_abs_df['outlier'].value_counts()

#%%
train_abs_df.loc[train_abs_df['outlier'], ['mmsi', 'year', 'IMO.Number']].sort_values(['mmsi', 'year']).to_csv(os.path.join(trackeddatapath, 'outliers_train_abs.csv'), index=False)

#%%
train_rel_df = final_df[final_df['within_tol_rel'] & (final_df['set'] == 'train')].copy()

#%% Are outliers incorrectly matched ships?
train_abs_df[['outlier', 'MRV.ship.type', 'Type']].value_counts()


#### JUMP DISTANCE #####
# %%
sns.histplot(data=train_abs_df.loc[train_abs_df['total_jump_distance'] > 10], x='total_jump_distance', hue='outlier', bins=100)
plt.title('Total jump distance')

#%%
sns.histplot(data=train_abs_df, x='distance_sum', hue='outlier', bins=100)
plt.title('Total distance')

# %%
train_abs_df['jump_frac'] = train_abs_df['total_jump_distance']/train_abs_df['distance_sum']
sns.histplot(data=train_abs_df.loc[train_abs_df['jump_frac'] > 0.00001], x='jump_frac', hue='outlier', bins=100)
plt.title('Total jump distance as fraction of total distance')

#%%