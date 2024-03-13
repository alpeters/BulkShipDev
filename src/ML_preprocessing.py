"""
Match MRV & WFR data with AIS ship-level aggregated yearly fuel consumption
Input(s): bulkers_WFR.csv, MRV.rda, AIS_..._EU_yearly_stats.csv
Output(s): df_ml.csv
"""

import os 
import pyreadr
import numpy as np
import pandas as pd
import pyreadr


datapath = 'src/data'
callvariant = 'speed' 
EUvariant = '_EEZ' 
filename = 'portcalls_' + callvariant + '_EU'


bulkers_wfr_df = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))['bulkers_df']
mrv_df = pyreadr.read_r(os.path.join(datapath, 'MRV.Rda'))['MRV_df']
mrv_df = mrv_df.loc[:, ['imo.number', 'reporting.period', 'EU.distance', 'total.fc']]
ais_eu_df = pd.read_csv(os.path.join(datapath, 'AIS_' + callvariant + EUvariant + '_EU_yearly_stats.csv'))


mrv_df.columns = ['IMO.Number' if x == 'imo.number' else 
                  'year' if x == 'reporting.period' else
                  x for x in mrv_df.columns]

bulkers_wfr_df.columns = ['mmsi' if x == 'MMSI' else 
                          x for x in bulkers_wfr_df.columns]

bulkers_wfr_df.dropna(subset=['IMO.Number'], inplace=True)
mrv_df.dropna(subset=['IMO.Number'], inplace=True)

bulkers_wfr_df['IMO.Number'] = bulkers_wfr_df['IMO.Number'].astype(int)
mrv_df['IMO.Number'] = mrv_df['IMO.Number'].astype(int)

# First, join bulkers_wfr_df with mrv_df on 'IMO.Number' and 'Year'
merged_df = pd.merge(bulkers_wfr_df, mrv_df, how='inner', on='IMO.Number')
merged_df['year'] = merged_df['year'].astype('int64')

# Now, merge the resulting df with ais_eu_df on 'MMSI' and 'Year'
final_df = pd.merge(merged_df, ais_eu_df, how='inner', on=['mmsi', 'year'])
final_df['age'] = final_df['year'] - final_df['Built.Year']
final_df['FC_sum'] = final_df['FC_sum'] / 1E6
final_df['residual'] = np.log1p(final_df['total.fc'].values) - np.log1p(final_df['FC_sum'].values)
# We define residual as reported minus calculated

# Only keep observations where the bias between reported distance and calculated distance is within our tolerance
tolerance = 500 
final_df['distance_difference'] = final_df['EU.distance'] - final_df['distance_sum']
final_df = final_df[abs(final_df['distance_difference']) <= tolerance]

# Rename columns for clarity 
final_df = final_df.rename(columns={'total.fc': 'report_fc', 'FC_sum': 'cal_fc'})

final_df['log_report_fc'] = np.log1p(final_df['report_fc'].values)
final_df['log_cal_fc'] = np.log1p(final_df['cal_fc'].values)
final_df.head()

# Save the final dataset
final_df.to_csv(os.path.join(datapath, 'df_ml.csv'), index=False)