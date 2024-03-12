#%%

import pandas as pandas
import random


#%% Create pandas df with speed column with 10 random values between 0 and 10
gdf = pandas.DataFrame({'speed': [random.uniform(0, 10) for i in range(10)]})
gdf = gdf.sort_values(by='speed')
#%%
phase_conditions = [
    (gdf['speed'] <= 3, 'Anchored'),
    ((gdf['speed'] >= 4) & (gdf['speed'] <= 5), 'Manoeuvring'),
    (gdf['speed'] > 5, 'Sea')
]

#%%
for condition, phase in phase_conditions:
    gdf.loc[condition, 'phase'] = phase

#%%
gdf
# %%
