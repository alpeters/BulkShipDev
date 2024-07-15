'''
Runs ML_FC_features.ipynb with different feature sets
Input(s): ML_FC_variants.csv, ML_FC_features.ipynb
Output(s): ML_FC_variants_generated.csv, ML_FC_features_[variant].ipynb
'''

#%%
import pandas as pd
import papermill as pm

trackeddatapath = 'Machine Learning/tracked_data/'
notebookpath = 'Machine Learning/ML_FC_features'
notebookoutpath = 'Machine Learning/data/ML_FC_features'

#%%
feature_sets_df = pd.read_csv(trackeddatapath + 'ML_FC_variants.csv')
variants = feature_sets_df.select_dtypes(include=bool).columns.tolist()
print(variants)

#%% Create combined features sets
# feature_sets_df['calc_agg_adm_sep1'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1']

#%% Create disjoint feature sets
# for i in feature_sets_df['disjoint'].unique():
#     feature_sets_df[str(i)] = feature_sets_df['disjoint'] == i

#%% Incremental 
name = 'djdrank'
feature_count = 10
feature_sets_df[name + '1'] = feature_sets_df['variable'] == 'cal_fc'

for i in range(2, feature_count+1):
    feature_sets_df[name + str(i)] = feature_sets_df[name + str(i-1)] | (feature_sets_df[name] == i)

#%% Create disjoint feature sets on top of cal_fc
name = 'djdrank'

for i in feature_sets_df[name].unique():
    feature_sets_df[name + 'seppluscal' + str(i)] = (feature_sets_df[name] == i) | (feature_sets_df['variable'] == 'cal_fc')

#%% Create defined feature sets on top of cal_fc
# base_name = 'decomp'
# set_count = 3

# for i in range(1, set_count+1):
#     feature_sets_df['cal' + base_name + str(i)] = feature_sets_df[base_name + str(i)] | (feature_sets_df['variable'] == 'cal_fc')


#%% Create single char feature sets on top of incr_b4
# add_chars = feature_sets_df.loc[feature_sets_df['incr_b'] == 5]['variable']
# #%%
# for i, char in enumerate(add_chars):
#     feature_sets_df['char' + str(i)] = feature_sets_df['djbrank_4'] | (feature_sets_df['variable'] == char)

#%%
variants = feature_sets_df.select_dtypes(include=bool).columns.tolist()
print(variants)

feature_sets_df.to_csv(trackeddatapath + 'ML_FC_variants_generated.csv')

#%%
pm.inspect_notebook(notebookpath + '.ipynb')
# %%
fast_only = True
# feature_sets = ['incr_c' + str(i) for i in range(1, 4)]
feature_sets = ['djdrank' + str(i) for i in range(2, 11)]
print(feature_sets)

#%%
for feature_set in feature_sets:
    out_suffix = feature_set
    if fast_only:
        out_suffix += '_fast'
    print(f'Running {out_suffix}...')
    try:
        pm.execute_notebook(
            notebookpath + '.ipynb',
            notebookoutpath + '_' + out_suffix + '.ipynb',
            parameters=dict(
                feature_set=feature_set,
                fast_only=fast_only)
        )
        print(f'{out_suffix} completed')
    except:
        print(f'{out_suffix} failed')
# %%
