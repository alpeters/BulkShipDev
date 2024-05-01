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
feature_sets_df['calc_agg_adm_sep1'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1']

feature_sets_df['calc_agg_adm_sep2'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep2']

feature_sets_df['calc_agg_inst_adm_sep1'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1'] | feature_sets_df['inst_adm_sep1']

feature_sets_df['calc_agg_inst_adm_sep1_chars'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1'] | feature_sets_df['inst_adm_sep1'] | feature_sets_df['add_chars']

feature_sets_df['calc_inst_adm_fund_chars'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['inst_adm_fund'] | feature_sets_df['add_chars']

#%% Create disjoint feature sets
for i in feature_sets_df['disjoint'].unique():
    feature_sets_df[str(i)] = feature_sets_df['disjoint'] == i

#%% All characteristics not in calcs
feature_sets_df['add_chars_plus'] = feature_sets_df['add_chars'] | (feature_sets_df['disjoint'] == 99)

#%%
feature_sets_df['incr2'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['4']

feature_sets_df['incr3'] = feature_sets_df['incr2'] | feature_sets_df['6']

#%% Everything except work and its components
feature_sets_df['pref_noWork'] = (feature_sets_df['preferred'] & ~feature_sets_df['work'])

#%% Incremental a
feature_sets_df['incr_a1'] = feature_sets_df['variable'] == 'cal_fc'

for i in range(2, 11):
    feature_sets_df['incr_a' + str(i)] = feature_sets_df['incr_a' + str(i-1)] | (feature_sets_df['incr_a'] == i)

#%% Incremental b
feature_sets_df['incr_b1'] = feature_sets_df['variable'] == 'cal_fc'

for i in range(2, 10):
    feature_sets_df['incr_b' + str(i)] = feature_sets_df['incr_b' + str(i-1)] | (feature_sets_df['incr_b'] == i)

#%%
feature_sets_df['calc_dist'] = (feature_sets_df['variable'] == 'cal_fc') | (feature_sets_df['variable'] == 'distance_sum')

#%% Create disjoint feature sets on top of cal_fc
for i in feature_sets_df['disjoint_b'].unique():
    feature_sets_df['disjoint_b' + str(i)] = (feature_sets_df['disjoint_b'] == i) | (feature_sets_df['variable'] == 'cal_fc')

#%% Create incremental sets from disjoint on top of cal_fc rank
feature_sets_df['djbrank_1'] = feature_sets_df['variable'] == 'cal_fc'

for i in range(2,11):
    feature_sets_df['djbrank_' + str(i)] = feature_sets_df['djbrank_' + str(i-1)] | (feature_sets_df['djbrank'] == i)

#%% Create single char feature sets on top of incr_b4
add_chars = feature_sets_df.loc[feature_sets_df['incr_b'] == 5]['variable']
#%%
for i, char in enumerate(add_chars):
    feature_sets_df['char' + str(i)] = feature_sets_df['djbrank_4'] | (feature_sets_df['variable'] == char)

#%%
variants = feature_sets_df.select_dtypes(include=bool).columns.tolist()
print(variants)

feature_sets_df.to_csv(trackeddatapath + 'ML_FC_variants_generated.csv')

#%%
pm.inspect_notebook(notebookpath + '.ipynb')
# %%
fast_only = False
feature_sets = ['char' + str(i) for i in range(0, 11)]
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
