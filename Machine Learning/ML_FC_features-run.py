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
# notebookpath = 'Machine Learning/test'

#%%
feature_sets_df = pd.read_csv(trackeddatapath + 'ML_FC_variants.csv')
variants = feature_sets_df.select_dtypes(include=bool).columns.tolist()
print(variants)

#%% Create combined features sets
feature_sets_df['calc_agg_adm_sep1'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1']

feature_sets_df['calc_agg_adm_sep2'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep2']

feature_sets_df['calc_agg_inst_adm_sep1'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1'] | feature_sets_df['inst_adm_sep1']

feature_sets_df['calc_agg_inst_adm_sep1_chars'] = (feature_sets_df['variable'] == 'cal_fc') | feature_sets_df['agg_adm_sep1'] | feature_sets_df['inst_adm_sep1'] | feature_sets_df['add_chars']

#%%
variants = feature_sets_df.select_dtypes(include=bool).columns.tolist()
print(variants)

feature_sets_df.to_csv(trackeddatapath + 'ML_FC_variants_generated.csv')

#%%
pm.inspect_notebook(notebookpath + '.ipynb')
# %%
fast_only = False
# feature_sets = [variants[i] for i in [3, 4]]
feature_sets = ['cal_fc'] + variants
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
            notebookpath + '_' + out_suffix + '.ipynb',
            parameters=dict(
                feature_set=feature_set,
                fast_only=fast_only)
        )
        print(f'{out_suffix} completed')
    except:
        print(f'{out_suffix} failed')
# %%
