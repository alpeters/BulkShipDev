'''
Runs ML_FC.ipynb with different parameters
Input(s): none
Output(s): none
'''

#%%
import pandas as pd
import papermill as pm

notebookpath = 'Machine Learning/ML_FC'
notebookoutpath = 'Machine Learning/data/ML_FC'

#%%
fast_only = False
feature_sets = ['djdrank10', 'oecd']
# feature_sets = ['djdrank4', 'speeddist'] #, 'work', 'djdrank10', 'oecd', 'djdrank7']
split_feature = "dwt" #"relseaspeed" # 
test_sets = ["testquart" + str(quart) + split_feature for quart in list(range(1,5))]

#%%
pm.inspect_notebook(notebookpath + '.ipynb')

#%%
for feature_set in feature_sets:
    for test_set_criterion in test_sets:
        out_suffix = feature_set + '_' + test_set_criterion
        if fast_only:
            out_suffix += '_fast'
        print(f'Running {out_suffix}...')
        try:
            pm.execute_notebook(
                notebookpath + '.ipynb',
                notebookoutpath + '_' + out_suffix + '.ipynb',
                parameters=dict(
                    feature_set=feature_set,
                    fast_only=fast_only,
                    test_set_criterion=test_set_criterion)
            )
            print(f'{out_suffix} completed')
        except:
            print(f'{out_suffix} failed')
# %%
