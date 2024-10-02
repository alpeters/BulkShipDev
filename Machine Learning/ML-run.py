'''
Runs ML.ipynb with different parameters
Input(s): none
Output(s): none
'''

#%%
import pandas as pd
import papermill as pm
import time

# # paths for local
notebookpath = 'Machine Learning/ML'
notebookoutpath = 'Machine Learning/data/ML'

# paths for CC
# notebookpath = 'ML'
# notebookoutpath = 'data/ML'


#%%
random_seed = 2652124
target = 'fc'
no_transform = True
models = 'struct' #'struct'
feature_sets = ['structfc']
# feature_sets = ['struct']
# feature_sets = ['speeddist'] #['djdrank4']
# feature_sets =  ['djdrank10', 'oecd', 'work', 'speeddist', 'djdrank4']
# feature_sets = ['djdrank4', 'speeddist'] #, 'work', 'djdrank10', 'oecd', 'djdrank7']
split_feature = "relseaspeed" # "dwt" #

test_sets = [["dec1relseaspeed"], ["dec10relseaspeed"], ["highrelseaspeed"], ["lowrelseaspeed"]]
train_sets = [["highrelseaspeed"], ["lowrelseaspeed"], ["dec1relseaspeed"], ["dec10relseaspeed"]]

# test_sets = [["trainquart" + str(quart) + split_feature] for quart in list(range(1,5))]
# test_sets = [["testquart" + str(quart) + split_feature] for quart in list(range(1,2))]

# test_sets = [["testquart4relseaspeed", "testhighdraught"]]
# test_sets = [["testquart4relseaspeed", "testhighdraught"],
#              ["testquart1relseaspeed", "testhighdraught"],
#              ["testquart4relseaspeed", "testlowdraught"],
#              ["testquart1relseaspeed", "testlowdraught"],]

#%%
pm.inspect_notebook(notebookpath + '.ipynb')

#%%
for feature_set in feature_sets:
    # for test_set_criteria in test_sets:    
    for test_set_criteria, train_set_criteria in zip(test_sets, train_sets):
        out_suffix = target + '_' + feature_set + '_' + ''.join(test_set_criteria)
        if models != 'all':
            out_suffix += '_' + models
        print(f'Running {out_suffix}...')
        try:
            pm.execute_notebook(
                notebookpath + '.ipynb',
                notebookoutpath + '_' + out_suffix + '.ipynb',
                parameters=dict(
                    random_seed=random_seed,
                    target=target,
                    no_transform=no_transform,
                    feature_set=feature_set,
                    models=models,
                    test_set_criteria=test_set_criteria,
                    train_set_criteria =train_set_criteria)
            )
            print(f'{out_suffix} completed')
        except:
            print(f'{out_suffix} failed')
    # time.sleep(5*60)
# %%

