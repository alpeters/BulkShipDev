# Basic feature selection from 39 features using SelectFromModel with initial framework

import os
import string
import sys
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
sys.path.append("../code/.")
from sklearn import datasets
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
df_ml = pd.read_csv(r'F:\ShippingEmissions\Machine Learning\df_ml.csv', low_memory=False)
df_ml.describe()

#%%
column_names_list = df_ml.columns.tolist()
print(column_names_list)
usable_features = ['Dwt', 'Size.Category', 'Beam.Mld..m.', 'Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'Holds.Total.No', 'Grain.Capacity..cu.m.', 'Type', 'Est.Crew.No', 'LOA..m.', 'GT', 'Beta.Atlantic.Pacific.Based..Last.12.Months.', 'Operational.Speed..knots.', 'TPC', 'NT', 'Ballast.Cap..cu.m.', 'Bale.Capacity..cu.m.', 'CGT', 'LBP..m.', 'Gear..Ind.', 'Speed..knots.', 'Speed.category', 'BWMS.Status', 'Hatches.Total.No', 'West.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'Depth.Moulded..m.', 'EU.distance', 'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique', 'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum', 't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum', 'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age']
df_ml[usable_features].info()

#%%
numerical_columns = df_ml[usable_features].select_dtypes(include=['float64','int64']).columns
print(numerical_columns)

#%%
# for feature in numerical_columns:
#     plot = df_ml[feature].plot(kind = 'hist', bins = 20, alpha = 0.5)
#     plot.set_xlabel(feature)
#     plot.set_ylabel("Frequency")
#     plot.set_title(f"Histogram of {feature}")
#     plt.show()

#%%
object_columns = df_ml[usable_features].select_dtypes(include='object').columns

# Calculate the number of unique values for each object column
fc_selected = df_ml.copy()
fc_selected = fc_selected[object_columns]
#fc_selected['Beta.Atlantic.Pacific.Based..Last.12.Months.'] = fc_selected['Beta.Atlantic.Pacific.Based..Last.12.Months.'].astype(str)
fc_selected.info()
fc_selected.fillna('A', inplace=True)
for column in object_columns:
    data_count = fc_selected[column].count()
    unique_values = np.unique(fc_selected[column])
    print(f"Column '{column}' has {unique_values}.")

#%%
log_transform_cols =['Dwt', 'Beam.Mld..m.', 'Draught..m.', 'HP.Total.Propulsion',
                     'Service.Speed..knots.', 'Holds.Total.No', 'Grain.Capacity..cu.m.',
                     'Est.Crew.No', 'LOA..m.', 'GT', 'Operational.Speed..knots.', 'TPC',
                     'NT', 'Ballast.Cap..cu.m.', 'Bale.Capacity..cu.m.', 'CGT', 'LBP..m.',
                     'Speed..knots.', 'Hatches.Total.No',
                     'West.Coast.Africa.Deployment..Time.in.Last.12.Months....',
                     'Depth.Moulded..m.', 'EU.distance', 'distance_sum', 'work_sum',
                     'work_IS_sum', 'trip_nunique', 'W_component_first', 'ME_W_ref_first',
                     't_m_times_v_n_sum', 't_over_t_ref_with_m_sum',
                     't_over_t_ref_without_m_sum', 'v_over_v_ref_with_n_sum',
                     'v_over_v_ref_without_n_sum', 'age']

ordinal_cols = ['Size.Category']
categorical_cols = ['Type', 'Beta.Atlantic.Pacific.Based..Last.12.Months.', 'Gear..Ind.', 'Speed.category',
                    'BWMS.Status']

median_imputer = SimpleImputer(strategy='median')
freq_imputer = SimpleImputer(strategy='most_frequent')

standard_scaler = StandardScaler()

log_transformer = FunctionTransformer(np.log1p, validate=True)
ordinal_transformer = OrdinalEncoder(categories=[['Capesize', 'Panamax', 'Handymax', 'Handysize']])
categorical_transformer = make_pipeline(freq_imputer, OneHotEncoder(handle_unknown="ignore"))

impute_and_transform = Pipeline(steps=[('imputer', median_imputer),
                                       ('log_transform', log_transformer)])

impute_scale_transform = make_pipeline(median_imputer, log_transformer, standard_scaler)

preprocessor_trees = ColumnTransformer(
    transformers=[
        ('impute_and_transform', impute_and_transform, log_transform_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)])

preprocessor_linear = ColumnTransformer(
    transformers=[
        ('impute_scale_transform', impute_and_transform, log_transform_cols),
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('onehot', categorical_transformer, categorical_cols)])


#%%

X_train, X_test, y_train, y_test = train_test_split(df_ml[usable_features], df_ml['residual'], test_size=0.2)
select_rf = SelectFromModel(
    # LinearRegression(), threshold="mean"
    RandomForestRegressor(n_estimators=1000, random_state=42), threshold="median"
)
pipe_lr_model_based = make_pipeline(
    preprocessor_linear,select_rf, Lasso(max_iter=1000000, alpha= 0.001)
)

param_grid_selector = {
    'selectfrommodel__estimator':[Ridge(alpha=0.01), Ridge(alpha=0.1), Ridge(alpha=0.5),Ridge(alpha=1), LinearRegression(), RandomForestRegressor(n_estimators=10),RandomForestRegressor(n_estimators=100), RandomForestRegressor(n_estimators=500),
                                  RandomForestRegressor(n_estimators=1000),RandomForestRegressor(n_estimators=2000),RandomForestRegressor(n_estimators=5000),
                                  RandomForestRegressor(n_estimators=10000)],
    'selectfrommodel__threshold':['mean','median']
}

grid_search_selector = GridSearchCV(pipe_lr_model_based,param_grid_selector,cv=5,scoring='r2',n_jobs=-1)
grid_search_selector.fit(X_train,y_train)

print(f"Best parameters for the selector is: {grid_search_selector.best_params_}")
print(f"Best score for the selector is: {grid_search_selector.best_score_}")
print(f"Test score for the selector is: {grid_search_selector.score(X_test, y_test)}")
# pd.DataFrame(
#     cross_validate(pipe_lr_model_based, X_train, y_train, return_train_score=True)
# ).mean()

#%%
pipe_lr_model_based.named_steps

#%%
pipe_lr_model_based.fit(X_train, y_train)
t = pipe_lr_model_based.predict(X_test)
r2_score(y_test,t)
selector = pipe_lr_model_based.named_steps['selectfrommodel']

onehot_cols = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['onehot'].get_feature_names_out(input_features=categorical_cols)
# Get the mask of selected features
selected_feats = selector.get_feature_names_out(log_transform_cols + ordinal_cols + onehot_cols.tolist())

# Get the names of the selected features
print(selected_feats)

#%%
df_ml[selected_feats].info()