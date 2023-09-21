# Initial single layer framework with All meaningful features

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
from sklearn.base import BaseEstimator, TransformerMixin
import re
df_ml = pd.read_csv(r'D:\ML\df_ml.csv', low_memory=False)
features = df_ml.columns.to_list()
pd.set_option('display.max_columns', None)
print(df_ml[features].info(verbose=1, show_counts=1))

#%%
selected_features = ['Dwt','Size.Category', 'Beam.Mld..m.', 'Draught..m.', 'Main.Engine.Fuel.Type', 'Main.Consumption.at.Service.Speed..tpd..',  'Main.Engine.Detail' ,
                     'HP.Total.Propulsion', 'SOx.Scrubber.Status', 'Eco...Electronic.Engine' , 'Service.Speed..knots.', 'Holds.Total.No', 'Grain.Capacity..cu.m.', 'Type',
                     'Main.Global.Zone..Last.12.Months...', 'Est.Crew.No' , 'LOA..m.', 'GT', 'Main.Bunker.Capacity..m3.', 'Beta.Atlantic.Pacific.Based..Last.12.Months.', 'Operational.Speed..knots.',
                     'TPC', 'NT', 'Strengthened.for.Heavy.Cargo', 'Strengthened.for.Ore' , 'Ballast.Cap..cu.m.', 'Propulsor.Detail' , 'Bale.Capacity..cu.m.', 'Owner.Size',  'Post.Panamax.Old.Locks..Ind.',
                     'SOx.Scrubber.1.Retrofit.Indicator', 'Panama.NT', 'Suez.NT', 'Air.Draft.From.Keel', 'EST.Number..Energy.Saving.Technologies.', 'Consumption..tpd.', 'Deballast.Cap..cu.m.h.', 'CGT', 'LBP..m.', 'Gear..Ind.',
                     'Log.fitted..Ind.' , 'Speed..knots.', 'Speed.category', 'Ice.Class..Ind.', 'BWMS.Status' , 'Heavy.Lift..Ind.', 'Hatches.Total.No', 'Beta.Deployment..Last.12.Months.' ,
                     'West.Coast.N.America.Deployment..Time.in.Last.12.Months....', 'West.Coast.S.America.Deployment..Time.in.Last.12.Months....' , 'East.Coast.N.America.Deployment..Time.in.Last.12.Months....', 'East.Coast.S.America.Deployment..Time.in.Last.12.Months....',
                     'West.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'UK.Cont.Deployment..Time.in.Last.12.Months....', 'East.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'Med.Black.Sea.Deployment..Time.in.Last.12.Months....',
                     'Australasia.Deployment..Time.in.Last.12.Months....', 'Middle.East.Deployment..Time.in.Last.12.Months....', 'East.Asia.Deployment..Time.in.Last.12.Months....', 'Indian.Subcont.Deployment..Time.in.Last.12.Months....',
                     'SE.Asia.Deployment..Time.in.Last.12.Months....', 'North.Asia.Deployment..Time.in.Last.12.Months....' , 'Depth.Moulded..m.', 'EU.distance', 'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique',
                     'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum', 't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum', 'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum','age']

fc_selected = df_ml[selected_features]

#%%
df_ml['Main.Engine.Fuel.Type'].unique()
df_ml['Type'].unique()
print(df_ml.loc[df_ml['Group'] == 'Bulk Ore Carrier'])
len(selected_features)

#%%
# object_columns = fc_selected.select_dtypes(include='object').columns
# fc_selected = df_ml.copy()
# fc_selected = fc_selected[object_columns]
#fc_selected['Beta.Atlantic.Pacific.Based..Last.12.Months.'] = fc_selected['Beta.Atlantic.Pacific.Based..Last.12.Months.'].astype(str)
# fc_selected.info()
# fc_selected.fillna('Missing Value', inplace=True)

# for column in object_columns:
#     data_count = fc_selected[column].count()
#     unique_values = np.unique(fc_selected[column])
#     print(f"Column '{column}' has {data_count} values, the unique value includes: {unique_values}.")

#%%
numerical_columns = df_ml[selected_features].select_dtypes(include=['float64','int64']).columns
ordinal_cols = ['Size.Category','Owner.Size','Speed.category']
median_imputer = SimpleImputer(strategy='median')
frequent_imputer = SimpleImputer(strategy='most_frequent')
standard_scaler = StandardScaler()

log_transformer = FunctionTransformer(np.log1p, validate=True)
ordinal_transformer = make_pipeline(frequent_imputer, OrdinalEncoder(categories=[['Capesize', 'Panamax', 'Handymax', 'Handysize'],
                                                                                 ['Very Small (1-5)', 'Small (6-10)', 'Medium (11-20)', 'Large (21-50)',
                                                                                  'Very Large (51-100)', 'Extra Large (100+)'],
                                                                                 ['Eco-Speed', 'Eco-Speed - Laden', 'Laden', 'Service', 'Trial Speed']]))


impute_and_transform = Pipeline(steps=[('imputer', median_imputer),
                                       ('log_transform', log_transformer)])

impute_transform_scale = make_pipeline(median_imputer, log_transformer, standard_scaler)
impute_onehot = make_pipeline(frequent_imputer, OneHotEncoder(handle_unknown='ignore'))
# Main.Engine.Detail Transformer, impute missing value with median
def extract_engine_details_final(engine_str):
    # Extract number of strokes
    strokes = int(re.search(r"(\d)-stroke", engine_str).group(1)) if re.search(r"(\d)-stroke", engine_str) else None

    # Extract number of cylinders
    cylinders = int(re.search(r"(\d)-cyl", engine_str).group(1)) if re.search(r"(\d)-cyl", engine_str) else None

    # Extract power in mKW
    power_match = re.search(r"(\d+,?\d*)mkW", engine_str)
    power = int(power_match.group(1).replace(',', '')) if power_match else None

    # Extract RPM (accounting for possible decimal values)
    rpm_match = re.search(r"at (\d+\.?\d*)rpm", engine_str)
    rpm = float(rpm_match.group(1)) if rpm_match else None

    return strokes, cylinders, power, rpm

# class EngineDetailTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         print(f'This is X: {X}')
#         # Apply the extraction function
#         details = X.apply(extract_engine_details_final)

#         # Convert the extracted details into a DataFrame
#         df_details = pd.DataFrame(details.tolist(), columns=['Strokes', 'Cylinders', 'Power_mKW', 'RPM'], index=X.index)

#         return df_details

class EngineDetailTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Apply the extraction function to each cell
        details = X.iloc[:,0].apply(extract_engine_details_final)

        # Convert the extracted details into a DataFrame
        df_details = pd.DataFrame(details.tolist(), columns=['Strokes', 'Cylinders', 'Power_mKW', 'RPM'], index=X.index)
        df_details.fillna(df_details.median(), inplace=True)
        return df_details

engine_detail_transformer = EngineDetailTransformer()

# Main.Global.Zone..Last.12.Months... Transformer

def extract_zone_details_safe(zone_str):
    if not isinstance(zone_str, str):
        return None, None

    # Extract location
    location_pattern = r"^(.*?) \("
    location_match = re.search(location_pattern, zone_str)
    location = location_match.group(1) if location_match else None

    # Extract percentage
    percentage_pattern = r"(\d+\.\d+) %"
    percentage_match = re.search(percentage_pattern, zone_str)
    percentage = float(percentage_match.group(1)) / 100 if percentage_match else None

    return location, percentage




class ZoneDetailTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_locations = ['East Asia', 'East Coast Africa', 'East Coast North America',
                                 'West Coast Africa', 'South East Asia', 'Australasia',
                                 'West Coast South America', 'Indian Subcontinent',
                                 'United Kingdom/Continent', 'Mediterranean / Black Sea',
                                 'East Coast South America', 'Middle East',
                                 'West Coast North America', 'North Asia']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply the extraction function
        details = X.iloc[:,0].apply(extract_zone_details_safe)

        # Convert the extracted details into a DataFrame
        df_details = pd.DataFrame(details.tolist(), columns=['Zone_Location', 'Zone_Percentage'], index=X.index)

        # One-hot encode the Zone_Location column with predefined unique locations
        df_encoded = pd.DataFrame(index=df_details.index)
        for location in self.unique_locations:
            df_encoded[f'Zone_Location_{location}'] = (df_details['Zone_Location'] == location).astype(int)

        # Add the Zone_Percentage column
        df_encoded['Zone_Percentage'] = df_details['Zone_Percentage']
        df_encoded.fillna(df_encoded.median(), inplace=True)
        return df_encoded



# Propulsor.Detail Transformer, impute missing value with median
def extract_rpm_from_propulsor(propulsor_str):
    # Check if the entry is a string
    if not isinstance(propulsor_str, str):
        return None

    # Extract RPM (accounting for possible decimal values)
    rpm_pattern = r"(\d+\.?\d*) ?rpm"
    rpm_match = re.search(rpm_pattern, propulsor_str)
    rpm = float(rpm_match.group(1)) if rpm_match else None

    return rpm

class PropulsorDetailTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply the RPM extraction function
        rpm_series = X.iloc[:,0].apply(extract_rpm_from_propulsor)

        # Convert the extracted RPM series into a DataFrame
        df_rpm = rpm_series.to_frame(name='RPM')
        df_rpm.fillna(df_rpm.median(), inplace=True)
        return df_rpm


binary_cols = ['SOx.Scrubber.Status', 'Eco...Electronic.Engine', 'Strengthened.for.Heavy.Cargo', 'Strengthened.for.Ore', 'Post.Panamax.Old.Locks..Ind.','SOx.Scrubber.1.Retrofit.Indicator', 'Gear..Ind.',
               'Log.fitted..Ind.', 'Ice.Class..Ind.', 'BWMS.Status','Heavy.Lift..Ind.','Beta.Deployment..Last.12.Months.']
onehot_cols = ['Main.Engine.Fuel.Type','Beta.Atlantic.Pacific.Based..Last.12.Months.','Type']

binary_transformer = make_pipeline(SimpleImputer(strategy='constant',fill_value='MissingValue'), OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'))

preprocessor_trees = ColumnTransformer(
    transformers=[
        ('impute_and_transform', impute_and_transform, numerical_columns),
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('engine_detail_transformer', EngineDetailTransformer(), ['Main.Engine.Detail']),
        ('zone_detail_transformer', ZoneDetailTransformer(), ['Main.Global.Zone..Last.12.Months...']),
        ('propulsor_detail_Transformer', PropulsorDetailTransformer(), ['Propulsor.Detail']),
        ('binary_transformer', binary_transformer, binary_cols),
        ('one-hot', impute_onehot, onehot_cols)
    ])

preprocessor_linear = ColumnTransformer(
    transformers=[
        ('impute_transform_scale', impute_transform_scale, numerical_columns),
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('engine_detail_transformer', EngineDetailTransformer(), ['Main.Engine.Detail']),
        ('zone_detail_transformer', ZoneDetailTransformer(), ['Main.Global.Zone..Last.12.Months...']),
        ('propulsor_detail_Transformer', PropulsorDetailTransformer(), ['Propulsor.Detail']),
        ('binary_transformer', binary_transformer, binary_cols),
        ('one-hot', impute_onehot, onehot_cols)
    ])




def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

models_trees = {
    "random forest": RandomForestRegressor(),
    "catboost": CatBoostRegressor(verbose=0),
    "lightgbm": LGBMRegressor(verbose=0),
    "xgboost": XGBRegressor(),
    "decision tree": DecisionTreeRegressor(),
}

models_linear = {
    "baseline": DummyRegressor(),
    "ridge": Ridge(),
    "linear regression": LinearRegression(),
    "lasso": Lasso(),
    "kNN": KNeighborsRegressor()
}

#%%
X_train, X_test, y_train, y_test = train_test_split(fc_selected, df_ml['residual'], test_size=0.2)
X_transformed = preprocessor_linear.fit_transform(X_train)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
contains_nan = np.any(np.isnan(X_transformed))

print(contains_nan)

#%%
results_dict = {} # dictionary to store all the results

# for model in models_trees:
#     pipe = make_pipeline(preprocessor_trees, models_trees[model])
#     results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

for model in models_linear:
    pipe = make_pipeline(preprocessor_linear, models_linear[model])
    results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True,error_score='raise')

print(pd.DataFrame(results_dict).T)

#%%
df_ml[binary_cols + onehot_cols].info()

#%%
object_columns = binary_cols
fc_selected = df_ml.copy()
fc_selected = fc_selected[object_columns]
fc_selected.info()
fc_selected.fillna('Missing', inplace=True)

for column in object_columns:
    data = fc_selected[column]
    data.value_counts().sort_index().plot(kind='bar', color=['blue', 'red'])
    plt.ylabel('Frequency')
    plt.xlabel('Category')
    plt.title(f'Frequency plot of {column}')
    plt.xticks(rotation=0)  # To make x-axis labels horizontal
    plt.show()
    data_count = fc_selected[column].count()
    unique_values = np.unique(fc_selected[column])
    print(f"Column '{column}' has {data_count} values, the unique value includes: {unique_values}.")

#%%
# Hyperparameter tuning for Ridge
param_grid_ridge = {
    'ridge__alpha':[0.1, 0.01, 0.001, 0.0001, 1, 10, 100, 1000, 10000, 100000]
}

model_ridge = Ridge()
pipe_ridge = make_pipeline(preprocessor_linear, model_ridge)

grid_search_ridge =  GridSearchCV(pipe_ridge, param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_search_ridge.fit(X_train, y_train)

result_ridge = pd.DataFrame(grid_search_ridge.cv_results_)[
    [
        "mean_test_score",
        "param_ridge__alpha",
        "mean_fit_time",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index().T

print(result_ridge)

print(f"Best parameters for Ridge is: {grid_search_ridge.best_params_}")
print(f"Best score for Ridge is: {grid_search_ridge.best_score_}")
print(f"Test score for Ridge is: {grid_search_ridge.score(X_test, y_test)}")

#%%
# Hyperparameter tuning for Lasso
param_grid_lasso = {
    'lasso__alpha':[0.00001,0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
}

model_lasso = Lasso(max_iter=10000)
pipe_lasso = make_pipeline(preprocessor_linear, model_lasso)

grid_search_lasso =  GridSearchCV(pipe_lasso, param_grid_lasso, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso.fit(X_train, y_train)

result_lasso = pd.DataFrame(grid_search_lasso.cv_results_)[
    [
        "mean_test_score",
        "param_lasso__alpha",
        "mean_fit_time",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index().T

print(result_lasso)

print(f"Best parameters for Lasso is: {grid_search_lasso.best_params_}")
print(f"Best score for Lasso is: {grid_search_lasso.best_score_}")
print(f"Test score for Lasso is: {grid_search_lasso.score(X_test, y_test)}")

#%%
# %%
## Tests and graphs for linear models(LinearRegression and Ridge)
results_var = ['total.fc', 'FC_sum']
models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=1),
    "lasso": Lasso(max_iter=1000000, alpha= 0.001)
}

X_train, X_test, y_train, y_test = train_test_split(df_ml[selected_features + results_var], df_ml['residual'], test_size=0.2)

# Convert pandas series to numpy arrays
total_fc = X_test['total.fc'].values
FC_sum = X_test['FC_sum'].values / 1000000
X_test.drop(columns=results_var, inplace=True)

for model in models:
    pipe = make_pipeline(preprocessor_linear, models[model])
    pipe.fit(X_train, y_train)

    # Generate x values and fitted y values
    predicted_residual = pipe.predict(X_test)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparison of Reported and Estimated Fuel Consumption (m tonnes) using ' + model)
    y_values = np.log(FC_sum) + predicted_residual
    x_values = np.log(total_fc)

    r_squared_eng = r2_score(x_values, np.log(FC_sum))

    # Calculate correlation
    correlation_eng = np.corrcoef(x_values, np.log(FC_sum))[0, 1]

    # Calculate mean squared error
    mse_eng = mean_squared_error(x_values, np.log(FC_sum))

    # Create a scatter plot
    #plt.figure(figsize=(10, 8))
    ax1.set_xlim(4, 10)
    ax1.set_ylim(4, 10)
    ax1.scatter(x_values, np.log(FC_sum), label='Engineering approach')

    # Plot x=y reference line
    ax1.plot([4, 10], [4, 10], 'g--', label='x=y line')

    # Set x and y labels
    ax1.set_xlabel('Reported Fuel Consumption')
    ax1.set_ylabel('Calculated Fuel Consumption')

    # Set a title for the plot
    ax1.set_title("log(total_fc) vs log(FC_sum)")

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation_eng, r_squared_eng, mse_eng)
    ax1.legend(title=legend_title)

    # Show the plot
    ax1.plot()
    # Calculate R-squared
    r_squared = r2_score(x_values, y_values)

    # Calculate correlation
    correlation = np.corrcoef(x_values, y_values)[0, 1]

    # Calculate mean squared error
    mse = mean_squared_error(x_values, y_values)

    # Create a scatter plot
    #plt.figure(figsize=(10, 8))
    ax2.set_xlim(4, 10)
    ax2.set_ylim(4, 10)
    ax2.scatter(x_values, y_values, label='Data')

    # Plot x=y reference line
    ax2.plot([4, 10], [4, 10], 'g--', label='x=y line')

    # Set x and y labels
    ax2.set_xlabel('Reported Fuel Consumption')
    ax2.set_ylabel('Estimated Fuel Consumption')

    # Set a title for the plot
    ax2.set_title("log(total_fc) vs log(FC_sum) + predicted residual")

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    ax2.legend(title=legend_title)

    # Show the plot
    ax2.plot()

    # Calculate R-squared for residual
    r_squared = r2_score(y_test, predicted_residual)

    # Calculate correlation
    correlation = np.corrcoef(y_test, predicted_residual)[0, 1]

    # Calculate mean squared error
    mse = mean_squared_error(y_test, predicted_residual)
    # Create a scatter plot
    #plt.figure(figsize=(10, 8))
    ax3.set_xlim(0, 1.5)
    ax3.set_ylim(0, 1.5)
    ax3.scatter(y_test, predicted_residual, label='residual')

    # Plot x=y reference line
    ax3.plot([0, 2], [0, 2], 'g--', label='x=y line')

    # Set x and y labels
    ax3.set_xlabel('Calculated Fuel Consumption Residual')
    ax3.set_ylabel('Predicted Fuel Consumption Residual')

    # Set a title for the plot
    ax3.set_title("True residual vs estimated residual")

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    ax3.legend(title=legend_title)

    # Show the plot
    ax3.plot()


#%%
from sklearn.feature_selection import SelectFromModel
X_train, X_test, y_train, y_test = train_test_split(df_ml[selected_features], df_ml['residual'], test_size=0.2)
select_rf = SelectFromModel(
    # LinearRegression(), threshold="mean"
    RandomForestRegressor(n_estimators=100, random_state=42), threshold="median"
)
pipe_lr_model_based = make_pipeline(
    preprocessor_linear,select_rf, Lasso(max_iter=1000000, alpha= 0.001)
)

pd.DataFrame(
    cross_validate(pipe_lr_model_based, X_train, y_train, return_train_score=True)
).mean()

#%%
# INVALID CODE, DO NOT EXECUTE

# pipe_lr_model_based.fit(X_train, y_train)
# t = pipe_lr_model_based.predict(X_test)
# oc = onehot_cols + ['infrequent']
# print(r2_score(y_test,t))
# selector = pipe_lr_model_based.named_steps['selectfrommodel']
# ordinal = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['ordinal'].get_feature_names_out()
# # engine = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['engine_detail_transformer'].get_feature_names_out(input_features=['Main.Engine.Detail'])
# # zone = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['zone_detail_transformer'].get_feature_names_out(input_features=['Main.Global.Zone..Last.12.Months...'])
# #prop = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['propulsor_detail_Transformer'].get_feature_names_out(input_features=['Propulsor.Detail'])
# #binary = zone = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['binary_transformer'].get_feature_names_out()
# onehot_cols = pipe_lr_model_based.named_steps['columntransformer'].named_transformers_['one-hot'].get_feature_names_out()
# # Get the mask of selected features
# print(onehot_cols)
# selected_feats = selector.get_feature_names_out(numerical_columns.tolist() + ordinal.tolist() + ['Strokes', 'Cylinders', 'Power_mKW', 'RPM'] + ['Zone_Location', 'Zone_Percentage'] + ['prop_rpm']+  onehot_cols.tolist())

# # Get the names of the selected features (assuming you have feature names in a list or array)
# print(selected_feats)

#%%
X_train, X_test, y_train, y_test = train_test_split(df_ml[selected_features], df_ml['residual'], test_size=0.2)
select_rf = SelectFromModel(
    # LinearRegression(), threshold="mean"
    RandomForestRegressor(n_estimators=100, random_state=42), threshold="median"
)
pipe_lr_model_based = make_pipeline(
    preprocessor_linear,select_rf, Ridge(max_iter=1000000, alpha= 1 )
)

pd.DataFrame(
    cross_validate(pipe_lr_model_based, X_train, y_train, return_train_score=True)
).mean()