# # Two layer Two residual model, Trees + Linear, both with model-selected features

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
df_ml = pd.read_csv(r'D:\ML\df_ml.csv', low_memory=False)
df_ml.describe()

#%%
# features = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
# 'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
# 'Bale.Capacity..cu.m.', 'Sister.Vessel.Code', 'LBP..m.', 'Speed..knots.',
# 'West.Coast.Africa.Deployment..Time.in.Last.12.Months....',
# 'technical.efficiency', 'total.co2', 'co2.between.ms.ports',
# 'co2.from.ms.ports', 'co2.to.ms.ports', 'co2.ms.ports.at.berth',
# 'time.at.sea', 'fc.per.distance', 'fc.per.work.mass', 'co2.per.distance',
# 'co2.per.work.mass', 'EU.distance', 'speed', 'emission.factor', 'work.mass',
# 'load', 'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique',
# 'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
# 't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
# 'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age',
# 'distance_difference','Size.Category']

# log_transform_cols = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
# 'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
# 'Bale.Capacity..cu.m.', 'Sister.Vessel.Code', 'LBP..m.', 'Speed..knots.',
# 'West.Coast.Africa.Deployment..Time.in.Last.12.Months....',
# 'technical.efficiency', 'total.co2', 'co2.between.ms.ports',
# 'co2.from.ms.ports', 'co2.to.ms.ports', 'co2.ms.ports.at.berth',
# 'time.at.sea', 'fc.per.distance', 'fc.per.work.mass', 'co2.per.distance',
# 'co2.per.work.mass', 'EU.distance', 'speed', 'emission.factor', 'work.mass',
# 'load', 'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique',
# 'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
# 't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
# 'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age',
# 'distance_difference']

# Removed columns: 'total.co2', 'co2.between.ms.ports',
# 'co2.from.ms.ports', 'co2.to.ms.ports', 'co2.ms.ports.at.berth', 'Sister.Vessel.Code'

features = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
            'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
            'Bale.Capacity..cu.m.', 'LBP..m.', 'Speed..knots.',
            'West.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'EU.distance',
            'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique',
            'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
            't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
            'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age','Size.Category']

log_transform_cols = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
                      'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
                      'Bale.Capacity..cu.m.', 'LBP..m.', 'Speed..knots.',
                      'West.Coast.Africa.Deployment..Time.in.Last.12.Months....','EU.distance',
                      'distance_sum', 'work_sum', 'work_IS_sum', 'trip_nunique',
                      'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
                      't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
                      'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age']
# no_transform_cols = ['age']
ordinal_cols = ['Size.Category']
# 'fc.per.distance', 'fc.per.work.mass', 'co2.per.distance','co2.per.work.mass', 'technical.efficiency', 'emission.factor'

# for feature in log_transform_cols:
#     plot = df_ml[feature].plot(kind = 'hist', bins = 20, alpha = 0.5)
#     plot.set_xlabel(feature)
#     plot.set_ylabel("Frequency")
#     plot.set_title(f"Histogram of {feature}")
#     plt.show()


#%%
# Train - Test split
X_train, X_test, y_train, y_test = train_test_split(df_ml[features], df_ml['residual'], test_size=0.2)
X_train.info()

#%%
median_imputer = SimpleImputer(strategy='median')

standard_scaler = StandardScaler()

log_transformer = FunctionTransformer(np.log1p, validate=True)
ordinal_transformer = make_pipeline(OrdinalEncoder(categories=[['Capesize', 'Panamax', 'Handymax', 'Handysize']]))


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
        ('ordinal', ordinal_transformer, ordinal_cols)])


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
results_dict = {} # dictionary to store all the results

for model in models_trees:
    pipe = make_pipeline(preprocessor_trees, models_trees[model])
    results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

for model in models_linear:
    pipe = make_pipeline(preprocessor_linear, models_linear[model])
    results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

print(pd.DataFrame(results_dict).T)

#%%
# Hyperparameter tuning for CatBoostRegressor
param_grid_pipe_catboostregressor = {
    'catboostregressor__depth': [4, 6, 7, 8, 9 ,10],
    'catboostregressor__iterations': [10, 30, 50, 100, 500],
    'catboostregressor__l2_leaf_reg': [1, 5, 7, 9, 11, 15]
}

model_catboostregressor = CatBoostRegressor(verbose = 0)
pipe_catboostregressor = make_pipeline(preprocessor_trees, model_catboostregressor)

grid_search_catboostregressor = RandomizedSearchCV(pipe_catboostregressor, param_grid_pipe_catboostregressor, n_iter=30, cv=5, scoring='r2', n_jobs=-1)
grid_search_catboostregressor.fit(X_train, y_train)

# result_catboostregressor = pd.DataFrame(grid_search_catboostregressor.cv_results_)[
#     [
#         "mean_test_score",
#         "param_catboostregressor__depth",
#         "param_catboostregressor__learning_rate",
#         "param_catboostregressor__iterations",
#         "param_catboostregressor__l2_leaf_reg",
#         "mean_fit_time",
#         "rank_test_score",
#     ]
# ].set_index("rank_test_score").sort_index().T

# print(result_catboostregressor)

print(f"Best parameters for CatBoostRegressor is: {grid_search_catboostregressor.best_params_}")
print(f"Best score for CatBoostRegressor is: {grid_search_catboostregressor.best_score_}")
print(f"Test score for CatBoostRegressor is: {grid_search_catboostregressor.score(X_test, y_test)}")

#%%
# Hyperparameter tuning for RandomForestRegressor
param_grid_randomforestregressor = {
    'randomforestregressor__n_estimators': [50, 100, 200, 300, 500, 700, 1000],
    'randomforestregressor__max_depth': [10, 20, 30, 40, 50, None],
}

model_randomforestregressor = RandomForestRegressor()
pipe_randomforestregressor = make_pipeline(preprocessor_trees, model_randomforestregressor)

grid_search_randomforestregressor =  GridSearchCV(pipe_randomforestregressor, param_grid_randomforestregressor, cv=5, scoring='r2', n_jobs=-1)
grid_search_randomforestregressor.fit(X_train, y_train)

result_randomforestregressor = pd.DataFrame(grid_search_randomforestregressor.cv_results_)[
    [
        "mean_test_score",
        "param_randomforestregressor__n_estimators",
        "param_randomforestregressor__max_depth",
        "mean_fit_time",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index().T

print(result_randomforestregressor)

print(f"Best parameters for RandomForestRegressor is: {grid_search_randomforestregressor.best_params_}")
print(f"Best score for RandomForestRegressor is: {grid_search_randomforestregressor.best_score_}")
print(f"Test score for RandomForestRegressor is: {grid_search_randomforestregressor.score(X_test, y_test)}")

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
## Multiple rounds tests for model performance
results_var = ['total.fc', 'FC_sum']
models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=1),
    "lasso": Lasso(max_iter=1000000, alpha= 0.0001)
}

for model in models:
    columns = ['Model', 'r2', 'correlation', 'MSE', 'r2_residual', 'correlation_residual', 'MSE_residual']
    test_results = pd.DataFrame(columns=columns)
    for i in range(0,10):

        X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2)
        pipe = make_pipeline(preprocessor_linear, models[model])
        total_fc = X_test['total.fc'].values
        FC_sum = X_test['FC_sum'].values / 1000000

        X_test.drop(columns=results_var, inplace=True)
        pipe.fit(X_train, y_train)

        # Generate x values and fitted y values
        predicted_residual = pipe.predict(X_test)

        y_values = np.log(FC_sum) + predicted_residual
        x_values = np.log(total_fc)
        # Calculate R-squared
        r_squared = r2_score(x_values, y_values)

        # Calculate correlation
        correlation = np.corrcoef(x_values, y_values)[0, 1]

        # Calculate mean squared error
        mse = mean_squared_error(x_values, y_values)

        # Calculate R-squared for residual
        r_squared_r = r2_score(y_test, predicted_residual)

        # Calculate correlation
        correlation_r = np.corrcoef(y_test, predicted_residual)[0, 1]

        # Calculate mean squared error
        mse_r = mean_squared_error(y_test, predicted_residual)
        new_row = pd.DataFrame({'Model': [model],'r2':[r_squared], 'correlation':[correlation], 'MSE':[mse],'r2_residual':[r_squared_r], 'correlation_residual':[correlation_r], 'MSE_residual':[mse_r]})

        # Appending the new row
        test_results = pd.concat([test_results, new_row], ignore_index=True)
    print(test_results)
    mean = test_results['r2'].mean()
    std_dev = test_results['r2'].std()
    mean_r = test_results['r2_residual'].mean()
    std_dev_r = test_results['r2_residual'].std()
    print(f"Model: {model}, Mean for r2: {mean:.3f}, Standard Deviation for r2: {std_dev:.3f}, Mean for r2_residual:{mean_r:.2f}, Standard Deviation for r2_residual: {std_dev_r:.3f}")


#%%
total_fc = df_ml['total.fc'].values
FC_sum = df_ml['FC_sum'].values / 1000000
r2_score(np.log(total_fc), np.log(FC_sum))


# %%
## Tests and graphs for linear models(LinearRegression and Ridge)
results_var = ['total.fc', 'FC_sum']
models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=0.1),
    "lasso": Lasso(max_iter=1000000, alpha= 0.00001)
}

X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,random_state=42)

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
## Multiple rounds tests for model performance
results_var = ['total.fc', 'FC_sum']
models = {
    "random forest": RandomForestRegressor(max_depth=30, n_estimators=300),
    "catboost": CatBoostRegressor(learning_rate=0.2, l2_leaf_reg=9, iterations=100,depth=4,verbose=0),
}

for model in models:
    columns = ['Model', 'r2', 'correlation', 'MSE', 'r2_residual', 'correlation_residual', 'MSE_residual']
    test_results = pd.DataFrame(columns=columns)
    for i in range(0,10):

        X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2)
        pipe = make_pipeline(preprocessor_trees, models[model])
        total_fc = X_test['total.fc'].values
        FC_sum = X_test['FC_sum'].values / 1000000

        X_test.drop(columns=results_var, inplace=True)
        pipe.fit(X_train, y_train)

        # Generate x values and fitted y values
        predicted_residual = pipe.predict(X_test)

        y_values = np.log(FC_sum) + predicted_residual
        x_values = np.log(total_fc)
        # Calculate R-squared
        r_squared = r2_score(x_values, y_values)

        # Calculate correlation
        correlation = np.corrcoef(x_values, y_values)[0, 1]

        # Calculate mean squared error
        mse = mean_squared_error(x_values, y_values)

        # Calculate R-squared for residual
        r_squared_r = r2_score(y_test, predicted_residual)

        # Calculate correlation
        correlation_r = np.corrcoef(y_test, predicted_residual)[0, 1]

        # Calculate mean squared error
        mse_r = mean_squared_error(y_test, predicted_residual)
        new_row = pd.DataFrame({'Model': [model],'r2':[r_squared], 'correlation':[correlation], 'MSE':[mse],'r2_residual':[r_squared_r], 'correlation_residual':[correlation_r], 'MSE_residual':[mse_r]})

        # Appending the new row
        test_results = pd.concat([test_results, new_row], ignore_index=True)
    print(test_results)
    mean = test_results['r2'].mean()
    std_dev = test_results['r2'].std()
    mean_r = test_results['r2_residual'].mean()
    std_dev_r = test_results['r2_residual'].std()
    print(f"Model: {model}, Mean for r2: {mean:.3f}, Standard Deviation for r2: {std_dev:.3f}, Mean for r2_residual:{mean_r:.2f}, Standard Deviation for r2_residual: {std_dev_r:.3f}")

#%%
# Tests and graphs for Tree-based models (RandomForestRegressor and CatBoostRegressor)
results_var = ['total.fc', 'FC_sum']
X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,random_state = 42)

models = {
    "random forest": RandomForestRegressor(max_depth=30, n_estimators=300),
    "catboost": CatBoostRegressor(learning_rate=0.2, l2_leaf_reg=9, iterations=100,depth=4,verbose=0),
}

#Convert pandas series to numpy arrays
total_fc = X_test['total.fc'].values
FC_sum = X_test['FC_sum'].values/1000000

X_test.drop(columns = results_var, inplace=True)

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
# Second Layer
from sklearn.model_selection import KFold
linear_model = RandomForestRegressor(max_depth=30, n_estimators=300)
results_var = ['total.fc', 'FC_sum']
X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,random_state=42)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_train['predicted_residual'] = np.nan
total_fc = X_test['total.fc'].values
FC_sum = X_test['FC_sum'].values / 1000000
X_test.drop(columns=results_var, inplace=True)
# Split the train set into k folds
k = 10
kf = KFold(n_splits=k, shuffle=True)

# Cross validation to train - test first layer and generate predicted residual
for train_idx, val_idx in kf.split(X_train, y_train):
    linear_model = RandomForestRegressor(max_depth=30, n_estimators=300)
    X_train_sub, X_test_sub = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_sub, y_test_sub = y_train.iloc[train_idx], y_train.iloc[val_idx]
    linear_pipe = make_pipeline(preprocessor_trees, linear_model)
    linear_pipe.fit(X_train_sub, y_train_sub)
    predicted_residual = linear_pipe.predict(X_test_sub)
    X_train.loc[val_idx, 'predicted_residual'] = predicted_residual

r_squared_eng = r2_score(x_values, np.log(FC_sum))

# Calculate correlation
correlation_eng = np.corrcoef(x_values, np.log(FC_sum))[0, 1]

# Calculate mean squared error
mse_eng = mean_squared_error(x_values, np.log(FC_sum))

print(f'r2 score for residual 1 after cross validation: {r2_score(y_train, X_train["predicted_residual"])}')

X_train['residual_2'] = y_train - X_train['predicted_residual']

# Now we have added X_train dataframe with our target value for the second layer machine learning model 'residual_2'
# Fit the linear model on the whole training set
linear_pipe.fit(X_train, y_train)
p = linear_pipe.predict(X_test)
print(f'r2 score for residual 1 after train on the whole X_train set: {r2_score(y_test, p)}')
y_train_2 = X_train['residual_2']
X_test['predicted_residual'] = linear_pipe.predict(X_test)
y_test_2 =y_test - X_test['predicted_residual']
# print(X_test['predicted_residual'])


# Train the second layer

results_dict = {} # dictionary to store all the results

#%%
for model in models_trees:
    pipe_2 = make_pipeline(preprocessor_trees, models_trees[model])
    results_dict[model] = mean_std_cross_val_scores(pipe_2, X_train, y_train_2, cv=5, return_train_score=True)

for model in models_linear:
    pipe_2 = make_pipeline(preprocessor_linear, models_linear[model])
    results_dict[model] = mean_std_cross_val_scores(pipe_2, X_train, y_train_2, cv=5, return_train_score=True)

print(pd.DataFrame(results_dict).T)

#%%
# Hyperparameter tuning for CatBoostRegressor
param_grid_pipe_catboostregressor = {
    'catboostregressor__depth': [4, 6, 7, 8, 9 ,10],
    'catboostregressor__iterations': [10, 30, 50, 100, 500],
    'catboostregressor__l2_leaf_reg': [1, 5, 7, 9, 11, 15]
}

model_catboostregressor = CatBoostRegressor(verbose = 0)
pipe_catboostregressor = make_pipeline(preprocessor_trees, model_catboostregressor)

grid_search_catboostregressor = RandomizedSearchCV(pipe_catboostregressor, param_grid_pipe_catboostregressor, n_iter=30, cv=5, scoring='r2', n_jobs=-1)
grid_search_catboostregressor.fit(X_train, y_train_2)

# result_catboostregressor = pd.DataFrame(grid_search_catboostregressor.cv_results_)[
#     [
#         "mean_test_score",
#         "param_catboostregressor__depth",
#         "param_catboostregressor__learning_rate",
#         "param_catboostregressor__iterations",
#         "param_catboostregressor__l2_leaf_reg",
#         "mean_fit_time",
#         "rank_test_score",
#     ]
# ].set_index("rank_test_score").sort_index().T

# print(result_catboostregressor)

print(f"Best parameters for CatBoostRegressor is: {grid_search_catboostregressor.best_params_}")
print(f"Best score for CatBoostRegressor is: {grid_search_catboostregressor.best_score_}")
print(f"Test score for CatBoostRegressor is: {grid_search_catboostregressor.score(X_test, y_test_2)}")

#%%
# Hyperparameter tuning for RandomForestRegressor
param_grid_randomforestregressor = {
    'randomforestregressor__n_estimators': [50, 100, 200, 300, 500, 700, 1000],
    'randomforestregressor__max_depth': [10, 20, 30, 40, 50, None],
}

model_randomforestregressor = RandomForestRegressor()
pipe_randomforestregressor = make_pipeline(preprocessor_trees, model_randomforestregressor)

grid_search_randomforestregressor =  GridSearchCV(pipe_randomforestregressor, param_grid_randomforestregressor, cv=5, scoring='r2', n_jobs=-1)
grid_search_randomforestregressor.fit(X_train, y_train_2)

# result_randomforestregressor = pd.DataFrame(grid_search_randomforestregressor.cv_results_)[
#     [
#         "mean_test_score",
#         "param_randomforestregressor__n_estimators",
#         "param_randomforestregressor__max_depth",
#         "mean_fit_time",
#         "rank_test_score",
#     ]
# ].set_index("rank_test_score").sort_index().T

# print(result_randomforestregressor)

print(f"Best parameters for RandomForestRegressor is: {grid_search_randomforestregressor.best_params_}")
print(f"Best score for RandomForestRegressor is: {grid_search_randomforestregressor.best_score_}")
print(f"Test score for RandomForestRegressor is: {grid_search_randomforestregressor.score(X_test, y_test_2)}")

#%%
# Tests and graphs for Linear-based models (RandomForestRegressor and CatBoostRegressor)

models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=0.1),
    "lasso": Lasso(max_iter=1000000, alpha= 0.00001)
}



for model in models:
    pipe = make_pipeline(preprocessor_linear, models[model])
    # rows_with_nan = X_train[X_train.isnull().any(axis=1)]
    # print(rows_with_nan)
    pipe.fit(X_train, y_train_2)

    # Generate x values and fitted y values
    predicted_residual = pipe.predict(X_test)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparison of Reported and Estimated Fuel Consumption (m tonnes) using ' + model)
    y_values = np.log(FC_sum) + X_test['predicted_residual'] + predicted_residual
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
    ax2.set_title("log(total_fc) vs log(FC_sum) + predicted residual + residual 2")

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    ax2.legend(title=legend_title)

    # Show the plot
    ax2.plot()

    # Calculate R-squared for residual
    r_squared = r2_score(y_test_2, predicted_residual)

    # Calculate correlation
    correlation = np.corrcoef(y_test_2, predicted_residual)[0, 1]

    # Calculate mean squared error
    mse = mean_squared_error(y_test_2, predicted_residual)
    # Create a scatter plot
    #plt.figure(figsize=(10, 8))
    ax3.set_xlim(0, 0.2)
    ax3.set_ylim(0, 0.2)
    ax3.scatter(y_test_2, predicted_residual, label='residual')
    # Plot x=y reference line
    ax3.plot([0, 0.2], [0, 0.2], 'g--', label='x=y line')

    # Set x and y labels
    ax3.set_xlabel('Calculated Fuel Consumption Residual')
    ax3.set_ylabel('Predicted Fuel Consumption Residual')

    # Set a title for the plot
    ax3.set_title("residual - predicted_residual vs residual 2")

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    ax3.legend(title=legend_title)

    # Show the plot
    ax3.plot()

#%%
# Test for the second layer
import random


linear_model = RandomForestRegressor(max_depth=30, n_estimators=300)
results_var = ['total.fc', 'FC_sum']
models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=0.1),
    "lasso": Lasso(max_iter=1000000, alpha= 0.00001)
}

columns = ['Model', 'r2', 'correlation', 'MSE', 'r2_residual', 'correlation_residual', 'MSE_residual','r2_2', 'correlation_2', 'MSE_2',
           'r2_residual_2', 'correlation_residual_2', 'MSE_residual_2']

test_results = pd.DataFrame(columns=columns)
for model in models:

    for i in range(0,10):
        random_state = random.randint(1, 1000)
        X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,random_state=random_state)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_train['predicted_residual'] = np.nan
        total_fc = X_test['total.fc'].values
        FC_sum = X_test['FC_sum'].values / 1000000
        X_test.drop(columns=results_var, inplace=True)
        # Split the train set into k folds
        k = 10
        kf = KFold(n_splits=k, shuffle=True)

        # Cross validation to train - test first layer and generate predicted residual
        for train_idx, val_idx in kf.split(X_train, y_train):
            linear_model = RandomForestRegressor(max_depth=30, n_estimators=300)
            X_train_sub, X_test_sub = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_sub, y_test_sub = y_train.iloc[train_idx], y_train.iloc[val_idx]
            linear_pipe = make_pipeline(preprocessor_trees, linear_model)
            linear_pipe.fit(X_train_sub, y_train_sub)
            predicted_residual = linear_pipe.predict(X_test_sub)
            X_train.loc[val_idx, 'predicted_residual'] = predicted_residual


        print(f'r2 score for residual 1 after cross validation: {r2_score(y_train, X_train["predicted_residual"])}')

        X_train['residual_2'] = y_train - X_train['predicted_residual']

        # Now we have added X_train dataframe with our target value for the second layer machine learning model 'residual_2'
        # Fit the linear model on the whole training set
        linear_pipe.fit(X_train, y_train)
        p = linear_pipe.predict(X_test)
        y_values = np.log(FC_sum) + p
        x_values = np.log(total_fc)
        # Calculate R-squared
        r_squared = r2_score(x_values, y_values)

        # Calculate correlation
        correlation = np.corrcoef(x_values, y_values)[0, 1]

        # Calculate mean squared error
        mse = mean_squared_error(x_values, y_values)

        # Calculate R-squared for residual
        r_squared_r = r2_score(y_test, p)

        # Calculate correlation
        correlation_r = np.corrcoef(y_test, p)[0, 1]

        # Calculate mean squared error
        mse_r = mean_squared_error(y_test, p)

        print(f'r2 score for residual 1 after train on the whole X_train set: {r2_score(y_test, p)}')
        y_train_2 = X_train['residual_2']
        X_test['predicted_residual'] = p
        y_test_2 =y_test - X_test['predicted_residual']

        pipe = make_pipeline(preprocessor_linear, models[model])
        # rows_with_nan = X_train[X_train.isnull().any(axis=1)]
        # print(rows_with_nan)
        pipe.fit(X_train, y_train_2)

        # Generate x values and fitted y values
        predicted_residual_2 = pipe.predict(X_test)
        y_values_2 = np.log(FC_sum) + p + predicted_residual_2
        x_values = np.log(total_fc)
        # Calculate R-squared
        r_squared_2 = r2_score(x_values, y_values_2)

        # Calculate correlation
        correlation_2 = np.corrcoef(x_values, y_values_2)[0, 1]

        # Calculate mean squared error
        mse_2 = mean_squared_error(x_values, y_values_2)
        # Calculate R-squared for residual2
        r_squared_r_2 = r2_score(y_test_2, predicted_residual_2)

        # Calculate correlation
        correlation_r_2 = np.corrcoef(y_test_2, predicted_residual_2)[0, 1]

        # Calculate mean squared error
        mse_r_2 = mean_squared_error(y_test_2, predicted_residual_2)

        new_row = pd.DataFrame({'Model': [model],'r2':[r_squared], 'correlation':[correlation], 'MSE':[mse],'r2_residual':[r_squared_r], 'correlation_residual':[correlation_r], 'MSE_residual':[mse_r],
                                'r2_2':[r_squared_2], 'correlation_2':[correlation_2], 'MSE_2':[mse_2],'r2_residual_2':[r_squared_r_2], 'correlation_residual_2':[correlation_r_2], 'MSE_residual_2':[mse_r_2]})

        # Appending the new row
        test_results = pd.concat([test_results, new_row], ignore_index=True)
test_results.loc['Mean'] = test_results.mean()
print(test_results)

#%%
print(test_results.mean)
