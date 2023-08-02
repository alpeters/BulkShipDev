# %%
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
df_ml = pd.read_csv(r'F:\ShippingEmissions\Machine Learning\df_ml.csv', low_memory=False)
df_ml.describe()

# %%
features = ["W_component_first", "ME_W_ref_first", "t_m_times_v_n_sum",
            "t_over_t_ref_with_m_sum", "t_over_t_ref_without_m_sum",
            "v_over_v_ref_without_n_sum", "age", "Dwt", "LBP..m.",
            "Beam.Mld..m.", "Draught..m.", "load", "TPC",
            "Service.Speed..knots.", "Size.Category"]
log_transform_cols = ["W_component_first", "ME_W_ref_first", "t_m_times_v_n_sum",
                      "t_over_t_ref_with_m_sum", "t_over_t_ref_without_m_sum",
                      "v_over_v_ref_without_n_sum", "Dwt", "LBP..m.",
                      "Beam.Mld..m.", "Draught..m.", "load", "TPC",
                      "Service.Speed..knots."]
no_transform_cols = ['age']
ordinal_cols = ['Size.Category']


for feature in log_transform_cols:
    plot = df_ml[feature].plot(kind = 'hist', bins = 20, alpha = 0.5)
    plot.set_xlabel(feature)
    plot.set_ylabel("Frequency")
    plot.set_title(f"Histogram of {feature}")
    plt.show()

print(df_ml.describe(include="all"))
plot = df_ml['residual'].plot(kind = 'hist', bins = 20, alpha = 0.5)

# %%
# Train - Test split
X_train, X_test, y_train, y_test = train_test_split(df_ml[features], df_ml['residual'], test_size=0.2, random_state=42)
X_train.info()

# %%
mean_imputer = SimpleImputer(strategy='mean')

standard_scaler = StandardScaler()

log_transformer = FunctionTransformer(np.log1p, validate=True)
ordinal_transformer = make_pipeline(OrdinalEncoder(categories=[['Capesize', 'Panamax', 'Handymax', 'Handysize']]))


impute_and_transform = Pipeline(steps=[('imputer', mean_imputer),
                                       ('log_transform', log_transformer)])

impute_scale_transform = make_pipeline(mean_imputer, standard_scaler, log_transformer)

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

# %%
results_dict = {}  # dictionary to store all the results

for model in models_trees:
    pipe = make_pipeline(preprocessor_trees, models_trees[model])
    results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

for model in models_linear:
    pipe = make_pipeline(preprocessor_linear, models_linear[model])
    results_dict[model] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

print(pd.DataFrame(results_dict).T)

# %%
# Hyperparameter tuning for CatBoostRegressor
param_grid_pipe_catboostregressor = {
    'catboostregressor__depth': [4, 6, 8, 10, 12],
    'catboostregressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'catboostregressor__iterations': [30, 50, 100],
    'catboostregressor__l2_leaf_reg': [1, 3, 5, 7, 9]
}

model_catboostregressor = CatBoostRegressor(verbose = 0)
pipe_catboostregressor = make_pipeline(preprocessor_trees, model_catboostregressor)

grid_search_catboostregressor = RandomizedSearchCV(pipe_catboostregressor, param_grid_pipe_catboostregressor, n_iter=30, cv=5, scoring='r2', n_jobs=-1)
grid_search_catboostregressor.fit(X_train, y_train)

result_catboostregressor = pd.DataFrame(grid_search_catboostregressor.cv_results_)[
    [
        "mean_test_score",
        "param_catboostregressor__depth",
        "param_catboostregressor__learning_rate",
        "param_catboostregressor__iterations",
        "param_catboostregressor__l2_leaf_reg",
        "mean_fit_time",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index().T

print(result_catboostregressor)

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

# %%
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

# %%
## Tests and graphs for linear models(LinearRegression and Ridge)
results_var = ['total.fc', 'FC_sum']
models = {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=0.01)
}

X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,
                                                    random_state=666)

# Convert pandas series to numpy arrays
total_fc = X_test['total.fc'].values
FC_sum = X_test['FC_sum'].values / 1000000

X_test.drop(columns=results_var, inplace=True)

for model in models:
    pipe = make_pipeline(preprocessor_linear, models[model])
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

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.xlim(4, 10)
    plt.ylim(4, 10)
    plt.scatter(x_values, y_values, label='Data')

    # Plot x=y reference line
    plt.plot([4, 10], [4, 10], 'g--', label='x=y line')

    # Set x and y labels
    plt.xlabel('Reported Fuel Consumption')
    plt.ylabel('Estimated Fuel Consumption')

    # Set a title for the plot
    plt.title('Comparison of Reported and Estimated Fuel Consumption (m tonnes) using ' + model)

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    plt.legend(title=legend_title)

    # Show the plot
    plt.show()

# %%
# Tests and graphs for Tree-based models (RandomForestRegressor and CatBoostRegressor)
results_var = ['total.fc', 'FC_sum']
X_train, X_test, y_train, y_test = train_test_split(df_ml[features + results_var], df_ml['residual'], test_size=0.2,
                                                    random_state=666)

models = {
    "random forest": RandomForestRegressor(max_depth=30, n_estimators=100),
    "catboost": CatBoostRegressor(learning_rate=0.2, l2_leaf_reg=7, iterations=100, depth=10, verbose=0),
}

# Convert pandas series to numpy arrays
total_fc = X_test['total.fc'].values
FC_sum = X_test['FC_sum'].values / 1000000

X_test.drop(columns=results_var, inplace=True)

for model in models:
    pipe = make_pipeline(preprocessor_trees, models[model])

    # Generate x values and fitted y values
    pipe.fit(X_train, y_train)
    predicted_residual = pipe.predict(X_test)
    x_values = np.log(total_fc)
    y_values = np.log(FC_sum) + predicted_residual

    # Calculate R-squared
    r_squared = r2_score(x_values, y_values)
    # Calculate correlation
    correlation = np.corrcoef(x_values, y_values)[0, 1]

    # Calculate mean squared error
    mse = mean_squared_error(x_values, y_values)
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.xlim(4, 10)
    plt.ylim(4, 10)
    plt.scatter(x_values, y_values, label='Data')

    # Plot x=y reference line
    plt.plot([4, 10], [4, 10], 'g--', label='x=y line')

    # Set x and y labels
    plt.xlabel('Reported Fuel Consumption')
    plt.ylabel('Estimated Fuel Consumption')

    # Set a title for the plot
    plt.title('Comparison of Reported and Estimated Fuel Consumption (m tonnes) using ' + model)

    # Add details in legend about correlation, r-squared and mean squared error
    legend_title = 'Correlation: {:.5f}\nR²: {:.5f}\nMSE: {:.5f}'.format(correlation, r_squared, mse)
    plt.legend(title=legend_title)

    # Show the plot
    plt.show()