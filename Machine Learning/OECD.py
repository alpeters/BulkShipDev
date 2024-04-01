import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV


if __name__ == '__main__':
    df_ml = pd.read_csv(r'df_interp_ml.csv', low_memory=False)
    df_OECD = pd.read_csv(r'src/tracked_data/df_ml_rel_train.csv')
    # our features
    features_we_used = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
                'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
                'Bale.Capacity..cu.m.', 'LBP..m.', 'Speed..knots.',
                'West.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'EU.distance',
                'distance_sum', 'work_sum', 'trip_nunique',
                'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
                't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
                'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age', 'Size.Category']

    log_transform_cols_we_used = ['Draught..m.', 'HP.Total.Propulsion', 'Service.Speed..knots.', 'LOA..m.',
                          'Operational.Speed..knots.', 'NT', 'Ballast.Cap..cu.m.',
                          'Bale.Capacity..cu.m.', 'LBP..m.', 'Speed..knots.',
                          'West.Coast.Africa.Deployment..Time.in.Last.12.Months....', 'EU.distance',
                          'distance_sum', 'work_sum', 'trip_nunique',
                          'W_component_first', 'ME_W_ref_first', 't_m_times_v_n_sum',
                          't_over_t_ref_with_m_sum', 't_over_t_ref_without_m_sum',
                          'v_over_v_ref_with_n_sum', 'v_over_v_ref_without_n_sum', 'age']

    ordinal_cols = ['Size.Category']

    X = df_ml[features_we_used]
    y = df_ml['residual']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    median_imputer = SimpleImputer(strategy='median')
    standard_scaler = StandardScaler()
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    ordinal_transformer = OrdinalEncoder(categories=[['Handysize', 'Handymax', 'Panamax', 'Capesize']])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(median_imputer, log_transformer, standard_scaler), log_transform_cols_we_used),
            ('cat', ordinal_transformer, ordinal_cols),
        ])

    models = {
        "Linear Regression": make_pipeline(preprocessor, LinearRegression()),
        "Ridge": make_pipeline(preprocessor, Ridge(alpha=1)),
        "Lasso": make_pipeline(preprocessor, Lasso(alpha=0.0001, max_iter=1000000)),
        "Gradient Boosting": make_pipeline(preprocessor,
                                           GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                                                     max_depth=3, max_features='sqrt',
                                                                     min_samples_leaf=20, loss='huber',
                                                                     min_samples_split=20, warm_start=True)),
        "Random Forest": make_pipeline(preprocessor, RandomForestRegressor(n_estimators=1000))
    }

    param_distributions = {
        "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
        "Ridge": {
            "ridge__alpha": np.logspace(-4, 4, 20)
        },
        "Lasso": {
            "lasso__alpha": np.logspace(-4, 2, 20),
            "lasso__max_iter": [5000, 10000, 15000]  # Increased the range for max_iter
        },
        "Gradient Boosting": {
            "gradientboostingregressor__n_estimators": [100, 200, 500, 1000, 1500],
            "gradientboostingregressor__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
            "gradientboostingregressor__max_depth": [3, 4, 5, 6, 7, 8],
            "gradientboostingregressor__min_samples_split": np.linspace(2, 20, 10, dtype=int),
            "gradientboostingregressor__min_samples_leaf": np.linspace(1, 12, 6, dtype=int),
            "gradientboostingregressor__max_features": ['sqrt', 'log2', None]
        },
        "Random Forest": {
        "randomforestregressor__n_estimators": [100, 300, 500, 800, 1200],
        "randomforestregressor__max_depth": [None, 5, 10, 15, 20, 25],
        "randomforestregressor__min_samples_split": [2, 5, 10, 15, 100],
        "randomforestregressor__min_samples_leaf": [1, 2, 5, 10],
        "randomforestregressor__max_features": ['sqrt', 'log2', None]
}

    }

    # OECD features Todo!
    features_OECD_used = ['Dwt', 'Depth.Moulded..m.', 'Draught..m.', 'GT', 'LOA..m.', 'Speed..knots.',
                          'Main.Bunker.Capacity..m3.']
    log_transform_cols_OECD_used = ['Dwt', 'Depth.Moulded..m.', 'Draught..m.', 'GT', 'LOA..m.', 'Speed..knots.',
                                    'Main.Bunker.Capacity..m3.']
    X_OECD = df_OECD[features_OECD_used]
    y_OECD = df_OECD['report_fc']/df_OECD['MRV.EU.distance']
    X_OECD_train, X_OECD_test, y_OECD_train, y_OECD_test = train_test_split(X_OECD, y_OECD, test_size=0.25, random_state=1)

    median_imputer = SimpleImputer(strategy='median')
    standard_scaler = StandardScaler()
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    preprocessor_OECD = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(median_imputer, log_transformer, standard_scaler), log_transform_cols_OECD_used),
        ])

    models_OECD = {
        "Linear Regression": make_pipeline(preprocessor_OECD, LinearRegression()),
        "Ridge": make_pipeline(preprocessor_OECD, Ridge(alpha=1)),
        "Lasso": make_pipeline(preprocessor_OECD, Lasso(alpha=0.1, max_iter=1000000)),
        "Gradient Boosting": make_pipeline(preprocessor_OECD,
                                           GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                                                     max_depth=3, max_features='sqrt',
                                                                     min_samples_leaf=20, loss='huber',
                                                                     min_samples_split=20, warm_start=True)),
        "Random Forest": make_pipeline(preprocessor_OECD, RandomForestRegressor(n_estimators=1000))
    }


    def perform_k_fold_cv_metrics(models, X, y, k=3):
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
        results = []

        for name, model in models.items():
            mae_scores = []
            r2_scores = []

            # w/ parameter tuning
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions[name],
                                               scoring='r2', n_iter=10, cv=kf, verbose=1, random_state=1, n_jobs=-1)

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # w/o parameter tuning
                # model.fit(X_train, y_train)
                # predictions = model.predict(X_test)

                # w/ parameter tuning
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                predictions = best_model.predict(X_test)

                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                mae_scores.append(mae)
                r2_scores.append(r2)

            avg_mae = np.mean(mae_scores)
            avg_r2 = np.mean(r2_scores)
            results.append({'Model': name, 'Average MAE': avg_mae, 'Average R2': avg_r2})

        return pd.DataFrame(results).sort_values(by='Average MAE')

    # our result
    model_performance_mae_df = perform_k_fold_cv_metrics(models, X_train, y_train)
    model_performance_mae_df = model_performance_mae_df.reset_index(drop=True)
    print("Our results")
    print(model_performance_mae_df)
    # OECD result
    model_performance_mae_OECD_df = perform_k_fold_cv_metrics(models_OECD, X_OECD_train, y_OECD_train)
    model_performance_mae_OECD_df = model_performance_mae_OECD_df.reset_index(drop=True)
    print("OECD results:")
    print(model_performance_mae_OECD_df)