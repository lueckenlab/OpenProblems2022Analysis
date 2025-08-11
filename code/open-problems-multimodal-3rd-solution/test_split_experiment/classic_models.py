from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


makotu_catboost_parameters = {
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 2,
    "loss_function": "RMSE",
    "task_type": "GPU",
    "iterations": 100000, # 800
    "od_type": "Iter",
    "boosting_type": "Plain",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.3,
    "allow_const_label": True,
    "random_state": 42,
    "verbose": 0
}

models = (
    LinearRegression,
    Ridge,
    Ridge,
    Ridge,
    Lasso,
    Lasso,
    Lasso,
    ElasticNet,
    ElasticNet,
    ElasticNet,
#     RandomForestRegressor,  # Removed: too slow
#     RandomForestRegressor,  # Removed: too slow
#     RandomForestRegressor,  # Removed: too slow
    XGBRegressor,
    XGBRegressor,
    XGBRegressor,
    XGBRegressor,
    LGBMRegressor,
    LGBMRegressor,
    LGBMRegressor,
    CatBoostRegressor,  # Is it possible to directly optimise task's loss?
    CatBoostRegressor,
#     SVR,  # Removed: too slow
)

models_params = (
    dict(),
    dict(alpha=0.1),
    dict(alpha=1.0),
    dict(alpha=10),
    dict(alpha=0.1),
    dict(alpha=1.0),
    dict(alpha=10),
    dict(alpha=0.1),
    dict(alpha=1.0),
    dict(alpha=10),
#     dict(min_samples_leaf=1, n_jobs=-1, max_depth=10),  # Removed: too slow
#     dict(min_samples_leaf=50, n_jobs=-1, max_depth=10),  # Removed: too slow
#     dict(min_samples_leaf=500, n_jobs=-1, max_depth=10),  # Removed: too slow
    dict(learning_rate=0.01, n_jobs=-1, tree_method="gpu_hist"),
    dict(learning_rate=0.1, n_jobs=-1, tree_method="gpu_hist"),
    dict(learning_rate=1, n_jobs=-1, tree_method="gpu_hist"),
    dict(learning_rate=10, n_jobs=-1, tree_method="gpu_hist"),
    dict(min_data_in_leaf=10, n_jobs=-1),
    dict(min_data_in_leaf=100, n_jobs=-1),
    dict(min_data_in_leaf=1000, n_jobs=-1),
    dict(verbose=0, task_type="GPU"),
    makotu_catboost_parameters,
#     dict(kernel="rbf"),  # Removed: too slow
)

model_names = (
    "LinearRegression",
    "Ridge_0.1",
    "Ridge_1",
    "Ridge_10",
    "Lasso_0.1",
    "Lasso_1",
    "Lasso_10",
    "ElasticNet_0.1",
    "ElasticNet_1",
    "ElasticNet_10",
#     "RandomForestRegressor_1",  # Removed: too slow
#     "RandomForestRegressor_50",  # Removed: too slow
#     "RandomForestRegressor_500",  # Removed: too slow
    "XGBRegressor_0.01",
    "XGBRegressor_0.1",
    "XGBRegressor_1",
    "XGBRegressor_10",
    "LGBMRegressor_10",
    "LGBMRegressor_100",
    "LGBMRegressor_1000",
    "CatBoostRegressor_default",
    "CatBoostRegressor_makotu",
#     "SVR_rbf",  # Removed: too slow
)
