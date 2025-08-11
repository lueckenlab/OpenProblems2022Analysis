import os
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from classic_models import models, models_params, model_names
from utils import correl_loss, std

random.seed(42)
np.random.seed(42)


preprocess_path = "../input/preprocess/cite/"
validation_path = "../input/fold/"

target_path = "../input/target/"
feature_path = "../input/features/cite/"
model_path = "../model/cite/classic/"

train_file = "X_svd_128.pickle"
test_file = "X_test_svd_128.pickle"

df_meta = pd.read_csv(validation_path + "validation_splits.csv")
cell_ids = np.load(preprocess_path + 'train_cite_inputs_idxcol.npz', allow_pickle=True)

fold_df = pd.DataFrame(list(cell_ids['index']), columns = ['cell_id'])

validation_cols = ("adversarial_val", "random_third_val", "random_fifth_val", "random_tenth_val", "day_2_val",
                   "day_4_val", "donor_32606_val", "donor_31800_val", "day_2_donor_32606_val", "day_2_donor_31800_val",
                   "day_4_donor_32606_val", "day_4_donor_31800_val")

fold_df = fold_df.merge(df_meta[['cell_id', 'rank_per_donor', *validation_cols]], on = ['cell_id'], how = 'inner')

del cell_ids

Y = pd.read_hdf(target_path + "train_cite_targets.h5")
Y = np.array(Y)

X = pd.read_pickle(feature_path + train_file)
X_test = pd.read_pickle(feature_path + test_file)

X = np.array(X)
X_test = np.array(X_test)

feature_dims = X.shape[1]

validation_results = []

for model_class, model_params, model_name in zip(models, models_params, model_names):
    print("Working with", model_name)

    cur_model_dir = f"{model_path}{model_name}"

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(cur_model_dir):
        os.mkdir(cur_model_dir)
        
    for val_strategy in tqdm(validation_cols):
        
        train_index = fold_df[fold_df[val_strategy] == 0].index
        val_index = fold_df[fold_df[val_strategy] == 1].index

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        val_scores = np.zeros(Y.shape[1])

        for target_idx in range(Y.shape[1]):

            model = model_class(**model_params)
            model.fit(X_train, y_train[:, target_idx])

            val_true = torch.Tensor(y_val[:, target_idx]).reshape(1, -1)
            val_pred = torch.Tensor(model.predict(X_val)).reshape(1, -1)
            val_score = -correl_loss(val_true, val_pred)

            val_scores[target_idx] = val_score.item()
        
        val_scores[np.isnan(val_scores)] = 0
        avg_val_score = np.mean(val_scores)
        print("Validation:", val_strategy, "average validation score:", avg_val_score)
        validation_results.append([model_name, val_strategy, avg_val_score])

    print("Training model on all data")
    
    pred = np.zeros([48203, 140])
    
    for target_idx in range(Y.shape[1]):
        model = model_class(**model_params)
        model.fit(X, Y[:, target_idx])
        pred[:, target_idx] = model.predict(X_test)

        with open(f"{cur_model_dir}/{target_idx}", "wb") as f:
            pickle.dump(model, file=f)
            
    pred = std(pred)
    
    cite_sub = pd.DataFrame(pred.round(6))
    cite_sub.to_csv(f"../summary/output/submit/{model_name}.csv", index=False)

validation_results = pd.DataFrame(validation_results, columns=("model_name", "val_strategy", "best_score"))
validation_results.to_csv("../model/cite/mlp/classic_validation_results.csv", index=False)
