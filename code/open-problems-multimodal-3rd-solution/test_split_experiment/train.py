import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from models import CiteModel, CiteModel_mish, OneLayerPerceptron, OneLayerPerceptronV2, ThreeLayersPerceptron, ThreeLayersPerceptronV2, FiveLayersPerceptron, FiveLayersPerceptronV2
from utils import CiteDataset, CiteDataset_test, train_loop, valid_loop


device = torch.device("cuda")


# This is modified Makotu's code from open-problems-multimodal-3rd-solution/code/4.model/train/cite/cite-mlp.py

preprocess_path = '../input/preprocess/cite/'
validation_path = '../input/fold/'

target_path = '../input/target/'
feature_path = '../input/features/cite/'
output_path = '../model/cite/mlp/'


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

train_file = "X_svd_128.pickle"
test_file = "X_test_svd_128.pickle"

X = pd.read_pickle(feature_path  + train_file)
X_test = pd.read_pickle(feature_path  + test_file)

X = np.array(X)
X_test = np.array(X_test)

feature_dims = X.shape[1]

models = (CiteModel, CiteModel_mish, OneLayerPerceptron, OneLayerPerceptronV2, ThreeLayersPerceptron,
          ThreeLayersPerceptronV2, FiveLayersPerceptron, FiveLayersPerceptronV2)
model_names = ("MakotuCiteModel", "CiteModelMish", "OneLayerPerceptron", "OneLayerPerceptronV2", "ThreeLayersPerceptron",
               "ThreeLayersPerceptronV2", "FiveLayersPerceptron", "FiveLayersPerceptronV2")

# ### train

es = 30
check_round = 5
epochs = 100000
target_num = Y.shape[1]
full_data_epochs = 50
save_model = True # If you want to save the model in the output path, set this to True.

validation_results = []

try:
    calculated_results = pd.read_csv(f'{output_path}/validation_results.csv')
except FileNotFoundError:
    calculated_results = pd.DataFrame()
    

for model_class, model_name in zip(models, model_names):

    print("Start training", model_name)
    
    if calculated_results.shape[0] and model_name in calculated_results["model_name"].values:
        print("Already have results, skipping")
        continue

    # test_ds = CiteDataset_test(X_test)
    # test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True,
    #                              shuffle=False, drop_last=False, num_workers=2)
    
    
    print("Checking validation scores")

    for val_strategy in validation_cols:

        print("Validation strategy:", val_strategy)

        train_index = fold_df[fold_df[val_strategy] == 0].index
        val_index = fold_df[fold_df[val_strategy] == 1].index

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        train_ds = CiteDataset(X_train, y_train)
        train_dataloader = DataLoader(train_ds,
                                    batch_size=128,
                                    pin_memory=True,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=2)

        print(X_train.shape, X_val.shape)

        val_ds = CiteDataset(X_val, y_val)
        val_dataloader = DataLoader(val_ds,
                                    batch_size=128,
                                    pin_memory=True,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=2)

        model = model_class(feature_dims)

        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        best_corr = 0
        es_counter = 0

        for epoch in range(epochs):

            model = train_loop(model, opt, train_dataloader, epoch)
            logit, val_cor = valid_loop(model, val_dataloader, y_val)

            if epoch == 0:
                logit_best = logit

            if val_cor > best_corr:
                best_corr = val_cor
                logit_best = logit
                es_counter = 0
            else:
                es_counter += 1

            if es_counter == es:
                break

        best_epoch = epoch - es
        print(f'best epoch:{best_epoch}, best corr: {best_corr}')

        validation_results.append([model_name, val_strategy, best_epoch, best_corr])

    print("Training model with all data on", full_data_epochs, "epochs")

    # full dataset train
    train_ds = CiteDataset(X, Y)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=128,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=2)

    # train new model
    model = CiteModel(feature_dims)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(full_data_epochs):
        model = train_loop(model, opt, train_dataloader, epoch)

    if save_model == True:
        torch.save(model.state_dict(), f'{output_path}/{model_name}_{full_data_epochs}')
        results_df = pd.DataFrame(validation_results, columns = ['model_name', 'val_strategy', 'best_epoch', 'best_score'])
        results_df = pd.concat([calculated_results, results_df])
        results_df.to_csv(f'{output_path}/validation_results.csv', index = False)
        
    print()
        
validation_results = pd.DataFrame(validation_results, columns = ['model_name', 'val_strategy', 'best_epoch', 'best_score'])
validation_results.to_csv(f'{output_path}/validation_results.csv', index = False)
