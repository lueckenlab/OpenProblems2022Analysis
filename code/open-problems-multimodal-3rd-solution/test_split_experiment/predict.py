import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from models import CiteModel, CiteModel_mish, OneLayerPerceptron, OneLayerPerceptronV2, ThreeLayersPerceptron, ThreeLayersPerceptronV2, FiveLayersPerceptron, FiveLayersPerceptronV2
from utils import CiteDataset_test, std, test_loop

device = torch.device("cuda")


raw_path =  '../input/raw/'

cite_target_path = '../input/target/cite/'
feature_path = '../input/features/cite/'
cite_mlp_path = '../model/cite/mlp/'


output_path = '../output/'

models = (CiteModel, CiteModel_mish, OneLayerPerceptron, OneLayerPerceptronV2, ThreeLayersPerceptron,
          ThreeLayersPerceptronV2, FiveLayersPerceptron, FiveLayersPerceptronV2)
model_names = ("MakotuCiteModel", "CiteModelMish", "OneLayerPerceptron", "OneLayerPerceptronV2", "ThreeLayersPerceptron",
               "ThreeLayersPerceptronV2", "FiveLayersPerceptron", "FiveLayersPerceptronV2")
full_data_epochs = 50


test_file = "X_test_svd_128.pickle"
X_test = pd.read_pickle(feature_path + test_file)
X_test = np.array(X_test)
feature_dims = X_test.shape[1]

for model_class, model_name in zip(models, model_names):
    print("Start training", model_name)
    
    pred = np.zeros([48203, 140])

    test_ds = CiteDataset_test(X_test)
    test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                                    shuffle=False, drop_last=False, num_workers=2)

    model = model_class(feature_dims)
        
    model = model.to(device)
    model.load_state_dict(torch.load(f"{cite_mlp_path}/{model_name}_{full_data_epochs}"))

    result = test_loop(model, test_dataloader).astype(np.float32)
    pred += std(result)

    torch.cuda.empty_cache()

    cite_sub = pd.DataFrame(pred.round(6))

    cite_sub.to_csv(f"../summary/output/submit/{model_name}_{full_data_epochs}.csv", index=False)
