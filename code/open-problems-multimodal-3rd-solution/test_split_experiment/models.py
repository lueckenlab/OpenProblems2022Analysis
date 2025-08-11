import torch
import torch.nn as nn

class CiteModel(nn.Module):
    """Makotu original model"""

    def __init__(self, feature_num):
        super(CiteModel, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.ReLU(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.ReLU(),
                                      )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out


class CiteModel_mish(nn.Module):
    """Makotu original model with Mish activation function"""

    def __init__(self, feature_num):
        super(CiteModel_mish, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.Mish(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.Mish(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.Mish(),
                                      )

        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out


class OneLayerPerceptron(nn.Module):
    """One layer perceptron"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(feature_num, 128),
                                   nn.LayerNorm(128),
                                   nn.ReLU()
                                   )

        self.head = nn.Linear(128, 140)

    def forward(self, X, y=None):

        X = self.layer(X)
        out = self.head(X)

        return out


class OneLayerPerceptronV2(nn.Module):
    """One layer perceptron with 256 neurons in the hidden layer"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(feature_num, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU()
                                   )

        self.head = nn.Linear(256, 140)

    def forward(self, X, y=None):

        X = self.layer(X)
        out = self.head(X)

        return out
    
class ThreeLayersPerceptron(nn.Module):
    """Three layer perceptron with ReLU activation functions"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer_1 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                      )
        self.layer_2 = nn.Sequential(nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_3 = nn.Linear(128, 140)

    def forward(self, X, y=None):

        X = self.layer_1(X)
        X = self.layer_2(X)
        out = self.layer_3(X)

        return out


class ThreeLayersPerceptronV2(nn.Module):
    """Three layer perceptron with ReLU activation functions and 256 neurons in the hidden layers"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer_1 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                      )
        self.layer_2 = nn.Sequential(nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                      )
        self.layer_3 = nn.Linear(256, 140)

    def forward(self, X, y=None):

        X = self.layer_1(X)
        X = self.layer_2(X)
        out = self.layer_3(X)

        return out
    

class FiveLayersPerceptron(nn.Module):
    """Five layer perceptron with ReLU activation functions"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer_1 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                      )
        self.layer_2 = nn.Sequential(nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_3 = nn.Sequential(nn.Linear(128, 128),
                                        nn.LayerNorm(128),
                                        nn.ReLU(),
                                    )
        self.layer_4 = nn.Sequential(nn.Linear(128, 128),
                                        nn.LayerNorm(128),
                                        nn.ReLU(),
                                    )
        self.layer_5 = nn.Linear(128, 140)

    def forward(self, X, y=None):

        X = self.layer_1(X)
        X = self.layer_2(X)
        X = self.layer_3(X)
        X = self.layer_4(X)
        out = self.layer_5(X)

        return out
    
class FiveLayersPerceptronV2(nn.Module):
    """Five layer perceptron with ReLU activation functions and more neurons in the hidden layers"""

    def __init__(self, feature_num):
        super().__init__()

        self.layer_1 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                                      )
        self.layer_2 = nn.Sequential(nn.Linear(256, 200),
                                       nn.LayerNorm(200),
                                       nn.ReLU(),
                                      )
        self.layer_3 = nn.Sequential(nn.Linear(200, 180),
                                        nn.LayerNorm(180),
                                        nn.ReLU(),
                                    )
        self.layer_4 = nn.Sequential(nn.Linear(180, 128),
                                        nn.LayerNorm(128),
                                        nn.ReLU(),
                                    )
        self.layer_5 = nn.Linear(128, 140)

    def forward(self, X, y=None):

        X = self.layer_1(X)
        X = self.layer_2(X)
        X = self.layer_3(X)
        X = self.layer_4(X)
        out = self.layer_5(X)

        return out
