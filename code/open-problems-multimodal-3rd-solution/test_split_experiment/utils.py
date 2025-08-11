import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda")

class CiteDataset(Dataset):

    def __init__(self, feature, target):

        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):

        d = {
            "X": self.feature[index],
            "y" : self.target[index],
        }
        return d


class CiteDataset_test(Dataset):

    def __init__(self, feature):
        self.feature = feature

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):

        d = {
            "X": self.feature[index]
        }
        return d
    

def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)


def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


def train_loop(model, optimizer, loader, epoch):

    model.train()
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()

    for d in loader:
        X = d['X'].to(device)
        y = d['y'].to(device)

        logits = model(X)
        loss = correl_loss(logits, y)
        #loss = torch.sqrt(loss_fn(logits, y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def valid_loop(model, loader, y_val):

    model.eval()
    oof_pred = []

    for d in loader:
        with torch.no_grad():
            val_X = d['X'].to(device).float()
            val_y = d['y'].to(device)
            logits = model(val_X)
            oof_pred.append(logits)

    #print(torch.cat(oof_pred).shape, torch.cat(oof_pred).detach().cpu().numpy().shape)
    cor = partial_correlation_score_torch_faster(torch.tensor(y_val).to(device), torch.cat(oof_pred))
    cor = cor.mean().item()
    logits = torch.cat(oof_pred).detach().cpu().numpy()

    return logits, cor


def std(x):
    x = np.array(x)
    return (x - x.mean(1).reshape(-1, 1)) / x.std(1).reshape(-1, 1)


def test_loop(model, loader):
    
    model.eval()
    predicts=[]

    for d in loader:
        with torch.no_grad():
            X = d['X'].to(device)
            logits = model(X)
            predicts.append(logits.detach().cpu().numpy())
            
    return np.concatenate(predicts)
