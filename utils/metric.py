import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prediction(l):
    l_arr = torch.from_numpy(np.vstack(l))
    _, pred = torch.max(F.softmax(l_arr, -1).data, -1)
    prob = F.softmax(l_arr, -1).data[:, 1:].squeeze(-1)
    return pred, prob


def cal_all_metric(logit, grou):
    pred, prob = prediction(logit)
    grou = np.hstack(grou).astype(np.float64)
    pred = pred.detach().cpu().numpy().astype(np.float64)
    prob = prob.detach().cpu().numpy().astype(np.float64)

    acc = accuracy_score(grou, pred)
    auc = roc_auc_score(grou, prob)

    tn, fp, fn, tp = confusion_matrix(grou, pred).ravel()
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)

    macro_prec = precision_score(grou, pred, average='macro')
    macro_recl = recall_score(grou, pred, average='macro')
    macro_f1sc = f1_score(grou, pred, average='macro')

    results = [round(acc, 4), round(auc, 4), round(sen, 4), round(spec, 4),
               round(macro_prec, 4), round(macro_recl, 4), round(macro_f1sc, 4)]
    return results


def cal_acc_metric(logit, grou):
    pred, _ = prediction(logit)
    grou = np.hstack(grou).astype(np.float64)
    pred = pred.detach().cpu().numpy().astype(np.float64)
    acc = accuracy_score(grou, pred)
    return acc


def best_epoch_min(loss):

    loss_ep = np.argwhere(loss == np.min(loss)).flatten().tolist()
    best_loss_ep = loss_ep

    return best_loss_ep

def best_epoch_max(loss):

    loss_ep = np.argwhere(loss == np.max(loss)).flatten().tolist()

    return loss_ep