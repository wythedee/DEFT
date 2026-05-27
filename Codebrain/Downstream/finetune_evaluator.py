import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]
            if pred.ndim == 3 and y.ndim == 2:
                truths += y.cpu().reshape(-1).numpy().tolist()
                preds += pred_y.cpu().reshape(-1).numpy().tolist()
            else:
                truths += y.cpu().squeeze().numpy().tolist()
                preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x).view(-1)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().view(-1).numpy().tolist()
            preds += pred_y.cpu().view(-1).numpy().tolist()
            scores += score_y.cpu().view(-1).numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        kappa = cohen_kappa_score(truths, preds)
        f1 = f1_score(truths, preds, average='binary')
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, kappa, f1, cm
