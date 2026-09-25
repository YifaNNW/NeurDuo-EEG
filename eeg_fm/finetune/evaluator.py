"""Validation and test metrics for classification, regression and sequence labelling, adapted from CBraMod."""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
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
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().reshape(-1).numpy().tolist()
            preds += pred_y.cpu().reshape(-1).numpy().tolist()
            scores += score_y.cpu().reshape(-1).numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_seq2seq(self, model, return_predictions=False):
        model.eval()
        truths, preds, probs = [], [], []
        for x, y, m in tqdm(self.data_loader, mininterval=1):
            x, y, m = x.cuda(), y.cuda(), m.cuda()
            logit = model(x, m)
            p = torch.softmax(logit, dim=-1)
            keep = m.reshape(-1)
            truths += y.reshape(-1)[keep].cpu().numpy().tolist()
            preds += p.reshape(-1, p.shape[-1])[keep].argmax(-1).cpu().numpy().tolist()
            if return_predictions:
                probs.append(p.reshape(-1, p.shape[-1])[keep].cpu().numpy())
        truths = np.array(truths)
        preds = np.array(preds)
        n_cls = self.params.num_of_classes
        labels = list(range(n_cls))
        out = dict(
            macro_f1=float(f1_score(truths, preds, average='macro', labels=labels,
                                    zero_division=0)),
            weighted_f1=float(f1_score(truths, preds, average='weighted', labels=labels,
                                       zero_division=0)),
            accuracy=float((truths == preds).mean()),
            balanced_acc=float(balanced_accuracy_score(truths, preds)),
            kappa=float(cohen_kappa_score(truths, preds, labels=labels)),
            per_class_f1=[float(v) for v in f1_score(truths, preds, average=None,
                                                     labels=labels, zero_division=0)],
            n_epochs=int(len(truths)),
        )
        cm = confusion_matrix(truths, preds, labels=labels)
        if return_predictions:
            return out, cm, dict(y_true=truths, y_pred=preds,
                                 y_prob=np.concatenate(probs) if probs else None)
        return out, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().reshape(-1).numpy().tolist()
            preds += pred.cpu().reshape(-1).numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse
