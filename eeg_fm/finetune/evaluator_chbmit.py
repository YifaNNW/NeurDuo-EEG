"""Seizure-detection metrics for CHB-MIT: AUROC, AUC-PR and false alarms per hour."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from tqdm import tqdm

from eeg_fm.finetune.evaluator import Evaluator

UNIT_SEC = 30.0


def _events(mask):
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [len(mask)]
    return list(zip(starts, stops))


def _fp_per_hour(y, score, thr, hours):
    pred = score >= thr
    ev = _events(pred)
    fp = sum(1 for a, b in ev if not y[a:b].any())
    return fp / hours if hours > 0 else float('nan')


def _thr_at_sensitivity(y, score, target):
    pos = score[y == 1]
    if len(pos) == 0:
        return float('nan')
    return float(np.quantile(pos, 1.0 - target))


class ChbmitEvaluator(Evaluator):
    def get_metrics_for_seq2seq(self, model, return_predictions=False):
        model.eval()
        truths, scores = [], []
        for x, y, m in tqdm(self.data_loader, mininterval=1):
            x, y, m = x.cuda(), y.cuda(), m.cuda()
            logit = model(x, m)
            p = torch.softmax(logit, dim=-1)[..., 1]
            keep = m.reshape(-1)
            truths += y.reshape(-1)[keep].cpu().numpy().tolist()
            scores += p.reshape(-1)[keep].float().cpu().numpy().tolist()

        y = np.asarray(truths, np.int8)
        s = np.asarray(scores, np.float64)
        hours = len(y) * UNIT_SEC / 3600.0
        npos = int(y.sum())

        if npos == 0 or npos == len(y):
            out = dict(auroc=float('nan'), auc_pr=float('nan'),
                       fp_per_hour=float('nan'), n_epochs=int(len(y)),
                       n_positive=npos, pos_rate=float(y.mean()), hours=float(hours))
            return (out, np.zeros((2, 2), int)) if not return_predictions else \
                   (out, np.zeros((2, 2), int), dict(y_true=y, y_score=s))

        auroc = float(roc_auc_score(y, s))
        prec, rec, _ = precision_recall_curve(y, s)
        auc_pr = float(auc(rec, prec))

        out = dict(
            auroc=auroc,
            auc_pr=auc_pr,
            auc_pr_baseline=float(y.mean()),
            n_epochs=int(len(y)), n_positive=npos,
            pos_rate=float(y.mean()), hours=float(hours),
        )
        for tgt in (0.50, 0.75, 0.90):
            thr = _thr_at_sensitivity(y, s, tgt)
            out[f'fp_per_hour_at_sens{int(tgt*100)}'] = float(_fp_per_hour(y, s, thr, hours))
        out['fp_per_hour'] = out['fp_per_hour_at_sens50']
        thr50 = _thr_at_sensitivity(y, s, 0.50)
        out['threshold_at_sens50'] = float(thr50)

        cm = confusion_matrix(y, (s >= thr50).astype(np.int8), labels=[0, 1])
        if return_predictions:
            return out, cm, dict(y_true=y, y_score=s)
        return out, cm
