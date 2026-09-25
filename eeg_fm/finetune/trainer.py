"""Fine-tuning loop with the common optimizer, schedule, model selection and test evaluation, adapted from CBraMod."""

import copy
import math
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from eeg_fm.finetune.evaluator import Evaluator


def _evaluator_cls(params):
    if str(getattr(params, 'dataset', '')).startswith('chbmit'):
        from eeg_fm.finetune.evaluator_chbmit import ChbmitEvaluator
        return ChbmitEvaluator
    return Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        _E = _evaluator_cls(params)
        self.val_eval = _E(params, data_loader['val'])
        self.test_eval = _E(params, data_loader['test'])
        self.model = model.cuda()
        self.task = getattr(params, 'task', 'multiclass')
        self.ckpt_path = getattr(params, 'ckpt_path', None)
        self.resume = getattr(params, 'resume', False)
        if self.task == 'regression':
            self.criterion = MSELoss().cuda()
        elif self.task == 'binary':
            self.criterion = BCEWithLogitsLoss().cuda()
        else:
            self.criterion = CrossEntropyLoss(label_smoothing=params.label_smoothing).cuda()
        self.best_model_states = None

        warmup_epochs = getattr(params, 'warmup_epochs', 0)
        head_lr = 0.001 * (params.batch_size / 256) ** 0.5
        print(f"[trainer] backbone lr {params.lr:.3e} | head lr {head_lr:.3e}", flush=True)

        backbone_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)
        groups = [{'params': backbone_params, 'lr': params.lr},
                  {'params': other_params, 'lr': head_lr}]
        self.optimizer = torch.optim.AdamW(groups, weight_decay=params.weight_decay)

        self.grad_accum = max(1, int(getattr(params, 'grad_accum', 1)))
        steps = math.ceil(len(data_loader['train']) / self.grad_accum)
        total_steps = params.epochs * steps
        warmup_steps = warmup_epochs * steps
        if warmup_steps > 0:
            assert warmup_epochs < params.epochs, \
                (f"warmup_epochs {warmup_epochs} >= epochs {params.epochs}: "
                 f"cosine would get T_max={total_steps - warmup_steps}")
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer, start_factor=1e-3, total_iters=warmup_steps),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
                ],
                milestones=[warmup_steps])
            print(f"[trainer] warmup {warmup_epochs} epochs ({warmup_steps} steps), "
                  f"then cosine over {total_steps - warmup_steps} steps", flush=True)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-6)

    def _save_ckpt(self, epoch, best, best_epoch, best_metrics):
        if not self.ckpt_path:
            return
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        tmp = self.ckpt_path + '.tmp'
        torch.save(dict(epoch=epoch, best=best, best_epoch=best_epoch,
                        best_metrics=best_metrics, geom=self._geom(),
                        model=self.model.state_dict(),
                        best_model_states=self.best_model_states,
                        optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(),
                        rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state()), tmp)
        os.replace(tmp, self.ckpt_path)

    def _geom(self):
        p = self.params
        return dict(dataset=getattr(p, 'dataset', None),
                    n_outputs=getattr(p, 'num_of_classes', None),
                    seg_sec=getattr(p, 'seg_sec', None),
                    context_len=getattr(p, 'context_len', None),
                    channels=len(getattr(p, 'channels', None) or []))

    def _load_ckpt(self):
        if not (self.ckpt_path and self.resume and os.path.exists(self.ckpt_path)):
            return 0, None, 0, None
        ck = torch.load(self.ckpt_path, map_location='cuda', weights_only=False)
        now, was = self._geom(), ck.get('geom')
        if was != now:
            raise RuntimeError(
                f"checkpoint {self.ckpt_path} was written for a different data geometry "
                f"({was} vs {now}); remove it before resuming")
        self.model.load_state_dict(ck['model'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.scheduler.load_state_dict(ck['scheduler'])
        self.best_model_states = ck['best_model_states']
        torch.set_rng_state(ck['rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        np.random.set_state(ck['numpy_rng'])
        print(f"[trainer] resumed from {self.ckpt_path} at epoch {ck['epoch']} "
              f"(best {ck['best']:.5f} @ epoch {ck['best_epoch']})", flush=True)
        return ck['epoch'], ck['best'], ck['best_epoch'], ck['best_metrics']

    def _optimizer_step(self):
        if self.params.clip_value > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.params.clip_value)
            if not torch.isfinite(total_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self._n_skipped = getattr(self, '_n_skipped', 0) + 1
                return
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def train(self):
        if self.task == 'regression':
            return self.train_for_regression()
        if self.task == 'binary':
            return self.train_for_binary()
        if self.task == 'seq2seq':
            return self.train_for_seq2seq()
        return self.train_for_multiclass()

    def train_for_seq2seq(self):
        start, best, best_epoch, best_metrics = self._load_ckpt()
        best = -1. if best is None else best
        sel = getattr(self.params, 'seq_select_key', 'kappa')
        accum = self.grad_accum
        for epoch in range(start, self.params.epochs):
            self.model.train()
            t0 = timer(); losses = []
            n = len(self.data_loader['train'])
            self.optimizer.zero_grad()
            for i, (x, y, m) in enumerate(self.data_loader['train']):
                x, y = x.cuda(), y.cuda()
                logit = self.model(x, m.cuda())
                loss = self.criterion(logit.reshape(-1, logit.shape[-1]), y.reshape(-1))
                group = min(accum, n - (i // accum) * accum)
                (loss / group).backward()
                losses.append(loss.item())
                if (i + 1) % accum and i + 1 < n:
                    continue
                self._optimizer_step()
            with torch.no_grad():
                mt, _ = self.val_eval.get_metrics_for_seq2seq(self.model)
            _sk = getattr(self, '_n_skipped', 0)
            _show = [k for k in ('kappa', 'macro_f1', 'accuracy',
                                 'auroc', 'auc_pr', 'fp_per_hour') if k in mt]
            print(f"Epoch {epoch+1}: loss {np.mean(losses):.5f}"
                  + (f" | SKIPPED {_sk} non-finite updates" if _sk else "")
                  + " | val " + ' '.join(f"{k} {mt[k]:.5f}" for k in _show)
                  + (f" | per-class F1 {[round(v, 3) for v in mt['per_class_f1']]}"
                     if 'per_class_f1' in mt else "")
                  + f" | lr {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
                  + f" | {(timer()-t0)/60:.2f}min", flush=True)
            if mt[sel] > best:
                best, best_epoch, best_metrics = mt[sel], epoch + 1, mt
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                print(f"  * new best val {sel} {best:.5f}", flush=True)
            self._save_ckpt(epoch + 1, best, best_epoch, best_metrics)

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            mt, cm, pred = self.test_eval.get_metrics_for_seq2seq(
                self.model, return_predictions=True)
        _tshow = [k for k in ('kappa', 'macro_f1', 'accuracy',
                              'auroc', 'auc_pr', 'fp_per_hour') if k in mt]
        print(f"***TEST (best val-{sel} epoch {best_epoch}): "
              + ' '.join(f"{k} {mt[k]:.5f}" for k in _tshow), flush=True)
        if 'per_class_f1' in mt:
            print('per-class F1:', [round(v, 4) for v in mt['per_class_f1']], flush=True)
        print(cm, flush=True)
        self.test_predictions = pred
        res = dict(best_epoch=best_epoch, confusion_matrix=cm.tolist())
        res.update({f'val_{k}': v for k, v in best_metrics.items()})
        res.update({f'test_{k}': v for k, v in mt.items()})
        return res

    def train_for_binary(self):
        auc_best, acc_best, pr_best, best_epoch = -1., 0., 0., 0
        for epoch in range(self.params.epochs):
            self.model.train()
            t0 = timer(); losses = []
            for x, y in self.data_loader['train']:
                self.optimizer.zero_grad()
                x, y = x.cuda(), y.cuda().float().reshape(-1)
                loss = self.criterion(self.model(x).reshape(-1), y)
                loss.backward()
                losses.append(loss.item())
                self._optimizer_step()
            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
            print(f"Epoch {epoch+1}: loss {np.mean(losses):.5f} | val acc {acc:.5f} pr_auc {pr_auc:.5f} "
                  f"roc_auc {roc_auc:.5f} | "
                  f"lr {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f} | {(timer()-t0)/60:.2f}min", flush=True)
            if roc_auc > auc_best:
                acc_best, pr_best, auc_best, best_epoch = acc, pr_auc, roc_auc, epoch + 1
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                print(f"  * new best val roc_auc {roc_auc:.5f}", flush=True)

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
        print(f"***TEST (best val-roc_auc epoch {best_epoch}): acc {acc:.5f} pr_auc {pr_auc:.5f} "
              f"roc_auc {roc_auc:.5f}", flush=True)
        print(cm, flush=True)
        return dict(best_epoch=best_epoch, val_roc_auc=float(auc_best), val_acc=float(acc_best),
                    val_pr_auc=float(pr_best), test_acc=float(acc), test_pr_auc=float(pr_auc),
                    test_roc_auc=float(roc_auc))

    def train_for_regression(self):
        r2_best, corr_best, rmse_best, best_epoch = -float('inf'), 0., 0., 0
        for epoch in range(self.params.epochs):
            self.model.train()
            t0 = timer(); losses = []
            for x, y in self.data_loader['train']:
                self.optimizer.zero_grad()
                x, y = x.cuda(), y.cuda().float().reshape(-1)
                pred = self.model(x).reshape(-1)
                loss = self.criterion(pred, y)
                loss.backward()
                losses.append(loss.item())
                self._optimizer_step()
            with torch.no_grad():
                corr, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
            print(f"Epoch {epoch+1}: loss {np.mean(losses):.5f} | val corr {corr:.5f} r2 {r2:.5f} rmse {rmse:.5f} | "
                  f"lr {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f} | {(timer()-t0)/60:.2f}min", flush=True)
            if r2 > r2_best:
                r2_best, corr_best, rmse_best, best_epoch = r2, corr, rmse, epoch + 1
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                print(f"  * new best val r2 {r2:.5f}", flush=True)

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            corr, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
        print(f"***TEST (best val-r2 epoch {best_epoch}): corr {corr:.5f} r2 {r2:.5f} rmse {rmse:.5f}",
              flush=True)
        return dict(best_epoch=best_epoch, val_r2=float(r2_best), val_corrcoef=float(corr_best),
                    val_rmse=float(rmse_best), test_corrcoef=float(corr), test_r2=float(r2),
                    test_rmse=float(rmse))

    def train_for_multiclass(self):
        kappa_best, acc_best, f1_best, best_epoch = -1, 0, 0, 0
        for epoch in range(self.params.epochs):
            self.model.train()
            t0 = timer(); losses = []
            for x, y in self.data_loader['train']:
                self.optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                losses.append(loss.item())
                self._optimizer_step()
            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
            print(f"Epoch {epoch+1}: loss {np.mean(losses):.5f} | val acc {acc:.5f} kappa {kappa:.5f} f1 {f1:.5f} | "
                  f"lr {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f} | {(timer()-t0)/60:.2f}min", flush=True)
            if kappa > kappa_best:
                acc_best, kappa_best, f1_best, best_epoch = acc, kappa, f1, epoch + 1
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                print(f"  * new best val kappa {kappa:.5f}", flush=True)

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
        print(f"***TEST (best val-kappa epoch {best_epoch}): acc {acc:.5f} kappa {kappa:.5f} f1 {f1:.5f}", flush=True)
        print(cm, flush=True)
        return dict(best_epoch=best_epoch, val_kappa=kappa_best, val_acc=acc_best, val_f1=f1_best,
                    test_acc=float(acc), test_kappa=float(kappa), test_f1=float(f1))
