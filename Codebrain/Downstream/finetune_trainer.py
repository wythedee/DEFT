from tqdm import tqdm
import torch
try:
    from Downstream.finetune_evaluator import Evaluator
except ImportError:
    from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from timeit import default_timer as timer
import numpy as np
import copy
import os
import json


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        task_kind = getattr(self.params, 'task_kind', None)
        if task_kind is None:
            multiclass_datasets = [
                'FACED', 'SEED-V', 'SEED-IV', 'SEED', 'MI-KoreaU',
                'BCIC-IV-2a', 'BCIC', 'ISRUC', 'ISRUC-S3', 'ISRUC_S1', 'ISRUC_S3', 'BCIC2020-T3', 'TUEV'
            ]
            binary_datasets = ['MentalArithmetic', 'SHU-MI', 'CHB-MIT', 'TUAB']
            if self.params.downstream_dataset in multiclass_datasets:
                task_kind = 'multiclass'
            elif self.params.downstream_dataset in binary_datasets:
                task_kind = 'binary'

        if task_kind == 'multiclass':
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif task_kind == 'binary':
            self.criterion = BCEWithLogitsLoss().cuda()
        else:
            raise ValueError(
                f"Unsupported task kind for criterion: {task_kind!r} (dataset={self.params.downstream_dataset})"
            )

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:

                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                               weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    @staticmethod
    def _uses_sequence_multiclass(pred: torch.Tensor, y: torch.Tensor) -> bool:
        return pred.ndim == 3 and y.ndim == 2

    def _selection_mode(self, task_kind: str) -> str:
        mode = str(getattr(self.params, "select_best_by", "") or "").strip()
        if mode:
            return mode
        return "val_kappa" if task_kind == "multiclass" else "val_roc_auc"

    def _selection_metric_multiclass(self, acc: float, kappa: float, f1: float):
        mode = self._selection_mode("multiclass")
        if mode == "val":
            return float(acc), "val_bacc"
        if mode == "val_f1":
            return float(f1), "val_f1"
        return float(kappa), "val_kappa"

    def _selection_metric_binary(self, acc: float, pr_auc: float, roc_auc: float, kappa: float, f1: float):
        mode = self._selection_mode("binary")
        if mode == "val":
            return float(acc), "val_bacc"
        if mode == "val_kappa":
            return float(kappa), "val_kappa"
        if mode == "val_f1":
            return float(f1), "val_f1"
        if mode == "val_pr_auc":
            return float(pr_auc), "val_pr_auc"
        return float(roc_auc), "val_roc_auc"

    def _maybe_write_summary_json(self, summary: dict) -> None:
        if not getattr(self.params, 'write_summary_json', False):
            return
        model_dir = getattr(self.params, 'model_dir', '')
        if not model_dir:
            return
        os.makedirs(model_dir, exist_ok=True)
        summary_path = os.path.join(model_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def _training_config_summary(self) -> dict:
        return {
            "epochs": int(getattr(self.params, "epochs", 0)),
            "batch_size": int(getattr(self.params, "batch_size", 0)),
            "num_workers": int(getattr(self.params, "num_workers", 0)),
            "lr": float(getattr(self.params, "lr", 0.0)),
            "weight_decay": float(getattr(self.params, "weight_decay", 0.0)),
            "optimizer": str(getattr(self.params, "optimizer", "")),
            "label_smoothing": float(getattr(self.params, "label_smoothing", 0.0)),
            "classifier": str(getattr(self.params, "classifier", "")),
            "dropout": float(getattr(self.params, "dropout", 0.0)),
            "multi_lr": bool(getattr(self.params, "multi_lr", False)),
            "use_pretrained_weights": bool(getattr(self.params, "use_pretrained_weights", False)),
            "foundation_dir": str(getattr(self.params, "foundation_dir", "")),
        }

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        selection_best = float('-inf')
        selection_metric_name = self._selection_mode("multiclass")
        cm_best = None
        best_f1_epoch = 1
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self._uses_sequence_multiclass(pred, y):
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                selection_value, selection_metric_name = self._selection_metric_multiclass(acc, kappa, f1)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                with open(self.params.file_name, "a") as file:
                    file.write(
                        "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, "
                        "LR: {:.5f}, Time elapsed {:.2f} mins\n".format(
                            epoch + 1,
                            np.mean(losses),
                            acc,
                            kappa,
                            f1,
                            optim_state['param_groups'][0]['lr'],
                            (timer() - start_time) / 60
                        )
                    )
                    file.write(str(cm) + "\n")
                if selection_value > selection_best or self.best_model_states is None:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    with open(self.params.file_name, "a") as file:
                        file.write("kappa increasing....saving weights !! \n")
                        file.write("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    selection_best = selection_value
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    print("***************************Test************************")
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test************************\n")
                    acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
                    print("***************************Test results************************")
                    print(
                        "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                            acc,
                            kappa,
                            f1,
                        )
                    )
                    print(cm)
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test results************************\n")
                        file.write(
                            "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                                acc,
                                kappa,
                                f1,
                            )
                        )
                        file.write(str(cm) + "\n")
        if self.best_model_states is None:
            self.best_model_states = copy.deepcopy(self.model.state_dict())
            best_f1_epoch = self.params.epochs
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test************************\n")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test results************************\n")
                file.write(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                    acc,
                    kappa,
                    f1,
                )
            )
                file.write(str(cm) + "\n")
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            with open(self.params.file_name, "a") as file:
                file.write("model save in " + model_path)
            self._maybe_write_summary_json(
                {
                    "downstream_dataset": getattr(self.params, "downstream_dataset", ""),
                    "datasets_dir": getattr(self.params, "datasets_dir", ""),
                    "seed": int(getattr(self.params, "seed", 0)),
                    "task_kind": "multiclass",
                    "selection": {
                        "select_best_by": self._selection_mode("multiclass"),
                        "best_selection_value": float(selection_best),
                        "selection_metric_name": str(selection_metric_name),
                    },
                    "best_epoch": int(best_f1_epoch),
                    "val_best": {
                        "acc": float(acc_best),
                        "bacc": float(acc_best),
                        "kappa": float(kappa_best),
                        "f1": float(f1_best),
                    },
                    "test": {
                        "acc": float(acc),
                        "bacc": float(acc),
                        "kappa": float(kappa),
                        "f1": float(f1),
                    },
                    "training_config": self._training_config_summary(),
                    "checkpoint_path": model_path,
                    "log_path": getattr(self.params, "file_name", ""),
                }
            )

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = float('-inf')
        pr_auc_best = 0
        kappa_best = float('-inf')
        f1_best = float('-inf')
        selection_best = float('-inf')
        selection_metric_name = self._selection_mode("binary")
        cm_best = None
        best_f1_epoch = 1
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda().float()
                pred = self.model(x).view(-1)
                loss = self.criterion(pred, y.view(-1))

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, kappa, f1, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                selection_value, selection_metric_name = self._selection_metric_binary(acc, pr_auc, roc_auc, kappa, f1)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                with open(self.params.file_name, "a") as file:
                    file.write(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, "
                    "roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins \n".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                    file.write(str(cm) + "\n")
                if selection_value > selection_best or self.best_model_states is None:
                    print("auroc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    with open(self.params.file_name, "a") as file:
                        file.write("auroc increasing....saving weights !! \n")
                        file.write("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    kappa_best = kappa
                    f1_best = f1
                    selection_best = selection_value
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    print("***************************Test************************")
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test************************\n")
                    acc, pr_auc, roc_auc, kappa, f1, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
                    print("***************************Test results************************")
                    print(
                        "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                            acc,
                            pr_auc,
                            roc_auc,
                        )
                    )
                    print(cm)
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test results************************\n")
                        file.write(
                        "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f} \n".format(
                            acc,
                            pr_auc,
                            roc_auc,
                        )
                    )
                        file.write(str(cm) + "\n")
        if self.best_model_states is None:
            self.best_model_states = copy.deepcopy(self.model.state_dict())
            best_f1_epoch = self.params.epochs
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test************************\n")
            acc, pr_auc, roc_auc, kappa, f1, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test results************************\n")
                file.write(
                    "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f} \n".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    )
                )
                file.write(str(cm) + "\n")
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            with open(self.params.file_name, "a") as file:
                file.write("model save in " + model_path)
            self._maybe_write_summary_json(
                {
                    "downstream_dataset": getattr(self.params, "downstream_dataset", ""),
                    "datasets_dir": getattr(self.params, "datasets_dir", ""),
                    "seed": int(getattr(self.params, "seed", 0)),
                    "task_kind": "binary",
                    "selection": {
                        "select_best_by": self._selection_mode("binary"),
                        "best_selection_value": float(selection_best),
                        "selection_metric_name": str(selection_metric_name),
                    },
                    "best_epoch": int(best_f1_epoch),
                    "val_best": {
                        "acc": float(acc_best),
                        "bacc": float(acc_best),
                        "pr_auc": float(pr_auc_best),
                        "roc_auc": float(roc_auc_best),
                        "kappa": float(kappa_best),
                        "f1": float(f1_best),
                    },
                    "test": {
                        "acc": float(acc),
                        "bacc": float(acc),
                        "pr_auc": float(pr_auc),
                        "roc_auc": float(roc_auc),
                        "kappa": float(kappa),
                        "f1": float(f1),
                    },
                    "training_config": self._training_config_summary(),
                    "checkpoint_path": model_path,
                    "log_path": getattr(self.params, "file_name", ""),
                }
            )
