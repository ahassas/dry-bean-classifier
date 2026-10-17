from torch.utils.data import DataLoader
import torch

from typing import Dict
from sklearn.metrics import f1_score
import copy
import wandb

class Trainer:

    def __init__(self, model, optimizer, loss_fn, project_name, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.project_name = project_name
        self.device = device

        self.scheduler = None   # optional, e.g. CosineAnnealingLR
        self.keep_best = True   # False -> keep the last epoch (used when val data is in training)
        self.config = {}        # hyperparameters, sent to W&B
        self.run_name = None
        self.__dict__.update(kwargs)
        self.model = self.model.to(self.device)

    def train(self, training_data: DataLoader, validation_data: DataLoader, epochs: int):
        best_f1, best_state = -1.0, None

        with wandb.init(project=self.project_name, name=self.run_name, config=self.config) as run:
            for epoch in range(epochs):
                train_result = self.train_loop(training_data)
                eval_result  = self.eval_loop(validation_data)
                if self.scheduler is not None:
                    self.scheduler.step()

                # keep the weights of the best validation epoch (early-stopping style)
                if eval_result["f1"] > best_f1:
                    best_f1 = eval_result["f1"]
                    best_state = copy.deepcopy(self.model.state_dict())

                self.log_wandb(run, epoch, train_result, eval_result)

            run.summary["best_val_f1"] = best_f1

        if self.keep_best:
            self.model.load_state_dict(best_state)
        return best_f1

    def _run_epoch(self, data: DataLoader, train: bool) -> Dict:
        self.model.train(train)
        total_loss, preds, labels = 0.0, [], []

        with torch.set_grad_enabled(train):
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                loss = self.loss_fn(out, y.argmax(1))

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(X)
                preds.append(out.argmax(1).cpu())
                labels.append(y.argmax(1).cpu())

        preds, labels = torch.cat(preds), torch.cat(labels)
        return {
            "loss": total_loss / len(labels),
            "acc": (preds == labels).float().mean().item(),
            "f1": f1_score(labels, preds, average="macro"),
        }

    def train_loop(self, data: DataLoader) -> Dict:
        return self._run_epoch(data, train=True)

    def eval_loop(self, data: DataLoader) -> Dict:
        return self._run_epoch(data, train=False)

    def log_wandb(self, run, epoch, training_result, eval_result):
        log = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
        log.update({f"train/{k}": v for k, v in training_result.items()})
        log.update({f"val/{k}": v for k, v in eval_result.items()})
        run.log(log)
