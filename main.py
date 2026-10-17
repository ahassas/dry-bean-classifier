# Usage:
#   python main.py                -> trains the final model on train+validation, saves saved_objects/model.pt
#   python main.py --experiments  -> runs all experiments below (logged to Weights & Biases)
#
from data_utils import Preprocessor, Dataset
from trainer_utils import Trainer
from model import Model

from torch.utils.data import ConcatDataset, DataLoader
import argparse
import random
import numpy as np
import torch

features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
            'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
            'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
            'ShapeFactor2', 'ShapeFactor3','ShapeFactor4']

PROJECT = "dry-bean-classifier"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Final config. hidden_dims must match the default in model.py,
# because inference code builds Model() without extra arguments.
FINAL = dict(name="final", hidden_dims=[64, 32], lr=1e-2, weight_decay=1e-4,
             batch_size=32, epochs=100, scheduler="cosine", log_size=False, seed=42)

# Each experiment changes ONE thing compared to FINAL
EXPERIMENTS = [
    FINAL,
    # architecture
    dict(FINAL, name="arch-16",          hidden_dims=[16]),
    dict(FINAL, name="arch-128x64",      hidden_dims=[128, 64]),
    dict(FINAL, name="arch-64x64x32",    hidden_dims=[64, 64, 32]),
    # learning rate
    dict(FINAL, name="lr-1e-3",          lr=1e-3),
    dict(FINAL, name="lr-1e-4",          lr=1e-4),
    # optimization / regularization
    dict(FINAL, name="no-scheduler",     scheduler=None),
    dict(FINAL, name="no-weight-decay",  weight_decay=0.0),
    dict(FINAL, name="batch-128",        batch_size=128),
    # preprocessing
    dict(FINAL, name="log-size-features", log_size=True),
]

# Final model: same settings, but trained on train + validation together.
# There is no held-out data left to pick the best epoch, so we use a fixed number
# of epochs (30). Chosen with a holdout simulation, see README.
RELEASE = dict(FINAL, name="release-train+val-30ep", epochs=30, use_val_for_training=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run(cfg):
    set_seed(cfg["seed"])
    pre = Preprocessor(feature_columns=features, label_column="Class", log_size=cfg["log_size"])
    train_dl = Dataset("./data/train.csv", pre, batch_size=cfg["batch_size"], shuffle=True).prepare_data()
    valid_dl = Dataset("./data/validation.csv", pre, batch_size=32, shuffle=False).prepare_data()

    full = cfg.get("use_val_for_training", False)
    X_fit = train_dl.dataset.tensors[0]
    if full:
        X_fit = torch.cat([X_fit, valid_dl.dataset.tensors[0]])
        both = ConcatDataset([train_dl.dataset, valid_dl.dataset])
        train_dl = DataLoader(both, batch_size=cfg["batch_size"], shuffle=True)

    model = Model(input_dim=16, output_dim=7, device=DEVICE, hidden_dims=cfg["hidden_dims"])
    model.fit_scaler(X_fit.to(DEVICE))   # mean/std only from the data we train on

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] == "cosine" else None)

    trainer = Trainer(model, optimizer, loss_fn, PROJECT, device=DEVICE,
                      scheduler=scheduler, config=cfg, run_name=cfg["name"],
                      keep_best=not full)   # with val in training, keep the last epoch
    best_f1 = trainer.train(train_dl, valid_dl, epochs=cfg["epochs"])
    print(f"{cfg['name']:<20} best val macro-F1 = {best_f1:.4f}")
    return model, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", action="store_true")
    args = parser.parse_args()

    if args.experiments:
        for cfg in EXPERIMENTS:
            run(cfg)
    else:
        model, _ = run(RELEASE)
        model.save("./saved_objects/model.pt")
        print("saved -> saved_objects/model.pt")
