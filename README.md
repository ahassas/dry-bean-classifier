# Dry Bean Classifier

A small feedforward neural network in PyTorch that classifies dry beans into 7 types (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) from 16 shape features such as area, perimeter, axis lengths and roundness.

Final result: about **0.94 macro F1**.

## Data

[Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) from UCI. The data is not included in this repo. Split it into `data/train.csv` and `data/validation.csv` (columns: the 16 features + `Class`).

Two things I noticed in the data:

- The feature ranges are very different. Area goes up to ~250,000 while ShapeFactor2 is around 0.001, so standardization is necessary.
- The classes are imbalanced (DERMASON is about 7x bigger than BOMBAY), so I tracked macro F1 instead of only accuracy.

## Model

- Fully-connected network: 16 → 64 → 32 → 7, ReLU
- CrossEntropyLoss, AdamW (lr 1e-2, weight decay 1e-4), cosine learning rate schedule, batch size 32
- The standardization mean/std are stored **inside the model** as buffers. This way `model.pt` is self-contained, and inference does not need a separate scaler file.
- No Dropout or BatchNorm, so the model behaves the same in train and eval mode. Regularization comes from weight decay and keeping the best validation epoch.

## Experiments

I picked a base config and changed one setting at a time (100 epochs, best validation epoch kept):

| Run | Change | Best val macro F1 |
|---|---|---|
| base | – | 0.9396 |
| arch-16 | one hidden layer of 16 | 0.9394 |
| arch-128x64 | wider | 0.9401 |
| arch-64x64x32 | deeper | 0.9405 |
| lr-1e-3 | lower lr | 0.9395 |
| lr-1e-4 | much lower lr | 0.9355 |
| no-scheduler | constant lr | 0.9425 |
| no-weight-decay | wd = 0 | 0.9389 |
| batch-128 | larger batch | 0.9413 |
| log-size-features | log on size features | 0.9406 |

What I learned from these:

- Standardization gives most of the improvement. After that, the other settings change the result very little.
- A bigger network does not help on this data.
- Running the same config with different seeds changes the score by about ±0.002, so most differences in the table are just noise.
- With lr 1e-2 and 64+ units, the validation loss goes up after the first part of training but the F1 stays flat. The model gets more confident on the samples it gets wrong, not more wrong overall.

## Training on train + validation

For the final model I also used the validation data for training. Then there is nothing left to pick the best epoch, so I tested how many epochs to use. I held out 20% of the training set and compared (3 seeds each):

| Setup | Held-out macro F1 |
|---|---|
| train only, best epoch | 0.9393 ± 0.0032 |
| train + val, 30 epochs | 0.9421 ± 0.0006 |
| train + val, 50 epochs | 0.9422 ± 0.0010 |
| train + val, 100 epochs | 0.9346 ± 0.0038 |

More data helps a bit and makes results more stable, but 100 epochs overfits. The final model uses 30 epochs.

## Errors

Most mistakes are between SIRA and DERMASON, then between BARBUNYA and CALI. These pairs have very similar shapes. BOMBAY is always correct because those beans are much larger than the others.

## Run

```bash
pip install torch pandas scikit-learn wandb
python main.py                 # train the final model -> saved_objects/model.pt
python main.py --experiments   # run all experiments (logged to Weights & Biases)
```

Load the trained model:

```python
from model import Model
model = Model(input_dim=16, output_dim=7, pretrained_path="saved_objects/model.pt")
```
