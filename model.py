from typing import List, Dict

import numpy as np
import torch

class Model(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int, pretrained_path: str = None, device: str = "cpu", **kwargs):
        super(Model, self).__init__()

        self.pretrained_path = pretrained_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Defaults = final architecture, so Model(16, 7) loads model.pt directly
        self.hidden_dims = [64, 32]
        self.__dict__.update(kwargs)

        # Initialize self.model object
        self.init_model()

        # Load model if pretrained model is available
        if pretrained_path != None:
            self.load_state_dict(torch.load(self.pretrained_path, weights_only=True))
            self.eval()   # loaded model is ready for inference

        self.to(self.device)

    def save(self, path: str):
        # store on CPU so the file loads on any machine (cuda / mps / cpu)
        torch.save({k: v.cpu() for k, v in self.state_dict().items()}, path)

    def init_model(self):
        # Feature standardization lives inside the model as buffers.
        # They are saved in model.pt together with the weights.
        self.register_buffer("mean", torch.zeros(self.input_dim))
        self.register_buffer("std", torch.ones(self.input_dim))

        layers, d = [], self.input_dim
        for h in self.hidden_dims:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, self.output_dim))
        self.net = torch.nn.Sequential(*layers)

    def fit_scaler(self, X: torch.Tensor):
        # call once with the full TRAINING features before training
        self.mean.copy_(X.mean(0))
        self.std.copy_(X.std(0) + 1e-8)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)   # raw logits (argmax works, CrossEntropyLoss expects logits)
