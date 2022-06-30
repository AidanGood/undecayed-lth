# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np
import torch.nn as nn
import torch
from datasets.base import DataLoader
import datasets.registry
from collections import OrderedDict, defaultdict

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Undecayed Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None, dataset=None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        
        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}
                
        # Get the (estimated) gradients #
        model = trained_model
        
        loss_func = nn.CrossEntropyLoss()
        dataset_hparams = dataset
        train_loader = datasets.registry.get(dataset_hparams, train=True)
        
        inputs, outputs = next(iter(train_loader))
        
        training = model.training
        model.train()
        pred = model(inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        loss = loss_func(pred.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), outputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        loss.backward()

        gradients = {}
        
        for name, param in model.named_parameters():
            gradients[name] = param.grad
                    
        # Get the undecayed scoring for each weight. 
        undecayed = {k: (-(v)*(v.grad)).clone().cpu().detach().numpy()+ (5e-4)*(pow((v.clone().cpu().detach().numpy()),2))
                   for k, v in model.named_parameters()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in undecayed.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        # Create mask
        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in undecayed.items()})
        
        
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
