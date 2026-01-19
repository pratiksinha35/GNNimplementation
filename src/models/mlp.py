import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
from copy import deepcopy

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, layer_size=5):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, layer_size))
        layers.append(nn.ReLU())
        
        # Hidden layers (n_layers - 1 additional hidden layers)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(layer_size, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_mlp_pytorch(input_dim, output_dim, n_layers=2, layer_size=5, 
                       lr=0.01, l2_reg=0.01):
    """
    Create MLP with specified parameters
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        n_layers: Number of hidden layers
        layer_size: Number of units per hidden layer
        lr: Learning rate
        l2_reg: L2 regularization strength (weight_decay)
    """
    model = MLP(input_dim, output_dim, n_layers, layer_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    return model, optimizer


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: If True, prints early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} -> {val_loss:.6f}). Saving model...')
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model state
            self.best_model_state = deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
            return self.early_stop



