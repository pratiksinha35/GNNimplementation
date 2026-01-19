import torch
import torch.nn as nn
import torch.optim as optim

class ResBlock(nn.Module):
    """Residual block for MLP"""
    def __init__(self, hidden_dim, block_dim, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, block_dim)
        self.fc2 = nn.Linear(block_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    """ResNet model"""
    def __init__(self, input_dim, output_dim, num_blocks, hidden_dim, block_dim, dropout_rate=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Create residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, block_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.activation(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x)
