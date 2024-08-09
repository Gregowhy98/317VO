import torch
import torch.nn as nn
import torch.nn.functional as F

class GreVONet(nn.Module):
    defult_config = {
        "num_classes": 1,
        "num_stacks": 2,
        "num_blocks": 1,
        "num_features": 256,
        "num_hourglass_features": 256
    }
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # Forward pass
        return self.hourglass(x)