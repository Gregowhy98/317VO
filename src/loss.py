import torch

import torch.nn as nn
import torch.nn.functional as F

class LineSegmentLoss(nn.Module):
    def __init__(self):
        super(LineSegmentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_lines, gt_lines, pred_desc=None, gt_desc=None):
        # Line detection loss (using MSE)
        line_loss = self.mse_loss(pred_lines, gt_lines)
        
        # Line description loss (using BCE)
        if pred_desc is not None and gt_desc is not None:
            desc_loss = self.bce_loss(pred_desc, gt_desc)
        else:
            desc_loss = 0
        
        # Total loss
        total_loss = line_loss + desc_loss
        
        return total_loss

# Example usage:
if __name__ == "__main__":
    pred_lines = torch.rand(1, 10, 2)
    gt_lines = torch.rand(1, 10, 2)
    pred_desc = torch.rand(1, 10, 128)
    gt_desc = torch.rand(1, 10, 128)
    
    loss_fn = LineSegmentLoss()
    loss = loss_fn(pred_lines, gt_lines, pred_desc, gt_desc)
    loss_1 = loss_fn(pred_lines, gt_lines)
    
    print(loss)
    print(loss_1)
    pass