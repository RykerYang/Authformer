import torch

def get_lr(optimizer):
    """Get the learning rate of the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy(output, target):
    """Calculate accuracy."""
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)