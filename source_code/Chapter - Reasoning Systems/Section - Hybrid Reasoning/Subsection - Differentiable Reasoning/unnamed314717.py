import torch
import torch.nn.functional as F

def red_round_rule(color, shape):
    # Assume color and shape are tensors representing the probability of each attribute
    red_score = color[:, 0]  # index for red
    round_score = shape[:, 1]  # index for round
    return F.sigmoid(red_score + round_score - 1)