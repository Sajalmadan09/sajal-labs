import torch
import torch.nn as nn

IN_DIM, HIDDEN_DIM, OUT_DIM = 64, 128, 10


class TinyMLP(nn.Module):
    """Linear -> ReLU -> Linear -> Softmax, per the brief's smallest teaching model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, OUT_DIM)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
