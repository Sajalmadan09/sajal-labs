import torch
import torch.nn as nn

IN_DIM, HIDDEN_DIM, OUT_DIM = 64, 128, 10


class TinyMLP(nn.Module):
    """Linear -> ReLU -> Linear -> Softmax, per the brief's smallest teaching model.

    Shape is a constructor arg (not a module-level constant) so exp9's real
    classifier can reuse this instead of redefining it — same pattern as
    tiny_transformer.py's TinyTransformerBlock.
    """

    def __init__(self, in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
