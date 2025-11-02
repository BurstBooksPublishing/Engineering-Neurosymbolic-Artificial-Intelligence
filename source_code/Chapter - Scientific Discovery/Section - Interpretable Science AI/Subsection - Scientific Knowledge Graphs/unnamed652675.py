import torch
import torch.nn as nn

class ReactionPredictor(nn.Module):
    def __init__(self):
        super(ReactionPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Assuming one input feature for simplicity
        self.fc2 = nn.Linear(10, 2)  # Output two features, one for each gas

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ReactionPredictor()
condition = torch.tensor([1.0])  # Example condition
prediction = model(condition)
print(prediction)