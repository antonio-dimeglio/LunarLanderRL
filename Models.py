import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int) -> None:
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)
    
class Critic(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int) -> None:
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)