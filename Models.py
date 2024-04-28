import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        super(Policy, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layer(x)
    
    
    
    
class Critic(nn.Module):
    def __init__(self, input_size:int) -> None:
        super(Critic, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layer(x)