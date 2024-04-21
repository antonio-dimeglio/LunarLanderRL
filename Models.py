import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(ActorCritic, self).__init__()
        
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.actor_layer = nn.Linear(hidden_size, output_size)
        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared_layer(x)
        
        value = self.critic_layer(x).item()

        action_probs = F.softmax(self.actor_layer(x), dim=-1)

        action = Categorical(action_probs).sample().item()
        log_prob = th.log(action_probs[action])

        return action, log_prob, value