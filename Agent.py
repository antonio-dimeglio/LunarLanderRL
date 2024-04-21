import gymnasium as gym
import torch as th
import numpy as np
from Models import *
import torch.nn.functional as F
from collections import namedtuple, deque 
import termcolor as tc 


Trace = namedtuple("TraceEntry", ["values", "log_probs", "rewards", "entropies"])

ROUNDING_PRECISION = 2

class Agent():
    """
        The Agent class for the LunarLander environment.

        Args:
            alpha (float): The learning rate of the agent, default is 1e-4.
            beta (float): The entropy regularization term, default is 1e-2.
            gamma (float): The discount factor, default is 0.999.
            model_type (str): The type of model to use for the agent, which can be 
                "ACBoot", "ACBaseline", "ACBootbaseline" or "REINFORCE", default is "ACBoot".
            n (int): The number of steps to look ahead for the critic, default is 10.
            convergence_threshold (float): The threshold for the convergence of the agent, default is 1e-3.
    """

    def __init__(self, 
                alpha: float = 1e-4, 
                beta: float = 1e-2, 
                gamma: float = 0.99, 
                model_type: str = "ACBoot",
                n: int = 10,
                convergence_threshold: float = 1e-3) -> None:
        self.env = gym.make("LunarLander-v2", continuous=False)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model_type = model_type
        self.n = n
        self.convergence_threshold = convergence_threshold

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        match model_type:
            case "ACBoot":
                self.actor = Actor(self.env.observation_space.shape[0], 128, self.env.action_space.n).to(self.device)
                self.critic = Critic(self.env.observation_space.shape[0], 128, 1).to(self.device)
                self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.alpha)
                self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=self.alpha)
                pass 
            case "ACBaseline":
                raise ValueError("Unimplemented model type.")
            case "ACBootbaseline":
                raise ValueError("Unimplemented model type.")
            case "REINFORCE":
                raise ValueError("Unimplemented model type.")
            case _:
                raise ValueError("Invalid model type. Please choose from 'ACBoot', 'ACBaseline', 'ACBootbaseline' or 'REINFORCE'.")
            
    def __ac_training(self, m: int, quiet: bool = False) -> None:
        """
            Actor-Critic training without bootstrapping or baseline subtraction.
        """

        state, _ = self.env.reset()
        state = th.tensor(state, dtype=th.float32).to(self.device)
        done = False 

        while not done:
            probs = F.softmax(self.actor(state), dim=-1)
            curr_state_value = self.critic(state)
            dist = th.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            next_state = th.tensor(next_state, dtype=th.float32).to(self.device)
            reward = th.tensor(reward, dtype=th.float32).to(self.device)
            done = terminated or truncated

            next_state_value = self.critic(next_state)

            advantage = (reward + self.gamma * next_state_value - curr_state_value)

            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage ** 2

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            state = next_state

            
    def train(self, m: int, quiet: bool = False) -> None:
        """
            Entry point for training the agent.

            Args:
                m (int): The number of episodes to train the agent.
                quiet (bool): Whether to suppress the output of the training process.
        """
        match self.model_type:
            case "ACBoot":
                self.__ac_training(m, quiet) 
            case "ACBaseline":
                raise ValueError("Unimplemented model type.")
            case "ACBootbaseline":
                raise ValueError("Unimplemented model type.")
            case "REINFORCE":
                raise ValueError("Unimplemented model type.")
            case _:
                raise ValueError("Invalid model type. Please choose from 'ACBoot', 'ACBaseline', 'ACBootbaseline' or 'REINFORCE'.")

    def visualize_agent(self) -> None:
        """
            Visualize the agent in the environment.
        """
        pass
    



if __name__ == "__main__":
    agent = Agent()
    agent.train(1000)
    