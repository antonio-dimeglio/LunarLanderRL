import gymnasium as gym
import torch as th
import numpy as np
from Models import *
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import termcolor
from enum import Enum

ROUNDING_PRECISION = 2

Trace = namedtuple("Trace", ["rewards", "values", "log_probs"])
Trace.__doc__ = "A named tuple for the traces of the agent."

class AgentType(Enum):
    """
        The AgentType class is an enumeration class for the type of the agent model.

        Attributes:
            ACBaseline (int): The Actor-Critic model with Baseline Subtraction.
            ACBoostrapping (int): The Actor-Critic model with Boostrapping.
            ACBoostrappingBaseline (int): The Actor-Critic model with both Boostrapping and Baseline Subtraction.
            REINFORCE (int): The REINFORCE model.
    """
    ACBaseline = 0
    ACBoostrapping = 1
    ACBoostrappingBaseline = 2
    REINFORCE = 3

class Agent():
    """
        The Agent class for a gym environment, assuming that the environment is a discrete action space.
        The Agent class is responsible for training the agent using the Actor-Critic or REINFORCE algorithm.
        The Actor-Critic algorithm can be used either by only using Boostrapping, Baseline Subtraction or both.

        Args:
            alpha (float): The learning rate for the optimizer.
            beta (float): The entropy coefficient for the loss function.
            gamma (float): The discount factor for the rewards.
            model (AgentType): The type of the agent model.
            n_steps (int): The number of steps to boostrap the rewards.
            environment (str): The name of the gym environment.
    """

    def __init__(self,
                    alpha: float = 1e-4,
                    beta: float = 0.01,
                    gamma: float = 0.99,
                    model: AgentType = AgentType.ACBoostrappingBaseline,
                    environment: str = "LunarLander-v2",
                    n_steps: int = 5):
    
        self.env = gym.make(environment, continuous = False)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.action_space = self.env.action_space.n

        match model:
            case AgentType.ACBaseline | AgentType.ACBoostrapping | AgentType.ACBoostrappingBaseline:
                self.actor = Actor(self.env.observation_space.shape[0], self.action_space).to(self.device)
                self.critic = Critic(self.env.observation_space.shape[0]).to(self.device)
            case AgentType.REINFORCE:
                self.actor = Actor(self.env.observation_space.shape[0], self.action_space).to(self.device)
            case _:
                raise ValueError(f"Invalid model type: {model}.")
        
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = None if model == AgentType.REINFORCE else th.optim.Adam(self.critic.parameters(), lr=alpha)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model = model
        self.n_steps = n_steps


    def __get_trace(self) -> tuple[Trace, float]:
        """
            The private method to get the trace of the agent.

            Returns:
                Trace: The trace of the agent.
                float: The entropy of the trace.
        """

        rewards = []
        values = []
        log_probs = []
        entropy = 0.0

        state, _ = self.env.reset()
        done = False

        while not done:
            state = th.from_numpy(state).to(self.device)

            probs = self.actor(state)
            value = self.critic(state).item() if self.model != AgentType.REINFORCE else 0.0

            action = np.random.choice(self.action_space, p=probs.cpu().detach().numpy())
            log_prob = th.log(probs[action])
            log_probs.append(log_prob.item())
            
            entropy += -(probs * th.log(probs)).sum().item()
            
            state, reward, terminated, truncated, _ = self.env.step(action)

            rewards.append(reward)
            values.append(value)

            done = terminated or truncated
        
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        log_probs = np.array(log_probs, dtype=np.float32)

        return Trace(rewards, values, log_probs), entropy
        
    def __get_discounted_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
            The private method to get the discounted rewards.

            Args:
                rewards (np.ndarray): The rewards of the agent.

            Returns:
                np.ndarray: The discounted rewards.
        """

        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0.0

        for i in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[i]
            discounted_rewards[i] = running_add

        return discounted_rewards

    def __get_cumulative_returns(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
            The private method to get the cumulative returns.

            Args:
                rewards (np.ndarray): The rewards of the agent.
                values (np.ndarray): The values of the agent.

            Returns:
                np.ndarray: The cumulative returns.
        """

        cumulative_returns = np.zeros_like(rewards, dtype=np.float32)

        for t in range(len(cumulative_returns)):
            n = min(self.n_steps, len(cumulative_returns) - t - 1)
            for k in range(n):
                cumulative_returns[t] += rewards[t+k] + values[t+n]

        return cumulative_returns
    
    def __get_actor_loss(self, returns: np.ndarray, log_probs: np.ndarray, values: np.ndarray, entropy: float) -> th.Tensor:
        """
            The private method to get the actor loss.

            Args:
                returns (np.ndarray): The returns of the agent.
                log_probs (np.ndarray): The log probabilities of the agent.
                values (np.ndarray): The values of the agent.
                entropy (float): The entropy of the agent.

            Returns:
                th.Tensor: The actor loss.
        """

        actor_loss = 0.0

        
        for i in range(len(returns)):
            if self.model == AgentType.REINFORCE:
                actor_loss += -log_probs[i] * returns[i]
            else:
                advantage = returns[i] - values[i]
                actor_loss += -log_probs[i] * advantage

        actor_loss = actor_loss - self.beta * entropy

        return th.tensor(actor_loss, dtype=th.float32, device=self.device, requires_grad=True)
    
    def __get_critic_loss(self, returns: np.ndarray, values: np.ndarray) -> th.Tensor:
        """
            The private method to get the critic loss.

            Args:
                returns (np.ndarray): The returns of the agent.
                values (np.ndarray): The values of the agent.

            Returns:
                th.Tensor: The critic loss.
        """

        critic_loss = th.tensor(np.sum((returns - values) ** 2), dtype=th.float32, device=self.device, requires_grad=True)

        return critic_loss
    
    def __print_training_trajectory(self, i: int, m:int, actor_loss: th.Tensor, critic_loss: th.Tensor, rewards: np.ndarray, entropy:float) -> None:
        """
            The private method to print the training trajectory.

            Args:
                i (int): The current episode.
                actor_loss (th.Tensor): The actor loss.
                critic_loss (th.Tensor): The critic loss.
                rewards (np.ndarray): The rewards of the agent.
                entropy (float): The entropy of the agent.
        """

        print(f"Episode {i}/{m}", end="\t")
        if self.model != AgentType.REINFORCE:
            print(f"Actor Loss: {actor_loss:.{ROUNDING_PRECISION}f}\tCritic Loss: {critic_loss:.{ROUNDING_PRECISION}f}", end="\t")
        else:
            print(f"Model Loss: {actor_loss:.{ROUNDING_PRECISION}f}", end="\t")

        print("Current average reward: ", end="")
        avg_reward = np.round(np.mean(rewards), ROUNDING_PRECISION)
        color = "green" if avg_reward > 0 else "yellow" if avg_reward > -100.0 else "red" 
        print(termcolor.colored(avg_reward, color), end="\t")
        print(f"Entropy: {entropy:.{ROUNDING_PRECISION}f}")

    def train(self, m:int, quiet=False) -> np.ndarray:
        """
            Train the agent for m episodes.

            Args:
                m (int): The number of episodes to train the agent.
                quiet (bool): Whether to print training trajectory.

            Returns:
                np.ndarray: The rewards of the agent for each episode.
        """

        rewards_per_episode = np.zeros(m, dtype=np.float32)


        for i in range(m):
            trace, entropy = self.__get_trace()

            if self.model == AgentType.ACBoostrapping or self.model == AgentType.ACBoostrappingBaseline:
                returns = self.__get_cumulative_returns(trace.rewards, trace.values)
            else:
                returns = self.__get_discounted_rewards(trace.rewards)

            actor_loss = self.__get_actor_loss(returns, trace.log_probs, trace.values, entropy)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.model != AgentType.REINFORCE:
                critic_loss = self.__get_critic_loss(returns, trace.values)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            rewards_per_episode[i] = np.sum(trace.rewards)

            if not quiet and i % 10 == 0 and i != 0:
                self.__print_training_trajectory(i, 
                        m, 
                        actor_loss, 
                        critic_loss if self.model != AgentType.REINFORCE else None, 
                        rewards_per_episode, 
                        entropy)

        return rewards_per_episode
    

if __name__ == "__main__":
    agent = Agent(model=AgentType.ACBoostrappingBaseline)
    rewards = agent.train(5000)

    smoothed_rewards = np.convolve(rewards, np.ones((100,))/100, mode="valid")

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("rewards.png")