import gymnasium as gym
import torch as th
import numpy as np
from Models import *
from collections import namedtuple
import matplotlib.pyplot as plt
import termcolor
from enum import Enum

ROUNDING_PRECISION = 2

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
            alpha_policy (float): The learning rate for the optimizer for the policy model.
            alpha_critic (float): The learning rate for the optimizer for the critic model.
            beta (float): The entropy coefficient for the loss function.
            gamma (float): The discount factor for the rewards.
            agent_type (AgentType): The type of the agent model.
            n_steps (int): The number of steps to boostrap the rewards.
            environment (str): The name of the gym environment.
    """

    def __init__(self,
        alpha_policy:float = 0.001,
        alpha_critic:float = 0.01,
        beta:float = 0.01,
        gamma:float = 0.99,
        agent_type:AgentType = AgentType.REINFORCE,
        n_steps:int = 5,
        environment:str = 'LunarLander-v2'
        ) -> None:
        
        self.alpha_policy = alpha_policy
        self.alpha_critic = alpha_critic
        self.beta = beta
        self.gamma = gamma
        self.agent_type = agent_type
        self.n_steps = n_steps
        self.env = gym.make(environment, continuous=False)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.state_space_size = self.env.observation_space.shape[0]
        self.action_space_size = self.env.action_space.n

        self.model = Policy(input_size=self.state_space_size, output_size=self.action_space_size).to(self.device)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.alpha_policy)

        if self.agent_type != AgentType.REINFORCE:
            self.critic = Critic(input_size=self.state_space_size).to(self.device)
            self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=self.alpha_critic)


    def save_model(self, model:th.nn.Module, path:str) -> None:
        """
            The save_model function is responsible for saving the model to a file.

            Args:
                model (th.nn.Module): The model to be saved.
                path (str): The path to save the model.
        """
        th.save(model.state_dict(), path)

    def load_model(self, model:th.nn.Module, path:str) -> th.nn.Module:
        """
            The load_model function is responsible for loading the model from a file.

            Args:
                model (th.nn.Module): The model to be loaded.
                path (str): The path to load the model.
        """
        model.load_state_dict(th.load(path))
        return model
    
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

        color = lambda x: termcolor.colored(x, "green") if x > 0 else termcolor.colored(x, "yellow") if x > -100.0 else termcolor.colored(x, "red")
        avg_reward = np.round(np.mean(rewards), ROUNDING_PRECISION)

        print(f"Episode {i}/{m}", end="\t")
        if self.agent_type != AgentType.REINFORCE:
            print(f"Actor Loss: {actor_loss:.{ROUNDING_PRECISION}f}\tCritic Loss: {critic_loss:.{ROUNDING_PRECISION}f}", end="\t")
        else:
            print(f"Model Loss: {actor_loss:.{ROUNDING_PRECISION}f}", end="\t")

        print(f"Current average reward: {color(avg_reward)}", end="\t")
        print(f"Last episode reward: {color(np.round(rewards[-1], ROUNDING_PRECISION))}", end="\t")
        print(f"Entropy: {entropy:.{ROUNDING_PRECISION}f}")

    def __get_trace(self) -> tuple[th.Tensor, th.Tensor, th.Tensor, float]:
        """
            The private method to get the trace of the agent.

            Returns:
                tuple[th.Tensor, th.Tensor, th.Tensor, float]: The tuple of the episode rewards, log probabilities, values and entropy.
        """
        state, _ = self.env.reset()
        done = False 
        entropy = 0.0
        episode_rewards = []
        episode_log_probs = []
        episode_values = []

        while not done:
            probs = self.model(th.tensor(state).float().to(self.device))
            if self.agent_type != AgentType.REINFORCE:
                value = self.critic(th.tensor(state).float().to(self.device))   
            else:
                value = th.tensor(0.0)

            episode_values.append(value)

            action = np.random.choice(self.action_space_size, p=probs.detach().cpu().numpy())
            

            log_prob = th.log(probs[action])
            episode_log_probs.append(log_prob)

            entropy += -(probs * th.log(probs)).sum()

            state, reward, terminated, truncated, _ = self.env.step(action)

            episode_rewards.append(reward)
            done = terminated or truncated

        episode_rewards = th.tensor(episode_rewards).to(self.device)
        episode_log_probs = th.stack(episode_log_probs)
        episode_values = th.stack(episode_values)

        return episode_rewards, episode_log_probs, episode_values, entropy
    
    def __get_discounted_rewards(self, rewards:th.Tensor) -> th.Tensor:
        """
            The private method to get the discounted rewards.

            Args:
                rewards (th.Tensor): The rewards of the agent.
            
            Returns:
                th.Tensor: The discounted rewards.
        """
        discounted_rewards = th.zeros(len(rewards)).to(self.device)
        R = 0

        for i, reward in enumerate(reversed(rewards)):
            R = reward + self.gamma * R
            discounted_rewards[i] = R

        discounted_rewards = discounted_rewards.flip(dims=(0,))

        return discounted_rewards
    
    def __get_policy_loss(self, log_probs:th.Tensor, 
                          discounted_rewards:th.Tensor, 
                          values:th.Tensor, 
                          entropy:float) -> th.Tensor:
        """
            The private method to get the policy loss.

            Args:
                log_probs (th.Tensor): The log probabilities of the agent.
                discounted_rewards (th.Tensor): The discounted rewards of the agent.
                values (th.Tensor): The values of the agent.
                entropy (float): The entropy of the agent.

            Returns:
                th.Tensor: The policy loss.
        """
        policy_loss = th.tensor(0.0, requires_grad=True).to(self.device)

        for log_prob, reward, value in zip(log_probs, discounted_rewards, values):
            if self.agent_type != AgentType.REINFORCE:
                advantage = reward - value
            else:
                advantage = reward
            
            policy_loss += -log_prob * advantage

        policy_loss -= self.beta * entropy

        return policy_loss
    
    def __get_critic_loss(self, values:th.Tensor, returns:th.Tensor) -> th.Tensor:
        """
            The private method to get the critic loss.

            Args:
                values (th.Tensor): The values of the agent.
                returns (th.Tensor): The returns of the agent.

            Returns:
                th.Tensor: The critic loss.
        """
        critic_loss = th.tensor(0.0, requires_grad=True).to(self.device)

        for value, ret in zip(values, returns):
            critic_loss += (value - ret) ** 2

        return critic_loss 
    
    def train(self, n_episodes) -> list[float]:
        rewards: list[float] = []

        for m in range(n_episodes):
            episode_rewards, episode_log_probs, episode_values, entropy = self.__get_trace()
            discounted_rewards = self.__get_discounted_rewards(episode_rewards)

            policy_loss = self.__get_policy_loss(episode_log_probs, discounted_rewards, episode_values, entropy)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            rewards.append(sum(episode_rewards).item())

            self.__print_training_trajectory(m+1, n_episodes, policy_loss, th.tensor(0.0), rewards, entropy)

        return rewards 

if __name__ == '__main__':
    agent = Agent()
    rewards = agent.train(1000)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE')
    plt.savefig('reinforce.png')
    plt.show()