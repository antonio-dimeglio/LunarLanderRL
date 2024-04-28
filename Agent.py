import gymnasium as gym
import torch as th
import numpy as np
from Models import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import termcolor
from enum import Enum
import cma

ROUNDING_PRECISION = 2

class AgentType(Enum):
    """
        The AgentType class is an enumeration class for the type of the agent model.

        Attributes:
            ACBaseline (int): The Actor-Critic model with Baseline Subtraction.
            ACBootstrapping (int): The Actor-Critic model with Boostrapping.
            ACBootstrappingBaseline (int): The Actor-Critic model with both Boostrapping and Baseline Subtraction.
            REINFORCE (int): The REINFORCE model.
    """
    ACBaseline = 0
    ACBootstrapping = 1
    ACBootstrappingBaseline = 2
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
        n_steps:int = 20,
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
    
    def __get_training_trajectory(self,
                                    actor_loss: th.Tensor, 
                                    critic_loss: th.Tensor, 
                                    rewards: np.ndarray,
                                    entropy: float) -> dict:
        """
            The private method to get the training trajectory.

            Args:
                actor_loss (th.Tensor): The actor loss.
                critic_loss (th.Tensor): The critic loss.
                rewards (np.ndarray): The rewards of the agent.

            Returns:
                dict: The dictionary of the training trajectory.
        """
        d  = {}
        color = lambda x: termcolor.colored(x, "green") if x > 0 else termcolor.colored(x, "yellow") if x > -100.0 else termcolor.colored(x, "red")
        avg_reward = np.round(np.mean(rewards), ROUNDING_PRECISION)

        if self.agent_type != AgentType.REINFORCE:
            d["Actor Loss"] = f"{actor_loss:1.{ROUNDING_PRECISION}}"
            d["Critic Loss"] = f"{critic_loss:1.{ROUNDING_PRECISION}}"
        else:
            d["Model Loss"]  = f"{actor_loss:1.{ROUNDING_PRECISION}}"

        d["Avg Reward"] = f"{color(avg_reward)}"
        d["Last reward"] =  f"{color(np.round(rewards[-1], ROUNDING_PRECISION))}"
        d["Entropy"] = f"{entropy:.{ROUNDING_PRECISION}f}"

        return d

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
        episode_log_probs = th.stack(episode_log_probs).to(self.device)
        episode_values = th.stack(episode_values).to(self.device).squeeze()

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
    
    def __get_cumulative_returns(self, rewards: th.Tensor, values: th.Tensor) -> th.Tensor:
        """
            The private method to get the cumulative returns.

            Args:
                rewards (np.ndarray): The rewards of the agent.
                values (np.ndarray): The values of the agent.

            Returns:
                np.ndarray: The cumulative returns.
        """

        cumulative_returns = th.zeros(len(rewards), dtype=th.float).to(self.device)
        discounts = th.tensor([self.gamma ** i for i in range(self.n_steps)]).to(self.device)

        for t in range(len(cumulative_returns)):
            n = min(self.n_steps, len(cumulative_returns) - t)
            if t + n < len(cumulative_returns):
                cumulative_returns[t] = th.sum(rewards[t:t+n] * discounts[:n]) + discounts[n-1] * values[t+n]
            else:
                cumulative_returns[t] = th.sum(rewards[t:] * discounts[:n]) + discounts[-1] * values[-1]
        return cumulative_returns
    
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

        if self.agent_type == AgentType.ACBaseline or self.agent_type == AgentType.ACBootstrappingBaseline:
            advantage = discounted_rewards - values
        else:
            advantage = discounted_rewards

        policy_loss = -th.sum(log_probs * advantage) - self.beta * entropy

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

        critic_loss = th.sum((values - returns) ** 2)

        return critic_loss
    
    def train(self, n_episodes, measure_loss_variance:bool = True) -> tuple[list[float], list[float]] | list[float]:
        rewards: list[float] = []
        loss_variances: list[float] = []

        for _ in (progress_bar := tqdm(range(n_episodes), position=0, leave=True)):
            episode_rewards, episode_log_probs, episode_values, entropy = self.__get_trace()
            if self.agent_type == AgentType.ACBootstrapping or self.agent_type == AgentType.ACBootstrappingBaseline:
                r = self.__get_cumulative_returns(episode_rewards, episode_values)
            else:
                r = self.__get_discounted_rewards(episode_rewards)
            
            policy_loss = self.__get_policy_loss(episode_log_probs, r, episode_values, entropy)

            loss_variances.append(policy_loss.item())

            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True if self.agent_type != AgentType.REINFORCE else False)
            self.optimizer.step()

            

            if self.agent_type != AgentType.REINFORCE:
                critic_loss = self.__get_critic_loss(episode_values, r)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


            rewards.append(sum(episode_rewards).item())

            traj_dict = self.__get_training_trajectory( 
                    policy_loss, 
                    th.tensor(0.0) if self.agent_type == AgentType.REINFORCE else critic_loss,
                    rewards,
                    entropy)
            
            progress_bar.set_postfix(traj_dict)
            
        if measure_loss_variance:
            return rewards, loss_variances
        else:
            return rewards


if __name__ == '__main__':
    agent = Agent(agent_type=AgentType.ACBootstrappingBaseline)
    rewards, gradients = agent.train(1000)
    rewards = np.convolve(rewards, np.ones(100), 'valid') / 100
    gradients = np.convolve(gradients, np.ones(100), 'valid') / 100
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(rewards)
    ax[0].set_title('Rewards')
    ax[1].plot(gradients)
    ax[1].set_title('Gradient Variance')
    plt.savefig('rewards_gradients.png')