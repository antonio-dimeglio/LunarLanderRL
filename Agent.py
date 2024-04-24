import gymnasium as gym
import torch as th
import numpy as np
from Models import *
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import termcolor
from enum import Enum

ROUNDING_PRECISION = 2

Trace = namedtuple("Trace", ["rewards", "values", "log_probs", "entropies"])
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
            minibatch_size (int): The size of the minibatch for the training, setting this value to None 
                                    will disable the minibatch training.
    """
    def __init__(self,
                alpha: float = 0.01,
                beta: float = 0.01, 
                gamma: float = 0.99,
                model: AgentType = AgentType.ACBoostrappingBaseline,
                n_steps: int = 5,
                environment: str = "LunarLander-v2",
                minibatch_size: int = None) -> None:
        

        self.env = gym.make(environment, continuous = False)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        match model:
            case AgentType.ACBaseline | AgentType.ACBoostrapping | AgentType.ACBoostrappingBaseline:
                self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
                self.critic = Critic(self.env.observation_space.shape[0]).to(self.device)
            case AgentType.REINFORCE:
                self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
            case _:
                raise ValueError(f"Invalid model type: {model}.")
        
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = None if model == AgentType.REINFORCE else th.optim.Adam(self.critic.parameters(), lr=alpha)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model = model
        self.n_steps = n_steps
        self.minibatch_size = minibatch_size

        if minibatch_size is not None:
            self.done = False 
            self.state = None


    def __get_trace(self) -> Trace:
        rewards = []
        values = []
        log_probs = []
        entropies = []

        state, _ = self.env.reset()
        done = False 

        while not done:
            state = th.tensor(state, dtype=th.float32, requires_grad=False).to(self.device)
            probs = self.actor(state)
            value = self.critic(state) if self.model != AgentType.REINFORCE else None

            dist = th.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, terminated, truncated, _ = self.env.step(action.item())

            rewards.append(abs(reward))
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)

            done = terminated or truncated

        return Trace(
            th.tensor(rewards, dtype=th.float32, requires_grad=False).to(self.device),
            th.tensor(values, dtype=th.float32, requires_grad=False).to(self.device) if self.model != AgentType.REINFORCE else None,
            th.tensor(log_probs, dtype=th.float32, requires_grad=False).to(self.device),
            th.tensor(entropies, dtype=th.float32, requires_grad=False).to(self.device)
        )
    
    def __get_minibatch_trace(self) -> Trace:
        rewards = []
        values = []
        log_probs = []
        entropies = []

        if self.done or self.state is None:
            self.done = False
            state, _ = self.env.reset()
        else:
            state = self.state 

        for _ in range(self.minibatch_size):
            state = th.tensor(state, dtype=th.float32, requires_grad=False).to(self.device)
            probs = self.actor(state)
            value = self.critic(state) if self.model != AgentType.REINFORCE else None

            dist = th.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, terminated, truncated, _ = self.env.step(action.item())

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)

            self.done = terminated or truncated

            if self.done:
                self.done = False
                state, _ = self.env.reset()
                
        self.state = state

        return Trace(
            th.tensor(rewards, dtype=th.float32, requires_grad=False).to(self.device),
            th.tensor(values, dtype=th.float32, requires_grad=False).to(self.device) if self.model != AgentType.REINFORCE else None,
            th.tensor(log_probs, dtype=th.float32, requires_grad=False).to(self.device),
            th.tensor(entropies, dtype=th.float32, requires_grad=False).to(self.device)
        )

    def __get_cumulative_returns(self, rewards: th.Tensor, values: th.Tensor) -> th.Tensor:
        """
            Calculate the cumulative returns of the agent.

            Args:
                rewards (th.Tensor): The rewards of the agent.
                values (th.Tensor): The values of the agent estimated by the Critic.
        """
        cumulative_returns = th.zeros_like(rewards, dtype=th.float32, device=self.device, requires_grad=False)
        
        for t in range(len(cumulative_returns)):
            n = min(self.n_steps, len(cumulative_returns) - t - 1)
            for k in range(n):
                cumulative_returns[t] += rewards[t+k] + values[t+n]
        
        return cumulative_returns

    def __get_returns(self, rewards:th.Tensor) -> th.Tensor:
        returns = deque(maxlen=len(rewards))
        R = 0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns.appendleft(R)

        return th.tensor(returns, dtype=th.float32, requires_grad=False).to(self.device)

    def __get_critic_loss(self, cumulative_returns: th.Tensor, values:th.Tensor) -> th.Tensor:
        critic_loss = th.tensor(0.0, dtype=th.float32, requires_grad=True).to(self.device)

        for t in range(len(cumulative_returns)):
            critic_loss += th.square(cumulative_returns[t] - values[t])
        
        return critic_loss


    def __get_actor_loss(self, cumulative_returns: th.Tensor, log_probs:th.Tensor, entropies:th.Tensor, values: th.Tensor = None) -> th.Tensor:
        if self.model == AgentType.ACBoostrappingBaseline or self.model == AgentType.ACBaseline:
            advantages = cumulative_returns - values
        else:
            advantages = cumulative_returns

        actor_loss = th.tensor(0.0, dtype=th.float32, requires_grad=True).to(self.device)

        for t in range(len(cumulative_returns)):
            actor_loss += -log_probs[t] * advantages[t] - self.beta * entropies[t]

        return actor_loss



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
            trace = self.__get_trace() if self.minibatch_size is None else self.__get_minibatch_trace()
            if self.model == AgentType.ACBoostrapping or self.model == AgentType.ACBoostrappingBaseline:
                returns = self.__get_cumulative_returns(trace.rewards, trace.values)
            else:
                returns = self.__get_returns(trace.rewards)

            actor_loss = self.__get_actor_loss(returns, trace.log_probs, trace.entropies, trace.values)
            actor_loss_value = actor_loss.item()

            if self.model != AgentType.REINFORCE:
                
                critic_loss = self.__get_critic_loss(returns, trace.values)

                # Normalize the critic loss
                critic_loss_value = critic_loss.item()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                del critic_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            del actor_loss

            rewards_per_episode[i] = sum(trace.rewards)

            if not quiet and i % 10 == 0:
                print(f"Episode {i}/{m}", end="\t")
                if self.model != AgentType.REINFORCE:
                    print(f"Actor Loss: {actor_loss_value:.{ROUNDING_PRECISION}f}\tCritic Loss: {critic_loss_value:.{ROUNDING_PRECISION}f}", end="\t")
                else:
                    print(f"Model Loss: {actor_loss_value:.{ROUNDING_PRECISION}f}", end="\t")

                print("Total average reward: ", end="")
                avg_reward = np.mean(rewards_per_episode[:i])
                if avg_reward > 0.0:
                    print(termcolor.colored(f"{avg_reward}", "green"))
                elif avg_reward > -100.0:
                    print(termcolor.colored(f"{avg_reward}", "yellow"))
                else:
                    print(termcolor.colored(f"{avg_reward}", "red"))

        return rewards_per_episode


    def visualize_agent(self, trials:int = 20) -> None:
        self.env = gym.make("LunarLander-v2", continuous = False, render_mode = "human")

        for i in range(trials):
            state, _ = self.env.reset()
            done = False

            while not done:
                state = th.tensor(state, dtype=th.float32, requires_grad=False).to(self.device)
                probs = self.actor(state)
                dist = th.distributions.Categorical(probs)
                action = dist.sample()
                state, _, done, _, _ = self.env.step(action.item())
                self.env.render()



if __name__ == "__main__":
    agent = Agent(model=AgentType.ACBoostrappingBaseline, minibatch_size=5)
    rewards = agent.train(10000)
    agent.visualize_agent()

    smoothed_rewards = np.convolve(rewards, np.ones(100)/100, mode="valid")

    plt.plot(smoothed_rewards)
    plt.title("Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("rewards.png")
