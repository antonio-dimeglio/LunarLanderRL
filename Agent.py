import gymnasium as gym
import torch as th
import numpy as np
from Models import *
from collections import namedtuple
import matplotlib.pyplot as plt
import termcolor

ROUNDING_PRECISION = 2

Trace = namedtuple("Trace", ["reward", "values", "log_prob", "entropy"])

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
    """

    def __init__(self, 
                alpha: float = 0.01, 
                beta: float = 1e-2, 
                gamma: float = 0.99,
                model_type: str = "ACBootbaseline",
                n: int = 5) -> None:
        self.env = gym.make("LunarLander-v2", continuous=False)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model_type = model_type
        self.n = n

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        match model_type:
            case "ACBoot" | "ACBaseline" | "ACBootbaseline":
                self.model = ActorCritic(self.env.observation_space.shape[0], 128, self.env.action_space.n).to(self.device)
            case "REINFORCE":
                raise ValueError("Unimplemented model type.")
            case _:
                raise ValueError("Invalid model type. Please choose from 'ACBoot', 'ACBaseline', 'ACBootbaseline' or 'REINFORCE'.")
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=alpha)

    def __collect_trace(self) -> Trace:
        """
            Collect a trace from the environment.

            Returns:
                Trace: The trace of the episode.
        """
        
        state, _ = self.env.reset()
        done = False

        rewards = []
        values = []
        log_probs = []
        entropies = []

        while not done:
            state = th.tensor(state, dtype=th.float32).to(self.device)
            action, log_prob, value, entropy = self.model(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)
            state = next_state
            done = terminated or truncated
        
        rewards = th.tensor(rewards, dtype=th.float32).to(self.device)
        values = th.tensor(values, dtype=th.float32).to(self.device)
        log_probs = th.tensor(log_probs, dtype=th.float32).to(self.device)
        entropies = th.tensor(entropies, dtype=th.float32).to(self.device)

        return Trace(rewards, values, log_probs, entropies)
    
    def __estimate_cumulative_return(self, rewards: th.Tensor, values: th.Tensor) -> th.Tensor:
        """
            Estimate the cumulative return of the episode.

            Args:
                rewards (th.Tensor): The rewards of the episode.
                values (th.Tensor): The values of the episode.

            Returns:
                th.Tensor: The cumulative return of the episode.
        """
        returns = th.zeros_like(rewards)
        
        for t in range(len(returns) - 1):
            n = min(self.n, len(returns) - t - 1)
            for k in range(n):
                returns[t] += rewards[t+k] + values[t+n]
        
        return returns
    
    def __compute_returns(self, rewards: th.Tensor) -> th.Tensor:
        """
            Compute the returns of the episode.

            Args:
                rewards (th.Tensor): The rewards of the episode.

            Returns:
                th.Tensor: The returns of the episode.
        """
        returns = th.zeros_like(rewards)
        R = 0.0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        
        return returns
    
    def __compute_critic_loss(self, returns: th.Tensor, values: th.Tensor) -> th.Tensor:
        """
            Compute the critic loss.

            Args:
                returns (th.Tensor): The returns of the episode.
                values (th.Tensor): The values of the episode.

            Returns:
                th.Tensor: The critic loss.
        """
        critic_loss = th.tensor(0.0).to(self.device)

        for t in range(len(returns)):
            critic_loss += (returns[t] - values[t]) ** 2
        
        return critic_loss 
    
    def __compute_actor_loss(self, returns: th.Tensor, 
                            values: th.Tensor, 
                            log_probs: th.Tensor, 
                            entropies: th.Tensor,
                            baseline: bool) -> th.Tensor:
        """
            Compute the actor loss.

            Args:
                returns (th.Tensor): The returns of the episode.
                values (th.Tensor): The values of the episode.
                log_probs (th.Tensor): The log probabilities of the episode.

            Returns:
                th.Tensor: The actor loss.
        """
        actor_loss = th.tensor(0.0).to(self.device)
        entropy_loss = th.tensor(0.0).to(self.device)

        for t in range(len(returns)):
            if baseline:
                advantage = returns[t] - values[t]
            else:
                advantage = returns[t]

            actor_loss += -log_probs[t] * advantage
            entropy_loss += entropies[t]
        
        return actor_loss - self.beta * entropy_loss

    
    def __ac_training(self, 
                    m: int, 
                    quiet: bool = False,
                    bootstrap: bool = False, 
                    baseline:bool = False) -> None:
        """
            Training loop for the Actor-Critic model.
            
            Args:
                m (int): The number of episodes to train the agent.
                quiet (bool): Whether to suppress the output of the training process.
                bootstrap (bool): Whether to use bootstrapping for the critic.
                baseline (bool): Whether to use a baseline for the critic.
        """
        losses = np.zeros(m)
        total_rewards = np.zeros(m)
        timesteps = 0

        for i in range(m):
            
            trace = self.__collect_trace()
            returns = self.__estimate_cumulative_return(trace.reward, trace.values) if bootstrap else self.__compute_returns(trace.reward)
            normalized_returns = (returns - returns.mean()) / (returns.std())
            actor_loss = self.__compute_actor_loss(normalized_returns, trace.values, trace.log_prob, trace.entropy, baseline)
            critic_loss = self.__compute_critic_loss(normalized_returns, trace.values)
            loss = th.tensor(actor_loss.clone().detach() + critic_loss.clone().detach(), requires_grad=True).to(self.device)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses[i] = loss.item()

            if not quiet:
                total_reward = sum(trace.reward)
                total_rewards[i] = total_reward
                print(f"Episode {i+1}/{m}\t\tLoss: {losses[i]:.{ROUNDING_PRECISION}f}\t\t", end="")
                if total_reward > 0.0:
                    print(termcolor.colored(f"Total Reward: {total_reward:.{ROUNDING_PRECISION}f}", "green"))
                elif total_reward > -100.0:
                    print(termcolor.colored(f"Total Reward: {total_reward:.{ROUNDING_PRECISION}f}", "yellow"))
                else:
                    print(termcolor.colored(f"Total Reward: {total_reward:.{ROUNDING_PRECISION}f}", "red"))

            timesteps+=len(trace.reward)

        smoothed_rewards = np.convolve(total_rewards, np.ones(100)/100, mode="valid")
        plt.plot(smoothed_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward vs Episode")
        plt.savefig("Total_Reward_vs_Episode.png")
        print(f"Total Timesteps: {timesteps}")


    def __reinforce_training(self, m: int, quiet: bool = False) -> None:
        """
            Training loop for the REINFORCE model.

            Args:
                m (int): The number of episodes to train the agent.
                quiet (bool): Whether to suppress the output of the training process.
        """
        pass 




    def train(self, m: int, quiet: bool = False) -> None:
        """
            Entry point for training the agent.

            Args:
                m (int): The number of episodes to train the agent.
                quiet (bool): Whether to suppress the output of the training process.
        """

        match self.model_type:
            case "ACBoot":
                self.__ac_training(m, quiet, bootstrap=True)
            case "ACBaseline":
                self.__ac_training(m, quiet, baseline=True)
            case "ACBootbaseline":
                self.__ac_training(m, quiet, bootstrap=True, baseline=True)
            case "REINFORCE":
                self.__reinforce_training(m, quiet)
            case _:
                raise ValueError("Invalid model type. Please choose from 'ACBoot', 'ACBaseline', 'ACBootbaseline' or 'REINFORCE'.")


    def visualize_agent(self) -> None:
        """
            Visualize the agent in the environment.
        """
        self.env = gym.make("LunarLander-v2", continuous=False, render_mode = "human")
        while True:
            state, _ = self.env.reset()
            done = False

            while not done:
                self.env.render()
                state = th.tensor(state, dtype=th.float32).to(self.device)
                action, _, _ = self.model(state)
                next_state, _, terminated, truncated, _ = self.env.step(action)
                state = next_state
                done = terminated or truncated
            
            self.env.close()    



if __name__ == "__main__":
    agent = Agent(model_type="ACBootbaseline")
    agent.train(2000)
    # agent.visualize_agent()