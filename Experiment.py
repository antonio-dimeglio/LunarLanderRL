from Agent import *
import matplotlib.pyplot as plt
import numpy as np

def smooth(x, window=100):
    return np.convolve(x, np.ones(window), 'valid') / window



def experiment_different_agent():
    agent_types = [
        AgentType.REINFORCE,
        AgentType.ACBaseline,
        AgentType.ACBoostrapping,
        AgentType.ACBoostrappingBaseline
    ]

    num_episodes = 1000

    for agent_type in agent_types:
        agent = Agent(model=agent_type)
        rewards = agent.train(m=num_episodes)
        plt.plot(smooth(rewards), label=agent_type)

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Different Agent Comparison')
    plt.savefig('different_agent_comparison.png')




if __name__ == '__main__':
    experiment_different_agent()