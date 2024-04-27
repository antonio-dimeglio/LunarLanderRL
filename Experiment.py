from Agent import *
import matplotlib.pyplot as plt
import numpy as np

def smooth(x, window=100):
    return np.convolve(x, np.ones(window), 'valid') / window



def experiment_different_agent():
    print('Running experiment_different_agent.')
    agent_types = [
        AgentType.REINFORCE,
        AgentType.ACBaseline,
        AgentType.ACBoostrapping,
        AgentType.ACBoostrappingBaseline
    ]

    num_episodes = 1000

    for agent_type in agent_types:
        print(f'Running {agent_type}.')
        agent = Agent(agent_type=agent_type)
        rewards = agent.train(num_episodes)
        plt.plot(smooth(rewards), label=agent_type)
        print(f'{agent_type} done.')

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Different Agent Comparison')
    plt.savefig('different_agent_comparison.png')

    print('Experiment done.')



if __name__ == '__main__':
    experiment_different_agent()