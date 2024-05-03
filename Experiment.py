from Agent import *
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap 


def smooth(x, window=200):
    return np.convolve(x, np.ones(window), 'valid') / window


def experiment_different_agent(n:int):
    print('Running experiment_different_agent.')
    agent_types = [
        AgentType.REINFORCE,
        AgentType.ACBaseline,
        AgentType.ACBootstrapping,
        AgentType.ACBootstrappingBaseline
    ]
    
    agent_rewards = []
    agent_losses = []

    for agent_type in agent_types:
        print(f'Running {agent_type.name}.')
        agent = Agent(agent_type=agent_type)
        rewards, losses = agent.train(n, True)
        rewards = smooth(rewards)
        losses = smooth(losses, 50)
        
        agent_rewards.append(rewards)
        agent_losses.append(losses)


        print(f'{agent_type} done.')

    plt.figure()
    plt.title('Average Rewards')
    for i, agent_type in enumerate(agent_types):
        plt.plot(agent_rewards[i], label=agent_type.name)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_agents.png')
    plt.clf()

    plt.figure()
    plt.title('Gradients Variance')

    for i, agent_type in enumerate(agent_types):
        plt.plot(agent_losses[i], label=agent_type.name)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    plt.savefig('losses_different_agents.png')
    plt.clf()

    
    print('Experiment done.')

def experiment_different_lr_actor(n:int):
    lrs = [0.0001, 0.001, 0.01]
    
    plt.figure()
    plt.title('Average Rewards with Different Learning Rates for Actor')

    for lr in lrs:
        print(f'Running experiment with lr = {lr}.')
        agent = Agent(agent_type=AgentType.ACBootstrappingBaseline, alpha_policy=lr)
        rewards = agent.train(n, False)
        rewards = smooth(rewards)
        plt.plot(rewards, label=r"$\alpha$ = " + str(lr))

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_lr_actor.png')
    plt.clf()

def experiment_different_lr_critic(n:int):
    lrs = [0.0001, 0.001, 0.01, 0.1]

    plt.figure()
    plt.title('Average Rewards with Different Learning Rates for Critic')

    for lr in lrs:
        print(f'Running experiment with lr = {lr}.')
        agent = Agent(agent_type=AgentType.ACBootstrappingBaseline, alpha_critic=lr)
        rewards = agent.train(n, False)
        rewards = smooth(rewards)
        plt.plot(rewards, label=r"$\alpha_c$ = " + str(lr))

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_lr_critic.png')
    plt.clf()

def experiment_different_gamma(n:int):
    gammas = [0.9, 0.95, 0.99, 0.999]

    plt.figure()
    plt.title('Average Rewards with Different Discount Factors')

    for gamma in gammas:
        print(f'Running experiment with gamma = {gamma}.')
        agent = Agent(agent_type=AgentType.ACBootstrappingBaseline, gamma=gamma)
        rewards = agent.train(n, False)
        rewards = smooth(rewards)
        plt.plot(rewards, label=r"$\gamma$ = " + str(gamma))

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_gamma.png')
    plt.clf()



def experiment_different_beta(n:int):
    betas = [0.01, 0.1, 0.2, 0.5]

    plt.figure()
    plt.title('Average Rewards with Different Entropy Coefficients')

    for beta in betas:
        print(f'Running experiment with beta = {beta}.')
        agent = Agent(agent_type=AgentType.ACBootstrappingBaseline, beta=beta)
        rewards = agent.train(n, False)
        rewards = smooth(rewards)
        plt.plot(rewards, label=r"$\beta$ = " + str(beta))

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_beta.png')
    plt.clf()

def experiment_different_n_steps(n:int):
    n_steps = [1, 2, 5, 10, 20, 50]

    plt.figure()
    plt.title('Average Rewards with Different N-Step Returns')

    for n_step in n_steps:
        print(f'Running experiment with n_step = {n_step}.')
        agent = Agent(agent_type=AgentType.ACBootstrappingBaseline, n_steps=n_step)
        rewards = agent.train(n, False)
        rewards = smooth(rewards)
        plt.plot(rewards, label=r"$n$ = " + str(n_step))

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_n_steps.png')
    plt.clf()

def experiment_different_agent_cartpole(n:int):
    print('Running experiment_different_agent_cartpole.')
    agent_types = [
        AgentType.REINFORCE,
        AgentType.ACBaseline,
        AgentType.ACBootstrapping,
        AgentType.ACBootstrappingBaseline
    ]
    
    agent_rewards = []
    agent_losses = []

    for agent_type in agent_types:
        print(f'Running {agent_type.name}.')
        agent = Agent(agent_type=agent_type, environment='CartPole-v1')
        rewards, losses = agent.train(n, True)
        rewards = smooth(rewards)
        losses = smooth(losses, 50)
        
        agent_rewards.append(rewards)
        agent_losses.append(losses)


        print(f'{agent_type} done.')

    plt.figure()
    plt.title('Average Rewards')
    for i, agent_type in enumerate(agent_types):
        plt.plot(agent_rewards[i], label=agent_type.name)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('rewards_different_agents_cartpole.png')
    plt.clf()

    plt.figure()
    plt.title('Gradients Variance')

    for i, agent_type in enumerate(agent_types):
        plt.plot(agent_losses[i], label=agent_type.name)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    plt.savefig('losses_different_agents_cartpole.png')
    plt.clf()

    
    print('Experiment done.')


if __name__ == '__main__':
    parser = ap.ArgumentParser(
        prog="Lunar Lander Experiment",
        description="Run different experiments on the Lunar Lander environment."
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3000,
        help="Number of episodes to run.",
        required=False
    )

    parser.add_argument(
        "--run_all",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run all experiments.",
        required=False
    )

    parser.add_argument(
        "--run_different_agent",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different agents.",
        required=False
    )

    parser.add_argument(
        "--run_different_lr_actor",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different learning rates for actor.",
        required=False
    )

    parser.add_argument(
        "--run_different_lr_critic",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different learning rates for critic.",
        required=False
    )

    parser.add_argument(
        "--run_different_gamma",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different discount factors.",
        required=False
    )

    parser.add_argument(
        "--run_different_beta",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different entropy coefficients.",
        required=False
    )

    parser.add_argument(
        "--run_different_n_steps",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different n-step returns.",
        required=False
    )

    parser.add_argument(
        "--cartpole",
        type=bool,
        action=ap.BooleanOptionalAction,
        default=False,
        help="Run different agents on CartPole environment.",
        required=False
    )

    args = parser.parse_args()

    num_episodes = args.num_episodes

    if args.run_all:
        experiment_different_agent(num_episodes)
        experiment_different_lr_actor(num_episodes)
        experiment_different_lr_critic(num_episodes)
        experiment_different_gamma(num_episodes)
        experiment_different_beta(num_episodes)
        experiment_different_n_steps(num_episodes)
        experiment_different_agent_cartpole(num_episodes)
    else:
        if args.run_different_agent:
            experiment_different_agent(num_episodes)
        if args.run_different_lr_actor:
            experiment_different_lr_actor(num_episodes)
        if args.run_different_lr_critic:
            experiment_different_lr_critic(num_episodes)
        if args.run_different_gamma:
            experiment_different_gamma(num_episodes)
        if args.run_different_beta:
            experiment_different_beta(num_episodes)
        if args.run_different_n_steps:
            experiment_different_n_steps(num_episodes)
        if args.cartpole:
            experiment_different_agent_cartpole(num_episodes)