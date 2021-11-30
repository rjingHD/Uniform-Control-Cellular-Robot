import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="Cell RL Project")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'cell_RL'
        env = Environment(env_name, args)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('cell_RL', args, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=10)


if __name__ == '__main__':
    args = parse()
    run(args)
