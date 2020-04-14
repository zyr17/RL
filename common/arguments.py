import argparse
import multiprocessing
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('-a', '--algo', type=str, default='a2c')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-n', '--n-timesteps', type=int, default=1000000)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-t', '--target-reward', type=float, default=1e100)
    parser.add_argument('-tx', '--tensorboardx-comment', type=str, default='')
    parser.add_argument('-s', '--save-path', type=str, default='model.pt')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--threads', type=int, default=-1)
    parser.add_argument('-r', '--render-steps', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('-c', '--config', type=str, default='')

    # DQN args
    parser.add_argument('-Dr', '--dqn-replay-size', type=int, default=10000)
    parser.add_argument('-Da', '--dqn-alpha', type=float, default=0.1)
    parser.add_argument('-Dei', '--dqn-initial-eps', type=float, default=1.0)
    parser.add_argument('-Def', '--dqn-final-eps', type=float, default=0.02)
    parser.add_argument('-Del', '--dqn-explore-len-frac', type=float, default=0.1)
    parser.add_argument('-Du', '--dqn-net-update-freq', type=int, default=1000)
    parser.add_argument('-Dn', '--dqn-n-steps', type=int, default=1)
    parser.add_argument('-DN', '--dqn-noisynet', action='store_true', default=False)
    parser.add_argument('-DDo', '--double-dqn', action='store_true', default=False)
    parser.add_argument('-DDu', '--dueling-dqn', action='store_true', default=False)
    parser.add_argument('-DDi', '--distributional-dqn', action='store_true', default=False)
    parser.add_argument('-DP', '--prioritized-dqn', action='store_true', default=False)
    # TODO: prioritized DQN and distributional dqn has other params

    # A2C args
    parser.add_argument('-An', '--a2c-n-steps', type=int, default=5)
    parser.add_argument('-Ae', '--a2c-entropy-beta', type=float, default=0.01)
    parser.add_argument('-Ac', '--a2c-clip-grad', type=float, default=0.1)

    args = parser.parse_args()

    # if config file is set, read it and parse it before cmd arguments
    if args.config != '':
        confs = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        carg = []
        for conf in confs.keys():
            carg.append('--' + conf)
            if type(confs[conf]) != type(True):
                carg.append(str(confs[conf]))
        print('config:', carg, confs, carg + sys.argv[1:])
        args = parser.parse_args(carg + sys.argv[1:])

    #if thread number not set, use half cpus
    if args.threads == -1:
        cpu_count = multiprocessing.cpu_count() // 2
        if cpu_count < 1:
            cpu_count = 1
        args.threads = cpu_count

    return args

if __name__ == "__main__":
    import sys
    print(sys.argv)
    print(parse_args())