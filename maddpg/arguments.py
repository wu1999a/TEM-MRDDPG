import argparse
from pickle import FALSE, TRUE
from numpy import True_
from torch import true_divide

"""
Here are the param for the training

"""
def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="scenes1", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--train-episode", type=int, default=5000, help="train-episode")
    parser.add_argument("--time-steps", type=int, default=450000, help="number of time steps")
    parser.add_argument("--begain_learn_episode", type=int, default=50, help="begain learning ")
    parser.add_argument("--decate", type=float, default=0.999, help="epsilon zhekou")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--evaluate-episodes", type=int, default=1, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=10, help="how often to evaluate model")
    parser.add_argument("--if_savefig", type=bool, default=False, help="if_savefig")
    args = parser.parse_args()

    return args
