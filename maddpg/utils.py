import numpy as np
import inspect
import functools
import sys

import MADDPG_v01.env.mapf_sim.env.env_base as eb


def make_env(args):

    world_name = 'config/dynamic_obs.yaml'
    env = eb.env_base(world_name = world_name, plot=True,teleop_key=True)

    args.n_agents =env.robot_number
    # state_shape =env.observation_space.shape[0]
    # args.action_shape = [env.action_space.shape[0]  for i in range(args.n_agents)]
    state_shape = 10
    args.action_shape = [2 for i in range(args.n_agents)]
    # print(env.get_obsshape())
    
    args.obs_shape = [env.get_obsshape() for i in range(args.n_agents)]  # 每一维代表该agent的obs维度i
    args.n_players = 0  # 包含敌人的所有玩家个数

    args.high_action = 1.5
    args.low_action = -1.5
    return env, args
