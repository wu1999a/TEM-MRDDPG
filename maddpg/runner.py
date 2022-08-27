from statistics import mode
from kiwisolver import Expression
from regex import P
from tqdm import tqdm
from agent import Agent
from replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pandas as pd

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        episode_list=[]
        for episode in tqdm(range(self.args.train_episode)):

            s = self.env.reset()

            rewards = 0
            for time_step in range(100):
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):

                        action = agent.select_action(s[agent_id], self.noise, self.epsilon,self.env,agent_id)

                        u.append(np.round(action,2))
                        actions.append(np.round(action,2))

                
                s_next, r,done_list,info_list= self.env.step(time_step,actions,vel_type = 'omni',stop=True)
            
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next

                if episode>= self.args.begain_learn_episode:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents) 
                # if   not False in info_list :          
                #     break
                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.epsilon - 0.0000005)
            if episode > 0 and episode % self.args.evaluate_rate == 0:
                # print(time_step,rewards)
                # print(" self.epsilon:",self.epsilon)
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.close()
                # np.save(self.save_path + '/returns.pkl', returns)
                episode_list.append(episode)
                dataframe = pd.DataFrame({'episode': episode_list, 'returns': returns})
                dataframe.to_csv(self.args.scenario_name+".csv", index=False, sep=',')

    def evaluate(self):
        returns = []
        save_step =0
       
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if    self.args.evaluate: 
                    if self.args.if_savefig:
                        save_step = episode*self.args.evaluate_episode_len + time_step
                        self.env.save_fig(image_path,save_step ) 
                    self.env.render(show_traj=True, show_goal=True)
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0,self.env,agent_id)
    
                        actions.append(action)

                s_next, r,done_list,info_list= self.env.step(time_step,actions,vel_type = 'omni',stop=True)
                # print("s:",s,"s_next:",s_next,"action:",actions,"reward:",r)
                s = s_next
                # if  not False in info_list :
                #     break
            rewards += sum(r)
            returns.append(rewards)
            print('Returns is', rewards)
        if self.args.if_savefig:
            self.env.save_ani(image_path, gif_path)
        return sum(returns) / self.args.evaluate_episodes