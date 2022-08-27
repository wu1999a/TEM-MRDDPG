import torch
import os
from actor_critic import Actor, Critic

class MADDPG:
    def __init__(self, args, agent_id):  # 
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.a=0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if self.args.evaluate: 
            self.actor_network.load_state_dict(torch.load('./model/' + self.args.scenario_name+ '/agent_' + str(self.agent_id)+'/233_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load('./model/'+ self.args.scenario_name + '/agent_' + str(self.agent_id)+'/233_critic_params.pkl'))

        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        #     self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
        #     self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
        #     print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                   self.model_path + '/actor_params.pkl'))
        #     print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                    self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 
        # print("reward self.agent_id:",self.agent_id,"----",r)

        o, u, o_next = [], [], [] 
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
    
        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
  
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
        
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        # print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
      
        self.a +=1
        if self.a%100 == 0:
            self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
    

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


