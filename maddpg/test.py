from requests import PreparedRequest
from torch import true_divide

from arguments import get_args
from utils import make_env


if __name__ == '__main__':

    args = get_args()

    env,args= make_env(args)
    for i in range (10):
        s = env.reset()
        print(s)
        for time_step in range(100):
            # env.robot_step(env.key_vel)
            # des_vel_list = []
            # for i in range (args.n_agents):
            #     des_vel = env.get_des_vel(i)
            #     des_vel_list.append(des_vel)
            # obs_list, reward_list,done_list,info_list=env.step(time_step,des_vel_list,vel_type = 'diff',stop=False)
            # if(env.key_vel[0]>0):
            # if time_step%10==0:
            #     print('des_vel:',des_vel_list)
            #     print("obs_list:",obs_list,"reward_list:",reward_list)
            #     print('\n')
   

        # if  True in done_list or not False in info_list:
        #         break
            env.render(show_traj=True, show_goal=True)

        # if env.collision_check() or env.arrive_check():
        #     break

    # env.show()


