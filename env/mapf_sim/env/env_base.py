from dis import dis
import imp
import yaml
import numpy as np
import sys
import matplotlib.pyplot as plt

# from env.mapf_sim.world import env_plot, mobile_robot, car_robot, obs_circle, obs_polygon
from  env_plot import*
from  mobile_robot import*
from  env_robot import*
import env_grid
from PIL import Image
from pynput import keyboard


class env_base:

    def __init__(self, world_name=None, plot=True,  **kwargs):

        self.reward_parameter = (0.45, 0.01, 0.1, 0.2, 0.2, 1, -5, 1)
        self.dis_reward_list = [0 for i in range (4)]
        self.disreward =2
        # p1, p2, p3, p4, p5, p6, p7, p8 = self.reward_parameter
        # p1 = 距离奖励 dis*p1
        # p2 = 时间步数奖励 time_step*p2
        # p7 = 碰撞
        # p8 = 到达终点


        if world_name != None:
            world_name = sys.path[0] + '/' + world_name

            with open(world_name) as file:
                com_list = yaml.load(file, Loader=yaml.FullLoader)

                world_args = com_list['world']
                self.__height = world_args.get('world_height', 10)
                self.__width = world_args.get('world_width', 10)
                self.offset_x = world_args.get('offset_x', 0)
                self.offset_y = world_args.get('offset_y', 0)
                self.step_time = world_args.get('step_time', 0.1)
                self.world_map = world_args.get('world_map', None)
                self.xy_reso = world_args.get('xy_resolution', 1)
                self.yaw_reso = world_args.get('yaw_resolution', 5)
                self.offset = np.array([self.offset_x, self.offset_y])

                self.robots_args = com_list.get('robots', dict())
                self.robot_number = kwargs.get(
                    'robot_number', self.robots_args.get('robot_number', 0))
                self.robot_init_mode=kwargs.get(
                    'robot_init_mode', self.robots_args.get('robot_init_mode', 0))

                self.cars_args = com_list.get('cars', dict())
                self.car_number = self.cars_args.get('number', 0)
                # obs_cir
                self.obs_cirs_args = com_list.get('obs_cirs', dict())
                self.obs_cir_number = self.obs_cirs_args.get('number', 0)
                self.obs_step_mode = self.obs_cirs_args.get('obs_step_mode', 0)

                # obs line
                self.obs_lines_args = com_list.get('obs_lines', dict())

                # obs polygons
                self.obs_polygons_args = com_list.get('obs_polygons', dict())
                self.vertexes_list = self.obs_polygons_args.get(
                    'vertexes_list', [])
                self.obs_poly_num = self.obs_polygons_args.get('number', 0)
        else:
            self.__height = kwargs.get('world_height', 10)
            self.__width = kwargs.get('world_width', 10)
            self.step_time = kwargs.get('step_time', 0.1)
            self.world_map = kwargs.get('world_map', None)
            self.xy_reso = kwargs.get('xy_resolution', 1)
            self.yaw_reso = kwargs.get('yaw_resolution', 5)
            self.offset_x = kwargs.get('offset_x', 0)
            self.offset_y = kwargs.get('offset_y', 0)
            self.robot_number = kwargs.get('robot_number', 0)
            self.obs_cir_number = kwargs.get('obs_cir_number', 0)
            self.car_number = kwargs.get('car_number', 0)
            self.robots_args = kwargs.get('robots', dict())
            self.obs_cirs_args = kwargs.get('obs_cirs', dict())
            self.cars_args = kwargs.get('cars', dict())
            self.obs_lines_args = kwargs.get('obs_lines', dict())
            self.obs_polygons_args = kwargs.get('obs_polygons', dict())
            self.vertexes_list = self.obs_polygons_args.get(
                'vertexes_list', [])
            self.obs_poly_num = self.obs_polygons_args.get('number', 0)

        self.plot = plot
        self.components = dict()
        self.init_environment(**kwargs)

        if kwargs.get('teleop_key', False):

            self.key_lv_max = 2
            self.key_ang_max = 2
            self.key_lv = 0
            self.key_ang = 0
            self.key_id = 1
            self.alt_flag = 0

            plt.rcParams['keymap.save'].remove('s')
            plt.rcParams['keymap.quit'].remove('q')

            self.key_vel = np.zeros(2,)

            print('start to keyboard control')
            print('w: forward', 's: backforward', 'a: turn left', 'd: turn right',
                  'q: decrease linear velocity', 'e: increase linear velocity',
                  'z: decrease angular velocity', 'c: increase angular velocity',
                  'alt+num: change current control robot id')

            self.listener = keyboard.Listener(
                on_press=self.on_press, on_release=self.on_release)
            self.listener.start()

        if kwargs.get('mouse', False):
            pass

    def init_environment(self, robot_class=mobile_robot, car_class=car_robot, obs_cir_class=obs_circle, obs_polygon_class=obs_polygon,  **kwargs):

        # world
        px = int(self.__width / self.xy_reso)
        py = int(self.__height / self.xy_reso)

        if self.world_map != None:

            world_map_path = sys.path[0] + '/' + self.world_map
            img = Image.open(world_map_path).convert('L')
            # img = Image.open(world_map_path)
            img = img.resize((px, py), Image.NEAREST)
            # img = img.resize( (px, py), Image.ANTIALIAS)
            # img.thumbnail( (px, py))

            map_matrix = np.array(img)
            map_matrix = 255 - map_matrix
            map_matrix[map_matrix > 255/2] = 255
            map_matrix[map_matrix < 255/2] = 0
            # map_matrix[map_matrix>0] = 255
            # map_matrix[map_matrix==0] = 0

            self.map_matrix = np.fliplr(map_matrix.T)
        else:
            self.map_matrix = None

        self.components['map_matrix'] = self.map_matrix
        self.components['xy_reso'] = self.xy_reso
        self.components['offset'] = np.array([self.offset_x, self.offset_y])

        # self.components['grid_map'] = env_grid(grid_map_matrix=kwargs.get('grid_map_matrix', None))

        self.components['obs_lines'] = env_obs_line(
            **{**self.obs_lines_args, **kwargs})
        self.obs_line_states = self.components['obs_lines'].obs_line_states

        self.components['obs_circles'] = env_obs_cir(obs_cir_class=obs_cir_class, obs_cir_num=self.obs_cir_number,
                                                     step_time=self.step_time, components=self.components, **{**self.obs_cirs_args, **kwargs})
        self.obs_cir_list = self.components['obs_circles'].obs_cir_list

        self.components['obs_polygons'] = env_obs_poly(
            obs_poly_class=obs_polygon_class, vertex_list=self.vertexes_list, obs_poly_num=self.obs_poly_num, **{**self.obs_polygons_args, **kwargs})
        self.obs_poly_list = self.components['obs_polygons'].obs_poly_list

        self.components['robots'] = env_robot(
            robot_class=robot_class, step_time=self.step_time, components=self.components, **{**self.robots_args, **kwargs})
        self.robot_list = self.components['robots'].robot_list

        self.components['cars'] = env_car(
            car_class=car_class, car_num=self.car_number, step_time=self.step_time, **{**self.cars_args, **kwargs})
        self.car_list = self.components['cars'].car_list

        if self.plot:
            self.world_plot = env_plot(self.__width, self.__height, self.components,
                                       offset_x=self.offset_x, offset_y=self.offset_y, **kwargs)

        self.time = 0

        if self.robot_number > 0:
            self.robot = self.components['robots'].robot_list[0]

        if self.car_number > 0:
            self.car = self.components['cars'].car_list[0]

    def collision_check(self,i):
        collision = False
        # for robot in self.components['robots'].robot_list:
        robot = self.components['robots'].robot_list[i]
        if robot.collision_check(self.components):
            collision = True

        for car in self.components['cars'].car_list:
            if car.collision_check(self.components):
                collision = True

        return collision

    def arrive_check(self,i):
        arrive = True
        # for robot in self.components['robots'].robot_list:
        robot = self.components['robots'].robot_list[i]
        if not robot.arrive_flag:
            arrive = False

        for car in self.components['cars'].car_list:
            if not car.arrive_flag:
                arrive = False

        return arrive

    def robot_step(self, vel_list, robot_id=None, **kwargs):
      
        if robot_id == None:

            if not isinstance(vel_list, list):
                self.robot.move_forward(vel_list, **kwargs)
            else:
                for i, robot in enumerate(self.components['robots'].robot_list):
                    robot.move_forward(vel_list[i], **kwargs)
        else:
            self.components['robots'].robot_list[robot_id -
                                                 1].move_forward(vel_list, **kwargs)

        for robot in self.components['robots'].robot_list:
            robot.cal_lidar_range(self.components)

    def car_step(self, vel_list, car_id=None, **kwargs):

        if car_id == None:
            if not isinstance(vel_list, list):
                self.car.move_forward(vel_list, **kwargs)
            else:
                for i, car in enumerate(self.components['cars'].car_list):
                    car.move_forward(vel_list[i], **kwargs)
        else:
            self.components['cars'].car_list[car_id -
                                             1].move_forward(vel_list, **kwargs)

        for car in self.components['cars'].car_list:
            car.cal_lidar_range(self.components)

    def obs_cirs_step(self, vel_list=[], obs_id=None, **kwargs):

        if self.obs_step_mode == 'default':
            if obs_id == None:
                for i, obs_cir in enumerate(self.components['obs_circles'].obs_cir_list):
                    obs_cir.move_forward(vel_list[i], **kwargs)
            else:
                self.components['obs_circles'].obs_cir_list[obs_id -
                                                            1].move_forward(vel_list, **kwargs)

        elif self.obs_step_mode == 'wander':
            # rvo
            self.components['obs_circles'].step_wander(**kwargs)

    def render(self, time=0.05, **kwargs):

        if self.plot:
            self.world_plot.com_cla()
            self.world_plot.draw_dyna_components(**kwargs)
            self.world_plot.pause(time)

        self.time = self.time + time
        
#定义自己的reset 和 step obs
    def obs(self, i,action ):

        #print("robot_state_list:",robot_state_list ,"nei_state_list:",nei_state_list ,"obs_circular_list:",obs_circular_list)
        # robot_xy = robot_state_list[0:2]
        # radius_collision = robot_state_list[4]*np.ones(1,)
        # robot_radius = nei_state_list[4]* np.ones(1,)
        robot_  = self.components['robots'].robot_list
        # curr_vel= robot_[i].vel_omni.squeeze()
        curr_vel= robot_[i].vel_diff.squeeze()
        robot_xy = robot_[i].state[0:2].squeeze()   #(x,y)
        r_other_xy_list =[]
        for j in range(self.robot_number):
            if(j==i):
                j+=1
            else :
                other_xy = robot_[j].state[0:2].squeeze()  
                r_other_xy = robot_xy -other_xy
                r_other_xy_list.append(r_other_xy)

        r_other_xy_list=np.array(r_other_xy_list).reshape(-1)
        # print('vel:',vel,"state:",state,"robot_xy:",robot_xy)
        des_vel= robot_[i].cal_des_vel_diff().squeeze()
        # des_vel =  robot_[i].cal_des_vel_omni().squeeze()
        goal_xy  = robot_[i].goal[0:2].squeeze()
        # position =robot_xy[0:2]
        r_xy = robot_xy - goal_xy
        # dis_goal = abs(np.linalg.norm(position - goal)* np.ones(1,))
       
        obs = np.round(np.concatenate([r_xy]),2) 
        # obs = np.round(obs,2)

        return obs

    def get_des_vel(self,i):

        robot_  = self.components['robots'].robot_list
        des_vel =  robot_[i].cal_des_vel().squeeze()

        return des_vel


    def get_obsshape(self):
        obs= self.reset()
        return np.size(obs[0])


    def reset(self, **kwargs):

        self.components['robots'].robots_reset(self.robot_init_mode, **kwargs)  #初始化
        ts = self.components['robots'].total_states()
        # ts = np.round(ts,2)
        obs_list =[]
        action =[0,0]
        for i in range(self.robot_number):
            robot_  = self.components['robots'].robot_list[i]
            obs_list.append(self.obs(i,action))
            self.dis_reward_list[i] = self.dis_reward(robot_)
        
        return obs_list

    def dis_reward(self,robot_):
        robot_xy = robot_.state[0:2].squeeze()   #(x,y)
        goal_xy  = robot_.goal[0:2].squeeze()
        position =robot_xy[0:2]
        dis_goal = abs(np.linalg.norm(position - goal_xy))
        dis_reward = self.disreward/dis_goal
        return dis_reward

    def step(self, time_step,actions, vel_type='omni', stop=True, **kwargs):

        ts = self.components['robots'].total_states()
        # ts = np.round(ts,2)s
        #cur_vel = np.squeeze(robot.vel_omni)

        # if not isinstance(action, list):  #转换action为list
        #     action = [action]

        self.robot_step(actions, vel_type=vel_type, stop=stop)
        self.obs_cirs_step()
       
        obs_list =[]
        reward_list=[]
        done_list =[]
        info_list =[]
        for i in range(self.robot_number):

            collision_flag=self.collision_check(i)
            arrive_reward_flag =self.arrive_check(i)
            robot_  = self.components['robots'].robot_list[i]
         
            obs_list.append(self.obs(i,actions[i]))
            
            mov_reward = self.mov_reward(obs_list[i],self.dis_reward_list[i],time_step,robot_,collision_flag, arrive_reward_flag, self.reward_parameter, min_exp_time=100)

            done = True if collision_flag else False
            info = True if arrive_reward_flag else False

            info_list.append(info)
            done_list.append(done)
            reward_list.append(mov_reward)

        return obs_list, reward_list,done_list,info_list

    def mov_reward(self, obs_list,dis_reward,time_step,robot_,collision_flag, arrive_reward_flag, reward_parameter=(0.5, 0.1, 0.1, 0.2, 0.2, 1, -20, 15), min_exp_time=100,dis2goal=100):

        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        collision_reward = p7 if collision_flag else 0
        arrive_reward = p8 if arrive_reward_flag else 0

        robot_xyr = robot_.state.squeeze()   #(x,y,radian)
        position = robot_xyr[0:2]

        goal  = robot_.goal[0:2].squeeze()
        dist = abs(np.linalg.norm(position - goal))
        # print('position:',position,"goal:",goal,"dist:",dist)
        dis_goal_reward=  -(dist*dist*dis_reward)
        # dis_goal_reward= np.exp(-dist)
        # others_dis = abs(obs_list[6:])
        # print(
        #     "others_dis",others_dis,"sum(others_dis[0:2])",sum(others_dis[0:2]),"sum(others_dis[3:])",sum(others_dis[3:])
        # )
        
        # if (sum(others_dis[0:2])<0.5 or sum(others_dis[3:])<0.5):
        #     close_reward = -3
        # else:
        #     close_reward= 0
 
        # if not arrive_reward_flag:
        time_reward = -(time_step*p2)
            
        mov_reward =  dis_goal_reward + collision_reward +arrive_reward
        # mov_reward = np.round(mov_reward,2)
        return np.round(mov_reward,2)  
#定义自己的reset 和 step

    def on_press(self, key):

        try:
            if key.char.isdigit() and self.alt_flag:
                if int(key.char) > self.robot_number:
                    print('out of number of robots')
                else:
                    self.key_id = int(key.char)
            if key.char == 'w':
                self.key_lv = self.key_lv_max
            if key.char == 's':
                self.key_lv = - self.key_lv_max
            if key.char == 'a':
                self.key_ang = self.key_ang_max
            if key.char == 'd':
                self.key_ang = -self.key_ang_max

            self.key_vel = np.array([self.key_lv, self.key_ang])

        except AttributeError:

            if key == keyboard.Key.alt:
                self.alt_flag = 1

    def on_release(self, key):

        try:
            if key.char == 'w':
                self.key_lv = 0
            if key.char == 's':
                self.key_lv = 0
            if key.char == 'a':
                self.key_ang = 0
            if key.char == 'd':
                self.key_ang = 0
            if key.char == 'q':
                self.key_lv_max = self.key_lv_max - 0.2
                print('current lv ', self.key_lv_max)
            if key.char == 'e':
                self.key_lv_max = self.key_lv_max + 0.2
                print('current lv ', self.key_lv_max)

            if key.char == 'z':
                self.key_ang_max = self.key_ang_max - 0.2
                print('current ang ', self.key_ang_max)
            if key.char == 'c':
                self.key_ang_max = self.key_ang_max + 0.2
                print('current ang ', self.key_ang_max)

            self.key_vel = np.array([self.key_lv, self.key_ang])

        except AttributeError:
            if key == keyboard.Key.alt:
                self.alt_flag = 0

    def save_fig(self, path, i):
        self.world_plot.save_gif_figure(path, i)

    def save_ani(self, image_path, ani_path, ani_name='animated', **kwargs):
        self.world_plot.create_animate(
            image_path, ani_path, ani_name=ani_name, **kwargs)

    def show(self, **kwargs):
        self.world_plot.draw_dyna_components(**kwargs)
        self.world_plot.show()

    def show_ani(self):
        self.world_plot.show_ani()
