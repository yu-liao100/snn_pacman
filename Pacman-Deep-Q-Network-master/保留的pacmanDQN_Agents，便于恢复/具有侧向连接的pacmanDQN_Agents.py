# import pacman game 
from pacman import Directions
from pacmanUtils import *
from game import Agent
import game

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DQN import *

#import other libraries
import os
import util
import random
import numpy as np
import time
import sys

from time import gmtime, strftime
from collections import deque
from cal_energy_new import ann_energy_cal
# model parameters
model_trained = False

GAMMA = 0.95  # discount factor
# LR = 0.01     # learning rate
LR =  0.0002

batch_size = 32            # memory replay batch size
memory_size = 50000		   # memory replay size
start_training = 300 	   # start training at this episode
TARGET_REPLACE_ITER = 100  # update network step
# TARGET_REPLACE_ITER = 10

epsilon_final = 0.1   # epsilon final
epsilon_step = 7500
# epsilon_step = 20000
# global_var = 0 # 多少个时间步后膜电位重新置为0
# global_var_2 = 0
# global_var_3 = 0
    
class PacmanDQN(PacmanUtils):
    def __init__(self, args):        
		
        print("Started Pacman DQN algorithm")
        if(model_trained == True):
            print("Model has been trained")
        else:
            print("Training model")

        # pytorch parameters
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
		
		# init model
        if(model_trained == True):
            self.policy_net = torch.load('train_medium_snn_model/pacman_policy_net_70000.pt').to(self.device)
            self.target_net = torch.load('train_medium_snn_model/pacman_target_net_70000.pt').to(self.device)
        else:
            # print("进来了")
            # return
            # self.policy_net = DQN().to(self.device)
            # self.target_net = DQN().to(self.device)
            # self.policy_net = DQN_SNN_SingleTimeSteps().to(self.device)
            # self.target_net = DQN_SNN_SingleTimeSteps().to(self.device)
            self.policy_net = DQN_SNN_SingleTimeSteps_change().to(self.device)
            self.target_net = DQN_SNN_SingleTimeSteps_change().to(self.device)
            # self.policy_net = DQN_Resnet().to(self.device)
            # self.target_net = DQN_Resnet().to(self.device)
        self.policy_net.double()        
        self.target_net.double()        

        # init optim
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        
        # init counters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0
        # 100次求reward平均更好看出变化
        self.sum_reward = 0
        self.mean_reward = []
        self.sum_score = 0
        self.mean_score = []                
        self.name = "pacman"

        if(model_trained == False):
            self.epsilon = 0.5     # epsilon init value
            # self.epsilon = 0.0 
    
        else:
            self.epsilon = 0.0     # epsilon init value

        # init parameters
        self.width = args['width']
        self.height = args['height']
        self.num_training = args['numTraining']
        self.tau = 0.005
        # statistics
        self.episode_number = 0
        self.last_score = 0
        self.last_reward = 0.
        self.last_mem1 = 0 # 保存上一时刻的膜电位
        self.last_mem2 = 0
        self.last_mem3 = 0
        self.last_output = 0
        self.new_output = 0 
        self.hidden = None
        self.last_move = None
		# memory replay and score databases
        self.replay_mem = deque()
        self.steps = 0
		# Q(s, a)
        self.Q_global = []  
		
		# open file to store information
        # self.f= open("smallGrid_SNN.txt","a")
        # self.f= open("mediumClassic_SNN.txt","a")
        self.train_step = -1
        self.move_step = 0
        self.temp_Q_found = None
    def write_to_file(self, data):
        with open(f"{config.save_filename}", "a") as f:
            f.write(data)

    def getMove(self, state): # epsilon greedy
        # if isinstance(self.policy_net, DQN):
        #     if config.period ==1:
        #         config.period =0
        #         return self.last_move
        #     else:
        #         config.period +=1
        random_value = np.random.rand() 
        if random_value > self.epsilon: # exploit 
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)
            # change singletimesteps
            # self.Q_found,v1,v2 = self.policy_net(temp_current_state,v1,v2)
            # 判断是否是 DQN_SNN_SingleTimeSteps 类的实例
            # if isinstance(self.policy_net, DQN):
            #     ann_energy_cal(self.policy_net,temp_current_state,temp_current_state.shape[0])
            # elif isinstance(self.policy_net, DQN_SNN_SingleTimeSteps):
            #     # print("进来了")
            #     config.hook_enabled = True
            # print("temp_current_state.shape: " , temp_current_state.shape)
            if isinstance(self.policy_net, DQN):
                self.Q_found = self.policy_net(temp_current_state)
            else:
                # if config.is_gameover:
                #     self.last_mem1 = 0 
                #     self.last_mem2 = 0
                #     self.last_mem3 = 0
                #     config.is_gameover = False
                self.policy_net.spike_1.mem = self.last_mem1
                self.policy_net.spike_2.mem = self.last_mem2
                self.policy_net.spike_3.mem = self.last_mem3
                self.Q_found,self.last_mem1,self.last_mem2,self.last_mem3,self.new_output = self.policy_net(temp_current_state,self.last_output)

            self.Q_found =  self.Q_found.detach().cpu()
            self.Q_found = self.Q_found.numpy()[0]
			# store max Qsa
            self.Q_global.append(max(self.Q_found))
			# get best_action - value between 0 and 3
            best_action = np.argwhere(self.Q_found == np.amax(self.Q_found))          
			
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = self.get_direction(best_action[random_value][0])
            else:
                move = self.get_direction(best_action[0][0])
        else: # explore
            # config.epsilon_mem = True
            random_value = np.random.randint(0, 4)  # random value between 0 and 3
            move = self.get_direction(random_value)

        # save last_action
        self.last_action = self.get_value(move)
        self.last_move = move
        self.steps +=1
        return move
    
    def observation_step(self, state):
        if self.last_action is not None:
            # get state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)
            # get reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # ate a ghost 
            elif reward > 0:
                self.last_reward = 10.    # ate food a
            elif reward < -10:
                self.last_reward = -500.  # was eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # didn't eat

            if(self.terminal and self.won):
                self.last_reward = 100.
                self.win_counter += 1
            self.episode_reward += self.last_reward

            # store transition 
            if isinstance(self.policy_net, DQN):
                transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
            else:
                if config.epsilon_mem:
                    transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal,0,0,0)
                    config.epsilon_mem = False
                else:
                    transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal,self.last_mem1,self.last_mem2,self.last_mem3,self.last_output)
                    self.last_output = self.new_output
            self.replay_mem.append(transition)
            if len(self.replay_mem) > memory_size:
                self.replay_mem.popleft()
            self.train()
            # if config.is_gameover:
            #     self.last_mem1 = 0 
            #     self.last_mem2 = 0
            #     self.last_mem3 = 0
            #     for _ in range(self.steps):
            #         self.train()
            #     config.is_gameover = False 
            #     self.steps = 0
        # next
        self.local_cnt += 1
        self.frame += 1
		
		# update epsilon
        if(model_trained == False):
            self.epsilon = max(epsilon_final, 1.00 - float(self.episode_number) / float(epsilon_step))
            # self.epsilon = 0

    def final(self, state):
        # print(self.frame)
        # Next
        self.episode_reward += self.last_reward

        # do observation
        self.terminal = True
        self.observation_step(state)
                

        # 每100个episode取一次平均reward
        self.sum_reward += self.episode_reward
        self.sum_score += self.current_score
		# print episode information
        print("Episode no = " + str(self.episode_number) + "; won: " + str(self.won) , "; iter_time_steps = " + str(self.frame)
		, "; Q(s,a) = " , str(max(self.Q_global, default=float('nan'))) , "; score = " + str(self.current_score), "; reward = " + str(self.episode_reward) ,
         "; snn_single_energy = " + str(config.snn_single_action_consumption),  "; snn_total_energy = " + str(config.snn_total_action_consumption),
        "; ann_single_energy = " + str(config.ann_single_action_consumption),  "; ann_total_energy = " + str(config.ann_total_action_consumption),
           "; and epsilon = " + str(self.epsilon))
		
		# copy episode information to file
        self.counter += 1

        if(self.counter % 100 == 0):
            mean_reward = self.sum_reward / 100.0
            mean_score = self.sum_score / 100.0
            self.mean_reward.append(mean_reward) 
            self.mean_score.append(mean_score)
            self.sum_reward = 0
            self.sum_score = 0
            self.write_to_file("Episode no = " + str(self.episode_number) + "; won: " + str(self.won)  + "; iter_time_steps = " + str(self.frame)
            + "; Q(s,a) = " + str(max(self.Q_global, default=float('nan'))) + "; score = " + str(self.current_score) + "; mean_score/100_episodes = " + str(mean_score) + "; reward = " +  str(self.episode_reward) 
            + "; mean_reward/100_episodes = " + str(mean_reward) + 
             "; snn_single_energy = " + str(config.snn_single_action_consumption)+ "; snn_total_energy = " + str(config.snn_total_action_consumption)+
            "; ann_single_energy = " + str(config.ann_single_action_consumption)+  "; ann_total_energy = " + str(config.ann_total_action_consumption)+
            "; and epsilon = " + str(self.epsilon) + ", win percentage = " + str(self.win_counter / 100.0) + ", " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + "\n")
            # self.f.write()
            self.win_counter = 0

        # if(self.counter % 10000 == 0):
        #     # save model
        #     torch.save(self.policy_net, f'train_mediumClassic_ANN_retest/pacman_policy_net_{self.counter}.pt')
        #     torch.save(self.target_net, f'train_mediumClassic_ANN_retest/pacman_target_net_{self.counter}.pt')

        if(self.episode_number % TARGET_REPLACE_ITER == 0):
            print("UPDATING target network")
            if config.is_soft_update:
                self.soft_update(self.policy_net,self.target_net)
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    
    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    def intTotensor(self,tuple):
        # 获取参考 tensor 的形状
        tensor_shape = next((elem.shape for elem in tuple if isinstance(elem, torch.Tensor)), None)
        if tensor_shape is None:
            return 0
        # 如果找到了 tensor 元素，就转换 int 为与 tensor 相同形状的 tensor
        tuple = [
            elem if isinstance(elem, torch.Tensor) else torch.zeros(tensor_shape, dtype=torch.float64,device=self.device)
            for elem in tuple
        ]
        # 将所有元素堆叠并 squeeze(1)
        tuple = torch.stack(tuple).squeeze(1)
        return tuple



    def train(self):
        if (self.local_cnt > start_training):
            batch = random.sample(self.replay_mem, batch_size)
            if isinstance(self.policy_net, DQN):
                batch_s, batch_r, batch_a, batch_n, batch_t  = zip(*batch)
            else:
                batch_s, batch_r, batch_a, batch_n, batch_t, batch_mem1, batch_mem2, batch_mem3,batch_output = zip(*batch)
            # convert from numpy to pytorch 

            batch_s = torch.from_numpy(np.stack(batch_s))
            batch_s = batch_s.to(self.device)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
            
            batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
            batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(self.device)
            if isinstance(self.policy_net, DQN):
                pass
            else:
                batch_mem1 = self.intTotensor(batch_mem1)
                batch_mem2 = self.intTotensor(batch_mem2)
                batch_mem3 = self.intTotensor(batch_mem3)
                batch_output = self.intTotensor(batch_output)
                self.policy_net.spike_1.mem = batch_mem1
                self.policy_net.spike_2.mem = batch_mem2
                self.policy_net.spike_3.mem = batch_mem3
            # get Q(s, a)
            config.hook_enabled = False
            if isinstance(self.policy_net, DQN):
                state_action_values = self.policy_net(batch_s).gather(1, batch_a)
            else:
                state_action_values = self.policy_net(batch_s,batch_output)[0].gather(1, batch_a)
           
            # state_action_values = torch.mean(self.policy_net(batch_s), dim=1).gather(1, batch_a)

            # get V(s')
            config.hook_enabled = False
            # self.target_net
            if isinstance(self.policy_net, DQN):
                next_state_values = self.target_net(batch_n)
            else:
                self.target_net.spike_1.mem = batch_mem1
                self.target_net.spike_2.mem = batch_mem2
                self.target_net.spike_3.mem = batch_mem3
                next_state_values = self.target_net(batch_n,batch_output)[0]
            # print("next_state_values.shape", next_state_values.shape)
            # Compute the expected Q values                        
            next_state_values = next_state_values.detach().max(1)[0]
            next_state_values = next_state_values.unsqueeze(1)
            
            expected_state_action_values = (next_state_values * GAMMA) + batch_r
            
			# calculate loss
            loss_function = torch.nn.SmoothL1Loss()                                                                                                                                                

            self.loss = loss_function(state_action_values, expected_state_action_values)
            
			# optimize model - update weights
            self.optim.zero_grad()
            self.loss.backward(retain_graph=True)
            self.optim.step()            
