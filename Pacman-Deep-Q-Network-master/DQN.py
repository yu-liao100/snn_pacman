import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import torchvision
from cal_energy_new import hook_fn
# architecture used for layout smallGrid
""" Deep Q Network """

class DQN(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(1024, num_actions)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x

class DQN_SNN_SingleTimeSteps(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN_SNN_SingleTimeSteps, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.spike_1 = LIFSpike_One_Step()
        self.spike_2 = LIFSpike_One_Step()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)
        self.spike_3 = LIFSpike_One_Step()
        # self.fc3 = nn.Linear(64, 512) r=1
        self.fc3 = nn.Linear(256, 512) # r=2
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        # self.fc3 = nn.Linear(64 * 19 * 37, 512)
        self.fc4 = nn.Linear(512, num_actions)
        self.hook1 = None
        self.hook2 = None
        self.hook3 = None
    def forward(self, x):
        batch_size = x.shape[0]
        # print(batch_size)
        if config.hook_enabled:
            # 为每一层注册 hook
            self.hook1 = self.spike_1.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 0,batch_size = batch_size))
            self.hook2 = self.spike_2.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 1,batch_size = batch_size))
            self.hook3 = self.spike_3.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 2,batch_size = batch_size))
        else:
            if self.hook1 is not None:
                self.hook1.remove()
            if self.hook2 is not None:
                self.hook2.remove()
            if self.hook3 is not None:
                self.hook3.remove()
        # print("x.shape", x.shape)
        x = self.conv1(x)
        x,mem1 = self.spike_1(x)
        x = self.conv2(x)
        x,mem2 = self.spike_2(x)
        x = self.fc3(x.view(x.size(0),-1))
        x,mem3 = self.spike_3(x)
        x = self.fc4(x)
        # print(x.shape)
        return x,mem1,mem2,mem3



# 标准DQN
# class DQN(nn.Module):
#     def __init__(self, num_inputs=6, num_actions=4):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
#         # self.relu = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(1024, 512)
#         # self.fc3 = nn.Linear(1024, num_actions)
#         # self.fc3 = nn.Linear(64 * 8 * 17, 512)
#         self.fc4 = nn.Linear(512, num_actions)
        
#     def forward(self, x):
#         batch_Size = x.shape[0]
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.fc3(x.view(x.size(0), -1)))
#         x = self.fc4(x)
#         return x
    
class DQN_change(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN_change, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.hidden_space = 512
        self.fc3 = nn.Linear(64, 512)
        # self.fc3 = nn.Linear(1024, num_actions)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        # self.gru = nn.GRU(512, num_actions, batch_first=True)
        self.lstm = nn.LSTM(self.hidden_space,self.hidden_space, batch_first=True)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self,x,h,c):
        h = h.detach()
        c = c.detach()
        batch_Size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = x.unsqueeze(1)
        x,(new_h,new_c) = self.lstm(x,(h,c))
        x = self.fc4(x)
        # hidden = self.init_hidden(batch_Size)
        # x = x.reshape(batch_Size, 1, 512)
        # print(x.shape)
        # print(h.shape)
        # x,new_h = self.gru(x,h)
        # print(x.shape)
        x= x.squeeze(1)
        return x,new_h,new_c

    def init_hidden(self, batch_size):
        # initialize hidden state to 0
        return torch.zeros(1, batch_size, 4, device= "cuda:4", dtype=torch.float64)

        # return torch.zeros(1, batch_size, self.hidden_space, device= "cuda:4", dtype=torch.float64),torch.zeros(1, batch_size, self.hidden_space, device= "cuda:4", dtype=torch.float64)



class DQN_SNN(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN_SNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.conv1_s = tdLayer(self.conv1)
        self.conv2_s = tdLayer(self.conv2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.relu = nn.ReLU(inplace=True)
        self.spike = LIFSpike()
        self.fc3 = nn.Linear(64, 512)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        # 添加T维度sssss
        T = 1
        x.unsqueeze_(1)
        x = x.repeat(1, T, 1, 1, 1)
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.fc3(x.view(x.size(0), x.size(1),-1))
        x = self.spike(x)
        x = self.fc4(x)
        x = torch.mean(x, dim=1)
        return x


# 标准DQN_SNN
# class DQN_SNN(nn.Module):
#     def __init__(self, num_inputs=6, num_actions=4):
#         super(DQN_SNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
#         self.conv1_s = tdLayer(self.conv1)
#         self.conv2_s = tdLayer(self.conv2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#         self.relu = nn.ReLU(inplace=True)
#         self.spike = LIFSpike()
#         self.fc3 = nn.Linear(1024, 512)
#         # self.fc3 = nn.Linear(64 * 8 * 17, 512)
#         self.fc4 = nn.Linear(512, num_actions)
        
#     def forward(self, x):
#         # 添加T维度sssss
#         T = 1
#         x.unsqueeze_(1)
#         x = x.repeat(1, T, 1, 1, 1)
#         x = self.conv1_s(x)
#         x = self.spike(x)
#         x = self.conv2_s(x)
#         x = self.spike(x)
#         x = self.fc3(x.view(x.size(0), x.size(1),-1))
#         x = self.spike(x)
#         x = self.fc4(x)
#         x = torch.mean(x, dim=1)
#         return x

# class DQN_SNN_SingleTimeSteps(nn.Module):
#     def __init__(self, num_inputs=6, num_actions=4):
#         super(DQN_SNN_SingleTimeSteps, self).__init__()
#         self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
#         self.spike_1 = LIFSpike_One_Step()
#         self.spike_2 = LIFSpike_One_Step()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#         self.relu = nn.ReLU(inplace=True)
#         self.spike_3 = LIFSpike_One_Step()
#         self.fc3 = nn.Linear(1024, 512)
#         # self.fc3 = nn.Linear(64 * 8 * 17, 512)
#         # self.fc3 = nn.Linear(64 * 19 * 37, 512)
#         self.fc4 = nn.Linear(512, num_actions)
#         self.hook1 = None
#         self.hook2 = None
#         self.hook3 = None
#     def forward(self, x):
#         batch_size = x.shape[0]
#         # print(batch_size)
#         if config.hook_enabled:
#             # 为每一层注册 hook
#             self.hook1 = self.spike_1.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 0,batch_size = batch_size))
#             self.hook2 = self.spike_2.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 1,batch_size = batch_size))
#             self.hook3 = self.spike_3.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 2,batch_size = batch_size))
#         else:
#             if self.hook1 is not None:
#                 self.hook1.remove()
#             if self.hook2 is not None:
#                 self.hook2.remove()
#             if self.hook3 is not None:
#                 self.hook3.remove()
#         # print("x.shape", x.shape)
#         x = self.conv1(x)
#         x,mem1 = self.spike_1(x)
#         x = self.conv2(x)
#         x,mem2 = self.spike_2(x)
#         x = self.fc3(x.view(x.size(0),-1))
#         x,mem3 = self.spike_3(x)
#         x = self.fc4(x)
#         # print(x.shape)
#         return x,mem1,mem2,mem3

class DQN_SNN_SingleTimeSteps_change(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN_SNN_SingleTimeSteps_change, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.spike_1 = LIFSpike_One_Step()
        self.spike_2 = LIFSpike_One_Step()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)
        self.spike_3 = LIFSpike_One_Step()
        self.fc3 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        self.fc4 = nn.Linear(512, num_actions)
        self.conv_s = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(64 * 4 * 4, 32 * 5 * 5)

    def forward(self, x,new_mem1):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x,mem1 = self.spike_1(x)
        mem1 = mem1 + new_mem1
        x = self.conv2(x)
        x,mem2 = self.spike_2(x)
        output = mem2.view(batch_size,-1)
        output = self.linear(output)
        # output = nn.functional.interpolate(output, size=(9, 18), mode='bilinear', align_corners=False)
        output = output.view(batch_size,32,5,5)
        x = self.fc3(x.view(x.size(0),-1))
        x,mem3 = self.spike_3(x)
        x = self.fc4(x)
        return x,mem1,mem2,mem3,output

class DQN_SNN_SingleTimeSteps_changev2(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN_SNN_SingleTimeSteps_changev2, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.spike_1 = LIFSpike_One_Step_nograd()
        self.spike_2 = LIFSpike_One_Step_nograd()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)
        self.spike_3 = LIFSpike_One_Step_nograd()
        self.fc3 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        # self.fc3 = nn.Linear(64 * 19 * 37, 512)
        self.fc4 = nn.Linear(512, num_actions)
        self.hook1 = None
        self.hook2 = None
        self.hook3 = None
    def forward(self, x,mem1,mem2,mem3):
        batch_size = x.shape[0]
        # print(batch_size)
        if config.hook_enabled:
            # 为每一层注册 hook
            self.hook1 = self.spike_1.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 0,batch_size = batch_size))
            self.hook2 = self.spike_2.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 1,batch_size = batch_size))
            self.hook3 = self.spike_3.register_forward_hook(lambda module, input, output: hook_fn(module, input, output,count = 2,batch_size = batch_size))
        else:
            if self.hook1 is not None:
                self.hook1.remove()
            if self.hook2 is not None:
                self.hook2.remove()
            if self.hook3 is not None:
                self.hook3.remove()
        # print("x.shape", x.shape)
        x = self.conv1(x)
        x,mem1 = self.spike_1(x,mem1)
        x = self.conv2(x)
        x,mem2 = self.spike_2(x,mem2)
        x = self.fc3(x.view(x.size(0),-1))
        x,mem3 = self.spike_3(x,mem3)
        x = self.fc4(x)
        # print(x.shape)
        return x,mem1,mem2,mem3


class DRQN(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.spike_1 = LIFSpike_One_Step()
        self.spike_2 = LIFSpike_One_Step()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.relu = nn.ReLU(inplace=True)
        self.spike_3 = LIFSpike_One_Step()
        self.lstm = nn.LSTM(self.hidden_space,self.hidden_space, batch_first=True)
        self.fc3 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(64 * 8 * 17, 512)
        # self.fc4 = nn.Linear(512, num_actions)
        self.gru = nn.GRU(512, num_actions, batch_first=True) # input shape (batch, seq, feature)
        self.linear = nn.Linear(512,4)
        self.linear1 = nn.Linear(4,4)

    def forward(self, x,hidden = None):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x,mem1 = self.spike_1(x)    
        # print("sss",mem1.shape)   
        x = self.conv2(x)
        x,mem2 = self.spike_2(x)
        x = self.fc3(x.view(x.size(0),-1))
        x,mem3 = self.spike_3(x)
        if hidden is None:
            hidden  = torch.zeros(1, batch_size,4, device="cuda:0", dtype=torch.float64)
        # mem3_temp = self.linear(mem3).unsqueeze(0)
        mem3_temp = mem3.unsqueeze(0).view(1,batch_size,-1,4).mean(dim=2)
        hidden = mem3_temp + hidden
        x_temp = x.reshape(batch_size,1,512)
        x,hidden = self.gru(x_temp,mem3_temp)
        x= x.squeeze(1)
        x = self.linear1(x)
        # x = self.fc4(x)
        return x,mem1,mem2,mem3,hidden




# architecture used for layout mediumClassic
# class DQN(nn.Module):
#     def __init__(self, num_inputs=6, num_actions=4):
#         super(DQN, self).__init__()
        
#         self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.fc3 = nn.Linear(7 * 16 * 64, 512)
#         self.fc4 = nn.Linear(512, num_actions)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.fc3(x.view(x.size(0), -1)))
#         return self.fc4(x)
