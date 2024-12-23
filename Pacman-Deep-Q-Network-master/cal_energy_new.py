# -- coding: utf-8 --
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from thop import profile
import config

def ann_energy_cal(model,input_tensor,batchsize):
    # flops, params = profile(model,(input_tensor,))
    # energy = 4.6 * flops / 1000000000.0
    # energy = 0.0032227968 * batchsize# smallGrid
    energy = 0.026921702399999997 * batchsize #mediumClassic
    config.ann_single_action_consumption = energy
    config.ann_total_action_consumption += energy
    # print(energy)
    return energy
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.5f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    # print('energy: %.5f mJ' % (energy))

# 计算发放率的辅助函数
def compute_firing_rate(spike_matrix):
    num_spikes = torch.sum(spike_matrix)  # 统计发放的次数（矩阵中1的数量）
    total_elements = spike_matrix.numel()  # 总的元素个数
    firing_rate = (num_spikes / total_elements).item()  # 发放率
    return firing_rate
    # return spike_matrix.flatten(-1).mean()

# Hook 函数
def hook_fn(module, input, output,count,batch_size):  
    # print("进来了")
    firing_rate = compute_firing_rate(output)

    # MediumClassic
    if count == 0:   
        energy = firing_rate * 1.12 /1000.0 * 0.9 * batch_size
        config.snn_single_action_consumption += energy
        config.snn_total_action_consumption += energy
    elif count ==1:
        energy = firing_rate * 4.46 /1000.0 * 0.9* batch_size
        config.snn_single_action_consumption += energy
        config.snn_total_action_consumption += energy
    elif count ==2:
        energy =  firing_rate * 287 /1000000.0 * 0.9* batch_size
        config.snn_single_action_consumption += energy
        config.snn_total_action_consumption += energy

    ## smallGrid
    # if count == 0:   
    #     energy = firing_rate * 132.1 /1000000.0 * 0.9 * batch_size
    #     config.snn_single_action_consumption += energy
    #     config.snn_total_action_consumption += energy
    # elif count ==1:
    #     energy = firing_rate * 524.8 /1000000.0 * 0.9* batch_size
    #     config.snn_single_action_consumption += energy
    #     config.snn_total_action_consumption += energy
    # elif count ==2:
    #     energy =  firing_rate * 46.1 /1000000.0 * 0.9* batch_size
    #     config.snn_single_action_consumption += energy
    #     config.snn_total_action_consumption += energy


    # print(config.snn_single_action_consumption)
# def snn_energy_cal():

# 示例网络
# class DQN(nn.Module):
#     def __init__(self, num_inputs=6, num_actions=4):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
#         self.fc3 = nn.Linear(64 * 8 * 17, 512)
#         # self.fc3 = nn.Linear(1024, 512)
#         self.fc4 = nn.Linear(512, num_actions)
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))  
#         x = F.relu(self.fc3(x.view(x.size(0), -1)))
#         return self.fc4(x)

# 测试函数
# model = DQN()
# input_tensor = torch.randn(1, 6, 11, 20) 
# energy = ann_energy_cal(model,input_tensor)
# print(energy)

# import torchvision
# from ptflops import get_model_complexity_info
# flops, params = get_model_complexity_info(model, (6, 7, 7), as_strings=True, print_per_layer_stat=True)
# print('flops: ', flops, 'params: ', params)
