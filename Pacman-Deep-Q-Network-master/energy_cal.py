import torch
import torch.nn as nn

def count_mac_and_ac(model, input_tensor):
    mac_count = 0
    ac_count = 0

    def conv2d_mac_ac(layer, input_tensor):
        # 获取输入输出形状和卷积核参数
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = layer.weight.shape
        
        # 输出尺寸计算
        out_height = (in_height - kernel_height) // layer.stride[0] + 1
        out_width = (in_width - kernel_width) // layer.stride[1] + 1
        
        # 计算 MAC 和 AC
        macs = out_height * out_width * out_channels * (kernel_height * kernel_width * in_channels)
        acs = out_height * out_width * out_channels * (kernel_height * kernel_width * in_channels)
        
        return macs, acs
    
    def linear_mac_ac(layer, input_tensor):
        # 获取输入输出形状
        in_features = layer.in_features
        out_features = layer.out_features
        
        # 计算 MAC 和 AC
        macs = in_features * out_features
        acs = in_features * out_features
        
        return macs, acs
    count = 0
    # 遍历模型的每一层
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            # 计算卷积层的 MAC 和 AC
            mac, ac = conv2d_mac_ac(layer, input_tensor)
            mac_count += mac
            ac_count += ac
            # 通过卷积层后的输出
            input_tensor = layer(input_tensor)
        elif isinstance(layer, nn.Linear):
           
            if count ==0:
                # print(input_tensor.shape)
                count+=1
                # 计算全连接层的 MAC 和 AC
                mac, ac = linear_mac_ac(layer, input_tensor.view(input_tensor.size(0),-1))
                mac_count += mac
                ac_count += ac
                # 通过全连接层后的输出
                input_tensor = layer(input_tensor.view(input_tensor.size(0),-1))
            else:
                mac, ac = linear_mac_ac(layer, input_tensor)
                mac_count += mac
                ac_count += ac
                input_tensor = layer(input_tensor)
    
    # 计算并输出 MAC 和 AC 的 G 单位
    mac_in_G = mac_count / 1e9
    ac_in_G = ac_count / 1e9
    print(f"MAC count: {mac_in_G:.6f} G")
    print(f"AC count: {ac_in_G:.6f} G")
    return mac_count, ac_count

# 示例网络
class DQN(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc3 = nn.Linear(64 * 8 * 17, 512)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        return self.fc4(x)

# 测试函数
model = DQN()
input_tensor = torch.randn(32, 6, 11, 20)  # batch_size=1, num_inputs=6, height=8, width=18
mac_count, ac_count = count_mac_and_ac(model, input_tensor)

print(f"MAC count: {mac_count}")
print(f"AC count: {ac_count}")
