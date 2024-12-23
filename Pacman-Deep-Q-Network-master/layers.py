import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        # 确保输入张量的类型与模块参数类型一致
        # target_dtype = next(self.module.parameters()).dtype
        # x_seq = x_seq.to(target_dtype)
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike_One_Step_nograd(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike_One_Step_nograd, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        # self.action_step = action_step
        # self.mem = 0
        # self.mem = 0

    def forward(self,x,mem):
        mem = mem * self.tau + x
        spike = self.act(mem - self.thresh, self.gama)
        mem = (1 - spike) * mem
        # mem = mem.detach()
        # self.mem =  self.mem.detach()
        return spike.to(torch.float64),mem



# 最常用的baseline
class LIFSpike_One_Step(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike_One_Step, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        # self.action_step = action_step
        # self.mem = 0
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = self.act(self.mem - self.thresh, self.gama)
        self.mem = (1 - spike) * self.mem
        self.mem =  self.mem.detach()
        return spike.to(torch.float64),self.mem

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1).to(torch.float64)

class LIFSpike_One_Step_v2(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike_One_Step_v2, self).__init__()
        # self.thresh = torch.tensor(thresh, dtype=torch.float64)  # 转为张量
        # self.thresh = nn.Parameter(torch.tensor(thresh, dtype=torch.float64))  # 可训练参数 
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.mem = 0

    def forward(self, x,v=1.0):
        self.mem = self.mem * self.tau + x
        # print("self.mem",self.mem.shape)
        # if torch.is_tensor(self.thresh):
        #     print("self.thresh",self.thresh.shape)
        spike = self.act(self.mem - v, self.gama)
        # spike = self.act(self.mem - self.thresh, self.gama)
        self.mem = (1 - spike) * self.mem
        self.mem = self.mem.detach()
        return spike.to(torch.float64), self.mem

    def update_th(self, vth):
        self.thresh = vth
        # self.thresh = self.thresh.detach()


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        # 确保输入张量的类型与模块参数类型一致
        # target_dtype = next(self.layer.parameters()).dtype
        # x = x.to(target_dtype)
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


# LIFSpike = LIF


class TimeStepRouter(nn.Module):
    def __init__(self):
        super(TimeStepRouter,self).__init__()
        self.weight_predictor = None
        self.input_dim = None
    
    def forward(self,x): #输入x维度为[B,T,C,H,W]
        if len(x.shape) < 5:
            B,T,D= x.shape
            x = x.view(B,T,D)
            self.input_dim = D
        else:
            B,T,C,H,W= x.shape
            x = x.view(B,T,C*H*W)
            self.input_dim = C*H*W
        if self.weight_predictor is None:
            self.weight_predictor = nn.Linear(self.input_dim,1).to(x.device)
        weights =  self.weight_predictor(x).squeeze(-1)
        return weights


class TimeMoD(nn.Module):
    def __init__(self,capacity,layer,T_dim):
        super(TimeMoD,self).__init__()
        self.T_dim = T_dim
        self.time_router = TimeStepRouter()
        self.layer = layer
        self.capacity = capacity
        self.training_step = 0
    
    def forward(self,x):
        if len(x.shape) < 5:
            B,T,D= x.shape
        else:
            B,T,C,H,W= x.shape
        weights =  self.time_router(x)
        # print(weights.shape)
        if self.time_router.training:
            self.training_step += 1 if self.training_step <1000 else 999
            self.capacity = 0.35 + ((1 - 0.35) * (1. / self.training_step)) 
        
        # 定义选择几个timestep(此处capacity可能得调整一下，因为T不会像seqlen那么大)
        k = max(1,int(self.capacity * self.T_dim))
        # k=2
        # print(k)
        top_k_values,_ = torch.topk(weights,k, dim=1,sorted=True)
        # print(weights[0][0],weights[0][1],weights[0][2],weights[0][3])
        threshold = top_k_values[:,-1]

        selected_mask = weights >= threshold.unsqueeze(-1) if k > 1 else weights >= threshold.unsqueeze(-1)
        # with open('selected_mask.txt', 'a') as f:
        #     print(selected_mask.shape, file=f)
        # print((x*selected_mask).shape)

        # # 统计 True 的数量
        # num_true = selected_mask.sum().item()
        # # 统计 False 的数量
        # num_false = (~selected_mask).sum().item()
        # print(f"True 的数量: {num_true}")
        # print(f"False 的数量: {num_false}")

        # 记录被处理的时间步
        processed_timesteps = torch.zeros_like(x)
        
        """
            不循环BATCH
        """
         # 扩展 selected_mask 以匹配 x 的形状，并确保维度对齐
        mask_expanded  = selected_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1, 1]
        mask_expanded  = mask_expanded.expand(-1, -1, C, H, W)  # 广播到 [B, T, C, H, W]
        selected_timesteps = x[mask_expanded].view(B,-1,C,H,W) 

        with open('selected_timesteps.txt', 'a') as f:
            print(selected_timesteps.shape, file=f)

        # 检查是否有被选中的时间步
        if selected_timesteps.size(1) > 0:
            # print("进来了")
            if len(x.shape) <5 :
                processed_timesteps[mask_expanded] = self.layer(selected_timesteps).view(-1)
            else:
                processed_timesteps[mask_expanded] = self.layer(selected_timesteps).view(-1)
        else:
            print("没进来")
        if len(x.shape) < 5:
            output = processed_timesteps + (x * (~selected_mask).unsqueeze(-1).to(x.dtype)) 
        else:
            output = processed_timesteps + (x * (~selected_mask).view(B,T,1,1,1).to(x.dtype))  



        # for i in range(B):
        #     current_selected_mask = selected_mask[i]
        #     selected_timesteps = x[i][current_selected_mask]

        #     # with open('selected_timesteps.txt', 'a') as f:
        #     #     print(selected_timesteps.shape, file=f)

        #     # 检查是否有被选中的时间步
        #     if selected_timesteps.size(0) > 0:
        #         # print("进来了")
        #         if len(x.shape) <5 :
        #             # processed_timesteps[i][selected_mask[i]] = self.layer(selected_timesteps.unsqueeze(0))[0] * weights[i][selected_mask[i]].unsqueeze(-1)
        #             processed_timesteps[i][selected_mask[i]] = self.layer(selected_timesteps.unsqueeze(0))[0]
        #         else:
        #             # processed_timesteps[i][selected_mask[i]] = self.layer(selected_timesteps.unsqueeze(0))[0] * weights[i][selected_mask[i]].view(selc_steps_nums,1,1,1)
        #             processed_timesteps[i][selected_mask[i]] = self.layer(selected_timesteps.unsqueeze(0))[0]
        #     else:
        #         print("没进来")
        # if len(x.shape) < 5:
        #     output = processed_timesteps + (x * (~selected_mask).unsqueeze(-1).to(x.dtype)) 
        # else:
        #     output = processed_timesteps + (x * (~selected_mask).view(B,T,1,1,1).to(x.dtype)) 

        return output
                














