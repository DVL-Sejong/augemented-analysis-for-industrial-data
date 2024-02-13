import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
from .gst_algo import *
import easydict

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2: # if supports_len == 1:
            Lk=Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None


    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, config, kernel_size=3):
        """
        :param c_in: in channels
        :param c_out: out channels
        :param kernel_size:
        TCN convolution
            input: (B, c_in, V, T)
            output:(B, c_out, V, T)
        """
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1,1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)

        self.norm = nn.LayerNorm([config.V, c_out])
    def forward(self, x, t, A_hat):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        h = self.tcn1(x)

        h += self.t_conv(t[:, :, None, None])

        h = self.tcn2(h)

        h = self.norm(h.transpose(1,3)).transpose(1,3) # (B, c_out, V, T)

        h = h.transpose(2,3) #(B, c_out, V, T)
        h = self.spatial(h, A_hat).transpose(2,3) # (B, c_out, V, T)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        return self.res(x, t, supports)

class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        _ = t
        _ = supports
        return self.conv(x)

class  UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        super().__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)

class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, supports):
        _ = t
        _ = supports
        return  self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, c_in, config):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in, config, kernel_size=3)

    def forward(self, x, t, supports):
        x = self.res1(x, t, supports)

        x = self.res2(x, t, supports)

        return x

class GSTNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        d_h = self.d_h = config.d_h
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = self.T_p + self.T_h
        self.mem_num = 2* T  # 추가

        self.We1 = nn.Parameter(torch.randn(self.config.V, self.mem_num), requires_grad = True).to(self.config.device) # 추가
        self.We2 = nn.Parameter(torch.randn(self.config.V, self.mem_num), requires_grad = True).to(self.config.device) # 추가
        self.Memory = nn.Parameter(torch.randn(self.mem_num, self.mem_num), requires_grad = True).to(self.config.device) # 추가
        self.Wq = nn.Parameter(torch.randn(self.mem_num, self.mem_num), requires_grad=True).to(self.config.device) # 수정 예정 
        self.F = self.config.F

        self.n_blocks = config.get('n_blocks', 2)

        n_resolutions = len(config.channel_multipliers)
        down = []
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config))
                in_channels = out_channels

            # down sample at all resolution except the last
            #if i < n_resolutions - 1:
            #    down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        # print(self.down)
        self.middle = MiddleBlock(out_channels, config)

        # #### Second half of U-Net - increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, config))

            out_channels = in_channels // config.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, config))
            in_channels = out_channels
            # up sample at all resolution except last
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(nn.Conv2d(self.d_h, self.F, (1,1)),
                                 nn.Linear(2 * T, T),)
        # for gcn
        '''
        a1 = asym_adj(config.A)
        a2 = asym_adj(np.transpose(config.A))
        self.a1 = torch.from_numpy(a1).to(config.device)
        self.a2 = torch.from_numpy(a2).to(config.device)
        config.supports_len = 2
        '''
        # for graph learning
        node_embeddings1 = torch.matmul(self.We1, self.Memory)
        node_embeddings2 = torch.matmul(self.We2, self.Memory)
        self.a1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        self.a2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)

    # for attention 
    def query_memory(self, h_t: torch.Tensor): # 수정 예정 주어진 모델의 rnn 구조를 보고 h_t 차원 등 수정해야됨
        h_att = []
        for h in h_t:
            # print(h.size())
            print(self.Wq.size())
            query = torch.matmul(h, self.Wq)
            att_score = torch.softmax(torch.matmul(query, self.Memory.t()), dim=-1)  # alpha: (B, N, M)
            value = torch.matmul(att_score, self.Memory)
            h_att.append(value)
        
        return h_att, query

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        """
        :param x: x_t of current diffusion step, (B, F, V, T)
        :param t: diffsusion step
        :param c: condition information
            used information in c:
                x_masked: (B, F, V, T)
        :return:
        """

        x_masked, pos_w, pos_d = c  # x_masked: (B, F, V, T), pos_w: (B,T,1,1), pos_d: (B,T,1,1)

        x = torch.cat((x, x_masked), dim=3) # (B, F, V, 2 * T)

        x = self.x_proj(x)

        t = TimeEmbedding(t, self.d_h)

        h = [x]

        supports = torch.stack([self.a1, self.a2])
        
        print(len(self.down))
        for m in self.down:
            print(m)
            x = m(x, t, supports) # 이걸 encoder 라 생각하면
            h.append(x)
            print(x.size())
            print('-----------------------------')

        # x = self.middle(x, t, supports)  # 삭제
        out = []
        h_att, _ = self.query_memory(h)
        print('done query_memory')
        ht_list = [torch.cat([h_i, h_att_i], dim =1) for h_i, h_att_i in zip(h, h_att)]
        print('done ht_list')
        # ht_list = [h_t] * len(self.up)
        #for t in range(self.config.horizon):
        for i, u in enumerate(self.up):
            s = ht_list.pop()
            x = torch.cat((x, s), dim=1)
            x = u(x,t, supports)          # 이걸 decoder 라 생각하면 # 여기 에러 고쳐야됨
        #out.append(self.out(x))
        print('done up')
        output= self.out(x)
        # e = self.out(x)
        #output = torch.stack(out, dim=1)
        return output
