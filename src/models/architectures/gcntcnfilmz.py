import ipdb as pdb
import torch
import math
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, kaiming_uniform


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)

def augment_label_en(x,y, num_classes):
    bs,nfeats, nframes,njoints= x.size()
    y = y[:, :, :,None].repeat((1, 1, 1,nframes))
    return y.permute((1,2,3,0))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 time_dim,
                 joints_dim,
                 input_channel
                 ):
        super(ConvTemporalGraphical,self).__init__()
        self.hidden_size = 256
        self.feature_size = input_channel*time_dim*joints_dim
        self.elu = nn.ELU()
        self.num_experts = 4
        self.nn_keep_prob = 0.7
        #self.BN = nn.BatchNorm1d(self.feature_size)
        self.gating_layer0 = nn.Linear(self.feature_size,self.hidden_size)
        self.gating_layer1 = nn.Linear(self.hidden_size,self.hidden_size)
        self.gating_layer2 = nn.Linear(self.hidden_size,self.num_experts)


        self.A=nn.Parameter(torch.FloatTensor(self.num_experts,time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(2))
        self.A.data.uniform_(-stdv,stdv)
    def get_AS(self, beta,controlweights, batch_size):
        b = torch.unsqueeze(beta, 0)  # 4*out*1   -> 4*1*out*1
        b = b.repeat([batch_size, 1, 1, 1,1])  # 4*1*out*1 -> 4*?*out*1
        w = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(controlweights, -1), -1)  ,-1)# 4*?        -> 4*?*1*1
        r = w * b  # 4*?*1*1 m 4*?*out*1
        return torch.sum(r, axis=1)  # ?*out*1


    def forward(self, x):
        Gating_input = x.reshape(-1, self.feature_size)
        h = nn.Dropout(self.nn_keep_prob)(Gating_input)
        h = self.elu(self.gating_layer0(h))

        h = nn.Dropout(self.nn_keep_prob)(h)
        h = self.elu(self.gating_layer1(h))
        h = nn.Dropout(self.nn_keep_prob)(h)
        h = self.gating_layer2(h)
        Gating_weight_s = nn.functional.softmax(h, dim=1)
        batch_size = x.size()[0]
        AS = self.get_AS(self.A,Gating_weight_s,batch_size)
        x = torch.einsum('nctv,ntvw->nctw', (x, AS))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()

class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim,in_channels) # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if stride != 1 or in_channels != out_channels:

            self.residual=nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual=nn.Identity()
        self.prelu = nn.PReLU()
    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x)
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x

class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
            ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)]
        self.block=nn.Sequential(*self.block)
    def forward(self, x):

        output= self.block(x)
        return output

class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_GCNTCNFILMZ(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, num_layers=4, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints_for_content = njoints
        self.njoints_for_label = njoints + 1
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.gcndrop = 0.7
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.prelu = nn.PReLU()
        self.latent_dim = latent_dim
        self.node_feats_dim = 6
        self.content_latent_dim = self.njoints_for_content * self.node_feats_dim
        self.label_latent_dim = self.njoints_for_label * self.node_feats_dim
        self.num_layers_for_content = 6
        self.num_layers_for_label = 2
        self.module_dim = 256
        self.cond_feat_size = 2 * self.module_dim
        self.IN = nn.InstanceNorm1d(8)
        # Layers
        self.label_embedding = nn.Parameter(torch.randn(self.num_classes, self.nfeats))

        self.feats_embedding = nn.ModuleList()
        self.feats_embedding.append(ST_GCNN_layer(self.nfeats,64,[1,1],1,self.num_frames,
                                                  self.njoints_for_label,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,256,[1,1],1,self.num_frames,
                                                  self.njoints_for_label,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(256,64,[1,1],1,self.num_frames,
                                                  self.njoints_for_label,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,self.node_feats_dim,[1,1],1,self.num_frames,
                                                  self.njoints_for_label,self.gcndrop))

        padding_mode = 'reflect'
        self.tcn_layers_for_label = nn.ModuleList()
        self.tcn_layers_for_content = nn.ModuleList()
        for i in range(self.num_layers_for_content):
            if i <2:
                kernel_size = 15
            elif i<4 and i>1:
                kernel_size = 7
            else:
                kernel_size = 1
            padding = (kernel_size - 1) // 2
            layer_conponent = []
            in_channels = self.content_latent_dim
            out_channels = self.content_latent_dim
            layer_conponent.append(NewSkeletonConv(in_channels,out_channels, kernel_size,
                                                   stride=2, padding=padding, bias=True,
                                                   padding_mode=padding_mode))
            layer_conponent.append(nn.GELU())
            self.tcn_layers_for_content.append(nn.Sequential(*layer_conponent))
        for i in range(self.num_layers_for_label):
            kernel_size = 15
            padding = (kernel_size - 1) // 2
            layer_conponent = []
            in_channels = self.label_latent_dim
            out_channels = self.label_latent_dim
            layer_conponent.append(NewSkeletonConv(in_channels,out_channels, kernel_size,
                                                   stride=1, padding=padding, bias=True,
                                                   padding_mode=padding_mode))
            layer_conponent.append(nn.GELU())
            self.tcn_layers_for_label.append(nn.Sequential(*layer_conponent))
        self.modulation = nn.Linear(self.label_latent_dim*num_frames, self.cond_feat_size)
        if self.modeltype == "cvae":
            self.mu = nn.Linear(self.njoints_for_content*self.node_feats_dim, self.latent_dim)
            self.var = nn.Linear(self.njoints_for_content*self.node_feats_dim, self.latent_dim)
        else:
            self.final = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, batch):
        x, y, mask, lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]
        bs ,num_joint,num_feats, num_frames = x.size()
        x_for_label = x
        x_for_con = x
        x_for_con = x_for_con.reshape(bs,self.content_latent_dim,num_frames)
        for i in range(self.num_layers_for_content):
            if i == 3:
                x_for_con = self.IN(x_for_con.permute(0,2,1)).permute(0,2,1)
            x_for_con = self.tcn_layers_for_content[i](x_for_con)
        x_for_con = torch.squeeze(x_for_con,2)


        y = self.label_embedding[y][None]
        x_for_label = x_for_label.permute((0, 2, 3, 1))
        label = augment_label_en(x_for_label,y,self.num_classes)
        x_for_label = torch.cat((x_for_label,label),axis=3)
        # Model
        for gcn in (self.feats_embedding):
            x_for_label = gcn(x_for_label)
        x_for_label = x_for_label.permute((0, 1, 3, 2)).reshape(bs,self.label_latent_dim,num_frames)
        for tcn in (self.tcn_layers_for_label):
            x_for_label = tcn(x_for_label)
        x_for_label = x_for_label.reshape(bs,self.label_latent_dim*num_frames)
        if self.modeltype == "cvae":
            return {"mu": self.mu(x_for_con), "logvar": self.var(x_for_con),"gamma": self.modulation (x_for_label)[:,:self.module_dim], "beta": self.modulation (x_for_label)[:,self.module_dim:]}
        else:
            return {"z": self.final(x_for_con),"gamma": self.modulation (x_for_label)[:,:self.module_dim], "beta": self.modulation (x_for_label)[:,self.module_dim:]}


class Decoder_GCNTCNFILMZ(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, dropout=0.1, num_layers=6,**kargs):
        super().__init__()
        self.num_layers = 4
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.tcn_layer = 6
        self.latent_dim = latent_dim
        self.gru_latent_dim = latent_dim
        self.dropout = dropout
        self.Dropout = nn.Dropout(p=dropout)
        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.gru1 = nn.GRU(self.gru_latent_dim, self.gru_latent_dim, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(self.gru_latent_dim, self.gru_latent_dim, num_layers=1, batch_first=True)
        self.gru3 = nn.GRU(self.gru_latent_dim, self.gru_latent_dim, num_layers=1, batch_first=True)
        self.gru4 = nn.GRU(self.gru_latent_dim, self.gru_latent_dim, num_layers=1, batch_first=True)

        self.feats_embedding = nn.Linear(self.latent_dim, self.gru_latent_dim)
        self.final_layer = nn.Linear(self.gru_latent_dim, self.njoints*self.nfeats)

    def forward(self, batch):
        gamma,beta,z, y, mask, lengths = batch["gamma"],batch["beta"],batch["z"], batch["y"], batch["mask"], batch["lengths"]
        gamma = self.Dropout(gamma)
        z_in = gamma*z + beta
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats
        z_in = z_in[:,None,:].repeat(1,nframes,1)
        z = z[:,None,:].repeat(1,nframes,1)
        timequeries = torch.zeros( nframes,bs,  latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries).permute(1,0,2)
        z = z + timequeries
        z_in = z_in + timequeries
        z_in = self.feats_embedding(z_in)
        z_in = self.gru1(z_in)[0]
        z_in = self.gru2(z_in+z)[0]
        z_in = self.gru3(z_in+z)[0]
        z_in = self.gru4(z_in+z)[0]
        z_out = self.final_layer(z_in)
        z_out[~mask] = 0
        z_out = z_out.permute(0,2,1).reshape(bs,njoints,nfeats,nframes)


        batch["output"] = z_out
        return batch

class NewSkeletonConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, padding_mode='zeros'):

        super(NewSkeletonConv, self).__init__()

        if padding_mode == 'zeros': padding_mode = 'constant'
        if padding_mode == 'reflect': padding_mode = 'reflect'

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)
        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       self.weight, self.bias, self.stride,
                       0, self.dilation, self.groups)
        return res