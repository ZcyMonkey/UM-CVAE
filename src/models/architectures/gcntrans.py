import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import Decoder_TRANSFORMER as Decoder_GCNTRANS
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

def augment_label_en(dtype,y, num_classes):
    if len(y.shape) == 1:  # can give on hot encoded as input
        y = F.one_hot(y, num_classes)
    y = y.to(dtype=dtype)
    y = y[:, None, :].repeat((1, 1, 1))
    return y


class Encoder_GCNTRANS(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, latent_dim=256, num_layers=4, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
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
        self.node_feats_latent_dim = 6
        self.feats_latent_dim = self.njoints * self.node_feats_latent_dim
        self.num_layers = num_layers

        # Layers
        self.input_feats = self.num_classes + self.feats_latent_dim


        self.embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.feats_embedding = nn.ModuleList()
        self.feats_embedding.append(ST_GCNN_layer(self.nfeats,64,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,256,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(256,64,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,self.node_feats_latent_dim,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))

        self.tcn = CNN_layer(self.num_frames,1,[3,3],0)

        if self.modeltype == "cvae":
            self.mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.var = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.final = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, batch):
        x, y, mask, lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]
        bs ,num_joint,num_feats, num_frames = x.size()
        x = x.permute((0, 2, 3, 1))
        label = augment_label_en(x.dtype,y,self.num_classes)

        # Model
        for gcn in (self.feats_embedding):
            x = gcn(x)
        #feats_latent = x.permute((0, 2, 3, 1)).reshape(bs,num_frames, self.njoints,self.node_feats_latent_dim)
        x = x.permute((0, 2, 3, 1))
        x = self.tcn(x)
        x = self.prelu(x).reshape(bs,1,self.njoints*self.node_feats_latent_dim)

        x = torch.squeeze(torch.cat((x,label),2),1)
        x = self.embedding(x)


        if self.modeltype == "cvae":
            return {"mu": self.mu(x), "logvar": self.var(x)}
        else:
            return {"z": self.final(x)}