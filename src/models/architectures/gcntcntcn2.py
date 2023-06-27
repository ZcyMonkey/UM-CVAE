import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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

def augment_label_en(x,y, num_classes):
    bs,nfeats, nframes,njoints= x.size()
    y = y[:, :, :,None].repeat((1, 1, 1,nframes))
    return y.permute((1,2,3,0))

class Encoder_GCNTCNTCN2(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, latent_dim=256, num_layers=4, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints+1
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
        self.feats_latent_dim = self.njoints * self.node_feats_dim
        self.num_layers = num_layers

        # Layers
        self.input_feats = self.num_classes + self.feats_latent_dim
        self.label_embedding = nn.Parameter(torch.randn(self.num_classes, self.nfeats))

        self.embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.feats_embedding = nn.ModuleList()
        self.feats_embedding.append(ST_GCNN_layer(self.nfeats,64,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,256,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(256,64,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.feats_embedding.append(ST_GCNN_layer(64,self.node_feats_dim,[1,1],1,self.num_frames,
                                                  self.njoints,self.gcndrop))
        self.tcn_embedding = nn.ModuleList()
        self.tcn_embedding.append(CNN_layer(self.num_frames,30,[3,3],0))
        self.tcn_embedding.append(CNN_layer(30,15,[3,3],0))
        self.tcn_embedding.append(CNN_layer(15,1,[3,3],0))
        self.activation_gelu = nn.ModuleList()
        for j in range(3):
            self.activation_gelu.append(nn.GELU())

        if self.modeltype == "cvae":
            self.mu = nn.Linear(self.njoints*self.node_feats_dim, self.latent_dim)
            self.var = nn.Linear(self.njoints*self.node_feats_dim, self.latent_dim)
        else:
            self.final = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, batch):
        x, y, mask, lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]
        bs ,num_joint,num_feats, num_frames = x.size()
        y = self.label_embedding[y][None]
        x = x.permute((0, 2, 3, 1))
        label = augment_label_en(x,y,self.num_classes)
        x = torch.cat((x,label),axis=3)
        # Model
        for gcn in (self.feats_embedding):
            x = gcn(x)
        #feats_latent = x.permute((0, 2, 3, 1)).reshape(bs,num_frames, self.njoints,self.node_feats_latent_dim)
        x = x.permute((0, 2, 3, 1))
        for i in range(3):
            x = self.activation_gelu[i](self.tcn_embedding[i](x))
        x = x.reshape(bs,1,self.njoints*self.node_feats_dim)

        x = torch.squeeze(x,1)


        if self.modeltype == "cvae":
            return {"mu": self.mu(x), "logvar": self.var(x)}
        else:
            return {"z": self.final(x)}

class Decoder_GCNTCNTCN2(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, dropout=0.1, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.tcn_layer = 4
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.tcn_decoding = nn.ModuleList()
        for i in range(self.tcn_layer):
            self.tcn_decoding.append(CNN_layer(self.num_frames,self.num_frames,[3,3],0))
        self.activation_gelu = nn.ModuleList()
        for j in range(self.tcn_layer):
            self.activation_gelu.append(nn.GELU())
        self.feature_embedding = nn.Linear(self.latent_dim, self.njoints*self.nfeats)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        z = z + self.actionBiases[y]
        z = z[:,None,:].repeat(1,nframes,1)
        timequeries = torch.zeros(bs,nframes, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        z = self.feature_embedding(z + timequeries).reshape(bs, nframes,njoints,nfeats)
        for i in range(self.tcn_layer):
            z = self.activation_gelu[i](self.tcn_decoding[i](z)) + z

        z[~mask] = 0
        z = z.permute(0,2,3,1)
        batch["output"] = z
        return batch