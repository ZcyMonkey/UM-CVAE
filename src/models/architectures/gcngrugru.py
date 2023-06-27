import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .gcntrans2 import Encoder_GCNTRANS2 as Encoder_GCNGRUGRU
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

class Decoder_GCNGRUGRU(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, latent_dim=256, num_layers=4, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dropout = 0.1
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.concatenate_time = concatenate_time
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Layers
        self.input_feats = self.latent_dim + self.num_classes
        if self.concatenate_time:
            self.input_feats += 1
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.feats_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=False)
        self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.output_feats = self.njoints*self.nfeats
        self.final_layer = nn.Linear(self.latent_dim, self.output_feats)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        z = z + self.actionBiases[y]
        z = z[None].repeat((self.num_layers, 1, 1))
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        # Model
        z = self.gru(timequeries,z)[0]
        z = self.final_layer(z)

        # Post process
        z = z.reshape(bs, nframes, self.njoints, self.nfeats)
        # 0 for padded sequences
        z[~mask] = 0
        z = z.permute(0, 2, 3, 1)

        batch["output"] = z
        return batch