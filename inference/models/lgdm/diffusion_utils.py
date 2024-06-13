import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        ).to(device)

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, extract_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.extract_dim = extract_dim
        self.pose_embedding = nn.Sequential(
            nn.Linear(self.input_feats, self.extract_dim//2),
            nn.Sigmoid(),
            nn.Linear(self.extract_dim//2, self.extract_dim),
            nn.Sigmoid(),
        )
        self.combination_extraction = nn.Sequential(
            nn.Linear(self.extract_dim*2, int(self.extract_dim*1.5)),
            nn.Sigmoid(),
            nn.Linear(int(self.extract_dim*1.5), self.extract_dim),
            nn.Sigmoid(),
        )
        if self.data_rep == 'rot_vel':
            self.vel_embedding = nn.Linear(self.input_feats, self.extract_dim)

    def forward(self, x, emb):
        """
        x: torch.Tensor.shape([bs, seq_len, dims, cat])
        emb: torch.Tensor.shape([bs, seq_len, extract_dim])
        """
        bs, pcd_points, dim = x.shape

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = x.to(torch.float32)
            x = self.pose_embedding(x)
            x = torch.cat((x, emb), dim=-1)
            x = self.combination_extraction(x)
            return x
        elif self.data_rep == 'rot_vel':
            # Undeveloping case of data representation
            raise "Undeveloping function"
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.pose_embedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.vel_embedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, extract_dim, pcd_points):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.extract_dim = extract_dim
        self.pcd_points = pcd_points
        self.pose_final = nn.Sequential(
            nn.Linear(self.extract_dim, self.extract_dim//2),
            nn.GELU(),
            nn.Linear(self.extract_dim//2, self.input_feats),
            nn.GELU(),
        )
        if self.data_rep == 'rot_vel':
            self.vel_final = nn.Linear(self.extract_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.pose_final(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            # Undeveloping case of data representation
            raise "Undeveloping function"
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.pose_final(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.vel_final(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, self.pcd_points, -1)
        return output