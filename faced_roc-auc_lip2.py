import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import argparse
#import data_prepare_eav as data_prepare
import data_prepare_faced as data_prepare
#import data_prepare_seed as data_prepare
import math
from loss import CustomLoss
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from scipy.stats import entropy
from torch.nn.functional import one_hot
from datetime import datetime

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        return self.dropout(x)

class MultiScaleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, scales=[1, 2, 4], lipschitz_num=1.0):
        super(MultiScaleAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scales = scales
        self.scaling = self.head_dim ** -0.5
        self.lipschitz_num = lipschitz_num

        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.v_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim).double()

    def lipschitz_normalize(self, scores, query, key):
        query_norm = torch.norm(query, dim=-1, keepdim=True)  
        key_norm = torch.norm(key, dim=-1, keepdim=True)  

        max_query_norm = query_norm.max(dim=-2, keepdim=True)[0]  
        max_key_norm = key_norm.max(dim=-2, keepdim=True)[0] 

        lipschitz_denominator = torch.clamp(max_query_norm * max_key_norm, min=1e-6)  
        normalized_scores = scores / lipschitz_denominator

        normalized_scores = normalized_scores * self.lipschitz_num
        return normalized_scores

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhld,bhmd->bhlm", query, key) * self.scaling
        scores = self.lipschitz_normalize(scores, query, key)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        output = torch.einsum("bhlm,bhmd->bhld", attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output
        
class LGCA(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, scales=[1, 2, 4, 8],lipschitz_num=1.0):
        super(LGCA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.scales = scales
        self.lipschitz_num=lipschitz_num

        self.attention = MultiScaleAttention(embed_dim, num_heads, dropout, scales,lipschitz_num=self.lipschitz_num)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward).double(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim).double()
        )

        self.layernorm1 = nn.LayerNorm(embed_dim).double()
        self.layernorm2 = nn.LayerNorm(embed_dim).double()

    def forward(self, x):
        attn_output = self.attention(x)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        x = self.layernorm1(x + attn_output)
        mlp_output = self.mlp(x)
        mlp_output = F.dropout(mlp_output, p=self.dropout, training=self.training)
        x = self.layernorm2(x + mlp_output)

        return x
    
class DilatedAttentionBlock_attention(nn.Module):
    def __init__(self, in_channels, seq_length, out_channels, dropout_rate=0.0, num_heads=2, lipschitz_num=1.0):
        super(DilatedAttentionBlock_attention, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"
        self.lipschitz_num = lipschitz_num

        self.q_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.k_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.v_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.out_linear_time = nn.Linear(out_channels, out_channels).double()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_time = nn.LayerNorm(in_channels).double()

    def forward(self, x_in):
        batch_size, in_channels, seq_length = x_in.shape
        x = x_in.permute(0, 2, 1) 

        Q_time = self.q_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_time = self.k_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_time = self.v_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores_time = torch.matmul(Q_time, K_time.transpose(-2, -1)) / math.sqrt(self.head_dim)

        score_norm = torch.norm(attention_scores_time, dim=-1, keepdim=True)
        attention_scores_time = attention_scores_time / (score_norm * self.lipschitz_num + 1e-12)
        attention_weights_time = F.softmax(attention_scores_time, dim=-1)
        attention_output_time = torch.matmul(attention_weights_time, V_time).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attention_output_time = self.out_linear_time(attention_output_time)
        attention_output_time = self.norm_time(attention_output_time)
        attention_output_time = attention_output_time.permute(0, 2, 1) 
        combined_output = self.elu(attention_output_time)
        combined_output = self.dropout(combined_output)
        combined_output = combined_output + x_in 

        return combined_output

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, input_dim).double()
        self.dropout = nn.Dropout(0.1).double()

    def forward(self, x, lipschitz_num=1.0):
        self.fc1.weight.data = self.fc1.weight.data / self.fc1.weight.data.norm() * lipschitz_num
        self.fc2.weight.data = self.fc2.weight.data / self.fc2.weight.data.norm() * lipschitz_num

        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.dropout(out)
        return residual + out 

class FeatureAttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, lipschitz_num=1.0):
        super(FeatureAttentionModule, self).__init__()
        self.query_mlp = ResidualMLP(input_dim, hidden_dim)
        self.key_mlp = ResidualMLP(input_dim, hidden_dim)
        self.value_mlp = ResidualMLP(input_dim, hidden_dim)
        self.lipschitz_num = lipschitz_num

    def forward(self, matrix):
        seq_len, input_dim = matrix.shape

        Q = self.query_mlp(matrix.T, lipschitz_num=self.lipschitz_num)  
        K = self.key_mlp(matrix.T, lipschitz_num=self.lipschitz_num)    
        V = self.value_mlp(matrix.T, lipschitz_num=self.lipschitz_num)  

        attention_scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(input_dim, dtype=torch.float64))
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_scores, V)  

        return attention_output.T, attention_scores

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).double()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = adj.sum(1)
        D_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = D_inv_sqrt.view(-1, 1)
        adj_normalized = D_inv_sqrt * adj * D_inv_sqrt.t()
        support = torch.mm(x, self.weight.to(x.device)) 
        out = torch.mm(adj_normalized, support) 
        out = F.relu(out)
        return out

class LGCBE(nn.Module):
    def __init__(self, num_channels: int = 30, seq_length: int = 250, sampling_rate: float = 250.0, lipschitz_num: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.lipschitz_num = lipschitz_num

        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(num_channels, num_channels, dtype=torch.float64),
            nn.Sigmoid()
        )

        self.spectral_attention = nn.Sequential(
            nn.Linear(len(self.bands), len(self.bands), dtype=torch.float64),
            nn.GELU(),
            nn.Linear(len(self.bands), len(self.bands), dtype=torch.float64),
            nn.Softmax(dim=-1)
        )

        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.norm = nn.LayerNorm(seq_length, dtype=torch.float64)

    def apply_lipschitz_constraint(self):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    weight_norm = torch.linalg.svdvals(module.weight)[0]
                    module.weight.data = module.weight.data / weight_norm * self.lipschitz_num

    def lipschitz_attention(self, scores):
        scores_norm = torch.norm(scores, dim=-1, keepdim=True)
        return scores / (scores_norm + 1e-6) * self.lipschitz_num

    def get_band_mask(self, freqs: torch.Tensor, band: str) -> torch.Tensor:
        low, high = self.bands[band]
        return ((freqs >= low) & (freqs <= high)).to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.to(dtype=torch.float64)
        identity = x
        batch_size = x.shape[0]

        self.apply_lipschitz_constraint()

        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(self.seq_length, 1 / self.sampling_rate).to(x.device, dtype=torch.float64)

        band_features = {}
        band_powers = []

        for band in self.bands.keys():
            mask = self.get_band_mask(freqs, band).to(x.device)
            X_band = X * mask.unsqueeze(0).unsqueeze(0)
            band_features[band] = X_band
            power = torch.sum(torch.abs(X_band).pow(2), dim=-1)
            band_powers.append(power)

        band_powers = torch.stack(band_powers, dim=-1)

        channel_weights = self.channel_attention(band_powers.mean(dim=-1))
        channel_weights = channel_weights.unsqueeze(-1) 

        spectral_input = band_powers.mean(dim=1)  
        spectral_scores = self.spectral_attention(spectral_input)  

        spectral_weights = self.lipschitz_attention(spectral_scores)

        X_combined = torch.zeros_like(X, dtype=torch.complex128)  
        for i, band in enumerate(self.bands.keys()):
            channel_weights_complex = channel_weights.to(dtype=torch.complex128)
            spectral_weights_complex = spectral_weights[:, i:i+1].unsqueeze(1).to(dtype=torch.complex128)
            X_combined += (band_features[band] * channel_weights_complex * spectral_weights_complex)

        output = torch.fft.irfft(X_combined, n=self.seq_length, dim=-1)
        output = self.norm(output)

        alpha = torch.sigmoid(self.alpha)
        output = alpha * output + (1 - alpha) * identity

        extracted_features = {
            "band_features": band_features,  
            "band_powers": band_powers,    
            "channel_weights": channel_weights,  
            "spectral_weights": spectral_weights  
        }

        return output, extracted_features
    
def LGCN(tensor, lipschitz_constant, eps=1e-8):
    mean = tensor.mean(dim=1, keepdim=True) 
    std = tensor.std(dim=1, keepdim=True) 
    
    normalized_tensor = (tensor - mean) / (std + eps)
    
    grad_norm = torch.norm(normalized_tensor, dim=1, keepdim=True) 
    
    lipschitz_factor = torch.clamp(grad_norm / lipschitz_constant, min=1.0)
    
    lipschitz_normalized_tensor = normalized_tensor / (lipschitz_factor + eps)
    
    return lipschitz_normalized_tensor

class GCN_energy(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.1):
        super(GCN_energy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).double()
        nn.init.xavier_uniform_(self.weight)
        self.dropout=nn.Dropout(dropout)

    def forward(self, features, adj):
        batchsize, channel, energy = features.size()
        adj_with_self_loop = adj + torch.eye(channel).to(adj.device).unsqueeze(0).expand(batchsize, -1, -1).double()
        D = torch.diag_embed(torch.pow(adj_with_self_loop.sum(dim=2), -0.5))
        adj_normalized = torch.bmm(torch.bmm(D, adj_with_self_loop), D)
        output = torch.bmm(adj_normalized, features)
        output = torch.bmm(output, self.weight.unsqueeze(0).expand(batchsize, -1, -1).to(output.device))  
        output=self.dropout(output)
        output = F.relu(output)
        return output
    
class LEREL(nn.Module):
    def __init__(self, eeg_channel,timepoint, dropout=0.1, num_class=5,lipschitz_num=1.0):
        super().__init__()
        self.channel=eeg_channel
        self.time=timepoint
        self.num_class=num_class
        self.lipschitz_num=lipschitz_num
        self.dp=dropout
        
        self.conv_c = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        ).double()
        
        self.transformer_conv_channel = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            LGCA(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
        ).double()
        self.dab_conv_channel=nn.Sequential(
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.3,4,self.lipschitz_num),
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.2,4,self.lipschitz_num),
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.1,4,self.lipschitz_num),
        )
        self.mlp_conv1 = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, num_class),
        ).double()
        self.attention_relation_channel=FeatureAttentionModule(timepoint,timepoint,self.lipschitz_num)

        self.gcn_channel=GraphConvolution(timepoint,timepoint)

        self.c_x = nn.Parameter(torch.tensor(1.0))

        self.nsam=LGCBE(eeg_channel,timepoint,timepoint//10,self.lipschitz_num)
        self.final=nn.Sequential(
            nn.LayerNorm(5+6*eeg_channel, dtype=torch.float64),
            nn.Linear(5+6*eeg_channel, (5+6*eeg_channel)//2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear((5+6*eeg_channel)//2, num_class),
        ).double()

        self.attention_relation_channel_energy=FeatureAttentionModule(5,5,self.lipschitz_num)
        self.gcn_energy=GCN_energy(5,5,dropout)
        self.c_energy = nn.Parameter(torch.tensor(1.0))
        self.band_energy_out=nn.Sequential(
            nn.LayerNorm(5*eeg_channel, dtype=torch.float64),
            nn.Linear(5*eeg_channel, 5*eeg_channel//2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(5*eeg_channel//2, num_class),
        ).double()

        self.band_magnitudes_out=nn.Sequential(
            nn.LayerNorm(eeg_channel*5*(timepoint//2+1), dtype=torch.float64),
            nn.Linear(eeg_channel*5*(timepoint//2+1), eeg_channel*5*(timepoint//2+1)//10),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel*5*(timepoint//2+1)//10, num_class),
        ).double()

        self.out1 = nn.Parameter(torch.tensor(1.0))
        self.out2 = nn.Parameter(torch.tensor(1.0))
        self.out3 = nn.Parameter(torch.tensor(1.0))
        self.out4 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = x.to(dtype=torch.float64)
        batchsize,channel,time=x.shape
        
        x_c = self.conv_c(x)
        channel_list=[]
        attention_output_list = []
        for b in x_c:
            attention_output,channel_relation=self.attention_relation_channel(b.T)
            channel_list.append(channel_relation)
            attention_output_list.append(attention_output)
        channel_adj = torch.stack(channel_list, dim=0)  
        channel_adj = channel_adj / channel_adj.sum(dim=-1, keepdim=True) 
        attention_output_stack = torch.stack(attention_output_list, dim=0).permute(0,2,1)
        x_conv1 = self.transformer_conv_channel(x_c.permute(0, 2, 1)).permute(0, 2, 1)
        x_conv1 = self.dab_conv_channel(x_conv1)
        gcn_channel_list=[]
        for b in range(batchsize):
            x_gcn = self.gcn_channel(x_conv1[b, :, :], channel_adj[b,:,:])  
            x_gcn = x_gcn.permute(1, 0)
            gcn_channel_list.append(x_gcn)
        x_gcn_c = torch.stack(gcn_channel_list, dim=0)
        x_gcn_c = x_gcn_c.permute(0,2,1)
        x_conv1 = x_conv1 + self.c_x*x_gcn_c 
        x_conv1 = x_conv1.mean(dim=2)
        x_out1 = self.mlp_conv1(x_conv1)

        x_nsam,x_nsam_features=self.nsam(x)
        x_nsam_band_feature=x_nsam_features["band_features"]
        x_nsam_band_powers=x_nsam_features["band_powers"]
        x_nsam_channel_weights=x_nsam_features["channel_weights"]
        x_nsam_spectral_weights=x_nsam_features["spectral_weights"]
        x_nsam_band_powers_flat = x_nsam_band_powers.view(x_nsam_band_powers.size(0), -1)
        x_nsam_channel_weights_flat = x_nsam_channel_weights.squeeze(-1) 
        x_nsam_combined_features = torch.cat([x_nsam_band_powers_flat, x_nsam_channel_weights_flat, x_nsam_spectral_weights], dim=1)
        x_out2=self.final(x_nsam_combined_features)

        x_nsam_band_energy=[]
        for band, feature in x_nsam_band_feature.items():
            energy = torch.sum(torch.abs(feature) ** 2, dim=-1) 
            x_nsam_band_energy.append(energy)
        x_band_energies = torch.stack(x_nsam_band_energy, dim=-1)
        channel_list_energy=[]
        attention_output_list_energy=[]
        for b in x_band_energies:
            attention_energy,channel_relation=self.attention_relation_channel_energy(b.T)
            channel_list_energy.append(channel_relation)
            attention_output_list_energy.append(attention_energy)
        channel_adj_energy = torch.stack(channel_list_energy, dim=0) 
        channel_adj_energy = channel_adj_energy/ channel_adj_energy.sum(dim=-1, keepdim=True) 
        attention_output_stack_energy = torch.stack(attention_output_list_energy, dim=0).permute(0,2,1)
        x_band_energies_conv = self.gcn_energy(x_band_energies,channel_adj_energy)
        x_band_energies = x_band_energies + self.c_energy*x_band_energies_conv
        x_band_energies_flat = x_band_energies.view(x_band_energies.size(0), -1)
        x_out3=self.band_energy_out(x_band_energies_flat)

        
        x_nasm_band_magnitudes = []
        for band, feature in x_nsam_band_feature.items():
            magnitude = torch.abs(feature)  
            x_nasm_band_magnitudes.append(magnitude)
        x_band_magnitudes = torch.stack(x_nasm_band_magnitudes, dim=-1)
        x_band_magnitudes_flat = x_band_magnitudes.view(x_band_magnitudes.size(0), -1)
        x_out4=self.band_magnitudes_out(x_band_magnitudes_flat)

        normalized_tensor1 = LGCN(x_out1,lipschitz_constant=self.lipschitz_num)
        normalized_tensor2 = LGCN(x_out2,lipschitz_constant=self.lipschitz_num)
        normalized_tensor3 = LGCN(x_out3,lipschitz_constant=self.lipschitz_num)
        normalized_tensor4 = LGCN(x_out4,lipschitz_constant=self.lipschitz_num)
        
        x_concat = torch.stack((
            self.out1*normalized_tensor1, 
            self.out2*normalized_tensor2, 
            self.out3*normalized_tensor3, 
            self.out4*normalized_tensor4,
        ), dim=0)
        x_out = torch.mean(x_concat, dim=0)
        return x_out

class EEGModelTrainer:
    def __init__(self,  train_dataloader, val_dataloader, model = [], sub = '', lr=0.001, batch_size = 64):
        if model:
            self.model = model
        else:
            self.model = EEGClassificationModel(eeg_channel=30)

        self.batch_size = batch_size
        self.test_acc = float()

        self.train_dataloader = train_dataloader
        self.test_dataloader = val_dataloader

        self.initial_lr = lr
        self.criterion = CustomLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.compile(self.model)
        model.to(self.device)
        print(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        accuracies = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
                accuracies.extend((predicted == labels).cpu().numpy())
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.2f}')
        return accuracy, predictions

    def calculate_f1(self, labels, preds, num_classes):
        eps = 1e-7  
        labels_onehot = one_hot(labels, num_classes=num_classes).float()
        preds_onehot = one_hot(preds, num_classes=num_classes).float()
    
        tp = torch.sum(labels_onehot * preds_onehot, dim=0) 
        fp = torch.sum(preds_onehot * (1 - labels_onehot), dim=0) 
        fn = torch.sum(labels_onehot * (1 - preds_onehot), dim=0)  
    
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
    
        return f1.mean().item() 


    def train(self, epochs=25, lr=None, freeze=False, num_class=5, log_dir="logs", sub=-1):
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        log_filename = f"train_log_sub{sub}_{current_time}.txt"
        log_path = os.path.join(log_dir, log_filename)
    
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
        with open(log_path, "w") as log_file:
            lr = lr if lr is not None else self.initial_lr
            if lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
    
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
    
            for param in self.model.parameters():
                param.requires_grad = not freeze
    
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                log_file.write(f"GPU: {torch.cuda.device_count()}\n")
    
            best_val_accuracy = 0.0
            best_epoch = 0
            best_val_f1 = 0.0
            best_model_state = None
            best_conf_matrix = None 
            best_true_labels = []
            best_pred_labels = []
    
            for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                noise_std = 0.2
    
                self.model.train()
                for inputs, labels in self.train_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    noise = torch.randn_like(inputs) * noise_std 
                    noisy_inputs = inputs + noise
                    self.optimizer.zero_grad()
                    outputs = self.model(noisy_inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
    
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    label = torch.argmax(labels, dim=1)
                    correct_predictions += (predicted == label).sum().item()
    
                train_loss = running_loss / len(self.train_dataloader.dataset)
                train_accuracy = correct_predictions / total_predictions
    
                self.model.eval()
                running_val_loss = 0.0
                val_correct_predictions = 0
                val_total_predictions = 0
                val_labels = []  
                val_preds = []  
                val_probs = [] 
                with torch.no_grad():
                    for inputs, labels in self.test_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        val_loss = self.criterion(outputs, labels)
                        running_val_loss += val_loss.item() * inputs.size(0)
    
                        _, predicted = torch.max(outputs.data, 1)
                        probabilities = F.softmax(outputs, dim=1)
                        val_probs.append(probabilities.cpu())
                        val_total_predictions += labels.size(0)
                        label = torch.argmax(labels, dim=1)
                        val_correct_predictions += (predicted == label).sum().item()
    
                        val_labels.append(label.cpu())  
                        val_preds.append(predicted.cpu())  
    
                val_loss = running_val_loss / len(self.test_dataloader.dataset)
                val_accuracy = val_correct_predictions / val_total_predictions
    
                val_labels = torch.cat(val_labels, dim=0)
                val_preds = torch.cat(val_preds, dim=0)
                val_probs = torch.cat(val_probs, dim=0)
    
                val_f1 = self.calculate_f1(val_labels, val_preds, num_classes=num_class)
    
                if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_f1 > best_val_f1):
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict()
    
                    val_conf_matrix = confusion_matrix(val_labels, val_preds)
    
                    best_true_labels = val_labels
                    best_pred_labels = val_preds
                    best_pred_probs = val_probs
    
                    val_conf_matrix_percentage = (val_conf_matrix.astype('float') / val_conf_matrix.sum(axis=1)[:, np.newaxis]) * 100
                    val_conf_matrix_percentage = np.round(val_conf_matrix_percentage, 2)
    
                    labels = [f"Class {i}" for i in range(num_class)]
                    best_conf_matrix = val_conf_matrix_percentage
                    best_conf_labels = labels
    
                log_content = (
                    f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                    f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}, Test F1 Score: {val_f1:.4f}\n'
                )
                log_file.write(log_content)
    
            final_log_content = (
                f"Best Test Accuracy: {best_val_accuracy:.4f} at Epoch {best_epoch}/{epochs}, with F1 Score: {best_val_f1:.4f}\n"
            )
            print(final_log_content, end="")
            log_file.write(final_log_content)

        self.model.load_state_dict(best_model_state)
        print("Loaded the best model for testing.")

        with open(log_path, "a") as log_file:
            log_file.write("Best Model Confusion Matrix (%) of this subject:\n")
            log_file.write(" " * 8)
            for label in best_conf_labels:
                log_file.write(f"{label:^10}")
            log_file.write("\n")
            for i, row in enumerate(best_conf_matrix):
                log_file.write(f"{best_conf_labels[i]:^8}")
                for value in row:
                    log_file.write(f"{value:^10.2f}")
                log_file.write("\n")

        return best_true_labels, best_pred_labels, best_pred_probs
    
def convert_hot_encoding(label):
    if not isinstance(label, np.ndarray) or label.ndim != 2 or label.shape[1] != 10:
        raise ValueError("输入的独热编码必须是二维数组，且第二维长度为10。")
    
    batch_size = label.shape[0]
    new_label = np.zeros((batch_size, 5), dtype=label.dtype)
    
    for i in range(batch_size):
        for j in range(5):
            if label[i, 2*j] == 1 or label[i, 2*j+1] == 1:
                new_label[i, j] = 1
    
    return new_label
def resample_data(data, original_time_points, new_time_points):
    num_samples, _, channels = data.shape

    original_indices = np.linspace(0, original_time_points - 1, original_time_points)
    new_indices = np.linspace(0, original_time_points - 1, new_time_points)
    resampled_data = np.zeros((num_samples, new_time_points, channels))

    for i in range(num_samples):
        for j in range(channels):
            channel_data = data[i, :, j]
            if len(channel_data) != original_time_points:
                raise ValueError(f"通道数据长度 {len(channel_data)} 与原始时间点数量 {original_time_points} 不一致。")
            resampled_channel_data = np.interp(new_indices, original_indices, channel_data)
            resampled_data[i, :, j] = resampled_channel_data

    return resampled_data
def filter_odd_even(data, labels, invert=False):
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("输入数据必须是形状为 (len, time, channel) 的三维 numpy 数组")
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("输入标签必须是形状为 (len, label) 的二维 numpy 数组")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("数据和标签的第一个维度长度必须一致")
    
    if invert:
        print("active mode")
        filtered_data = data[1::2, :, :]
        filtered_labels = labels[1::2, :]
    else:
        print("passive mode")
        filtered_data = data[::2, :, :]
        filtered_labels = labels[::2, :]
    
    return filtered_data, filtered_labels

if __name__ == "__main__":

    random_split_seed=100
    cut_len=200
    train_deep=200 
    num_class=9 
    eeg_channel=32
    batch_size = 500 
    shift_step = 25
    '''
    data_path='/path/EAV/perprocess'#/False
    dataset_path='/path/EAV/ori'
    '''
    data_path='/path/FACED/perprocess'
    dataset_path='/path/FACED'
    '''
    data_path='/path/SEED/perprocess'
    dataset_path='/path/SEED'
    '''
    data_list,label_list=data_prepare.load_data_from_file(args,dataset_path,data_path,data_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', default=1, type=int, help="Subject number")
    parser.add_argument('--n_ses', default=1, type=int, help="Number of sessions")
    parser.add_argument('--sfreq', default=250, type=int, help="Resampling frequency")
    parser.add_argument("--preprocessed_max_subject", type=int, default=122, help="Maximum subject number for preprocessed data check, eav max 42,faced max 122,seed max 15")
    args = parser.parse_args()

    data_list,label_list=data_prepare.load_data_from_file(args,dataset_path,data_path,data_path)

    all_val_labels = []
    all_val_probs = []
    
    if len(data_list)==len(label_list):
        start=0
        fin=40
        print(f"\n total: {fin-start}")
        all_best_true_labels=[]
        all_best_pred_labels=[]
        for i in range(start,fin):
            print(f"\n subject {i+1}")
            
            subject_data=np.transpose(data_list[i], (2, 0, 1))
            subject_label=np.transpose(label_list[i], (1, 0))

            #subject_data,subject_label=filter_odd_even(subject_data,subject_label,True)#fonly eav use

            subject_data_extend,subject_label_extend=data_prepare.cut_and_extend_data(subject_data,subject_label,cut_len)
            
            subject_data=resample_data(subject_data_extend,cut_len,train_deep)
            subject_label=subject_label_extend

            subject_data=np.transpose(subject_data, (0, 2, 1))
            subject_label = subject_label.astype(int)
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            
            for train_index, test_index in sss.split(subject_data, subject_label):
                train_data, test_data = subject_data[train_index], subject_data[test_index]
                train_labels, test_labels = subject_label[train_index], subject_label[test_index]
            
            train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
            test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

            train_labels = [torch.argmax(train_dataset[i][1]).item() for i in range(len(train_dataset))]
            test_labels = [torch.argmax(test_dataset[i][1]).item() for i in range(len(test_dataset))]
            
            train_label_distribution = Counter(train_labels)
            test_label_distribution = Counter(test_labels)
            
            print("Train Dataset Label Distribution:")
            for label, count in sorted(train_label_distribution.items()):
                print(f"Label {label}: {count} samples ({count / len(train_labels) * 100:.2f}%)", end="   ")
            
            print("\n\nTest Dataset Label Distribution:")
            for label, count in sorted(test_label_distribution.items()):
                print(f"Label {label}: {count} samples ({count / len(test_labels) * 100:.2f}%)", end="   ")

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
            model = LEREL(eeg_channel=eeg_channel,timepoint=train_deep,dropout=0.1,num_class=num_class,lipschitz_num=10.0)  # Example: 64 EEG channels
            trainer = EEGModelTrainer(train_dataloader, test_dataloader,model)
            print("lr=",trainer.initial_lr,"  batch_size=",batch_size,"   dropout=",model.dp,"    lipschitz_num=",model.lipschitz_num)
            best_true_labels, best_pred_labels, best_pred_probs = trainer.train(epochs=100,num_class=num_class,log_dir="/path/report/rocauc/pic_100",sub=i+1)

            best_pred_probs = torch.tensor(best_pred_probs) if not isinstance(best_pred_probs, torch.Tensor) else best_pred_probs

            all_val_labels.extend(best_true_labels)
            all_val_probs.append(best_pred_probs)
            
            all_best_true_labels.extend(best_true_labels)
            all_best_pred_labels.extend(best_pred_labels)
    
            current_overall_conf_matrix = confusion_matrix(all_best_true_labels, all_best_pred_labels)
            current_overall_conf_matrix_percentage = (current_overall_conf_matrix.astype('float') / current_overall_conf_matrix.sum(axis=1)[:, np.newaxis]) * 100
            current_overall_conf_matrix_percentage = np.round(current_overall_conf_matrix_percentage, 2)

        all_val_labels = torch.tensor(all_val_labels)
        all_val_probs = torch.cat(all_val_probs, dim=0) if isinstance(all_val_probs[0], torch.Tensor) else torch.tensor(all_val_probs)
        
        final_overall_conf_matrix = confusion_matrix(all_best_true_labels, all_best_pred_labels)
        final_overall_conf_matrix_percentage = (final_overall_conf_matrix.astype('float') / final_overall_conf_matrix.sum(axis=1)[:, np.newaxis]) * 100
        final_overall_conf_matrix_percentage = np.round(final_overall_conf_matrix_percentage, 2)



