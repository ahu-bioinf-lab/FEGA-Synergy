import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

# 药物特征的处理
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        # 使用 Embedding 来处理
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)
        seq_length = input_ids.size(1)
        # 创建等差数列[0, seq_length]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # 创建了位置参数
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings  # 药物初始结构向量和药物的位置向量
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class CellEncoder_CNN_Transformer(nn.Module):
    def __init__(self, in_dim=6, hidden=64, n_tokens=128, dim=128, depth=2, heads=4):
        super().__init__()

        # —— 1D CNN，多尺度卷积 —— #
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ——— 降采样：Adaptive Pool 固定 token 数 ——— #
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)   # 4079 → 128 tokens

        # ——— Transformer 用于基因间跨基因依赖 ——— #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (B, 4079, 6)
        x = x.transpose(1, 2)          # (B, 6, 4079)
        x = self.conv(x)               # (B, hidden, 4079)
        x = self.pool(x)               # (B, hidden, n_tokens)
        x = x.transpose(1, 2)          # (B, n_tokens, hidden)
        x = self.proj(x)               # (B, n_tokens, dim)
        x = self.transformer(x)        # (B, n_tokens, dim)
        return x


class CellNIN_A(nn.Module):
    def __init__(self, in_channel=6, feat_dim=64, hidden=32):
        super().__init__()

        self.omics_attn = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.omics_proj = nn.Sequential(
            nn.Conv1d(in_channel, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1)
        )

        self.cell_linear = nn.Linear(hidden, feat_dim)

    def forward(self, x):
        x = x.transpose(1, 2)      # (B, 6, L)
        attn = self.omics_attn(x)  # (B, 6, L)
        x = x * attn
        x = self.omics_proj(x)
        x = x.transpose(1, 2)
        return self.cell_linear(x)
    
class CellNIN_CNN(nn.Module):
    def __init__(self, feat_dim=128, drop_rate=0.2):
        super().__init__()

        # ========= 1️⃣ NIN：通道融合 =========
        self.nin = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

        # ========= 2️⃣ CNN：序列建模 =========
        self.cnn = nn.Sequential(
            # [B, 16, L] → [B, 32, L1]
            nn.Conv1d(16, 32, kernel_size=16),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(2),

            # [B, 32, L1] → [B, 64, L2]
            nn.Conv1d(32, 64, kernel_size=16),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(2),

            # [B, 64, L2] → [B, 165, L3]
            nn.Conv1d(64, 165, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(6),
        )

        # ========= 3️⃣ 投影到 embedding =========
        self.linear = nn.Linear(165, feat_dim)

    def forward(self, x):
        """
        x: [batch, L, 6]
        """
        x = x.transpose(1, 2)   # [B, 6, L]

        x = self.nin(x)         # [B, 16, L]
        x = self.cnn(x)         # [B, 165, L_out]

        x = x.transpose(1, 2)   # [B, L_out, 165]
        x = self.linear(x)      # [B, L_out, 128]

        return x
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


# CNN处理细胞系多组学数据
class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]

        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]

        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)


    def forward(self, x):

        # print('x_cell_embed.shape:',x_cell_embed.shape)
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  # [batch, out_channel, 53]
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) # [batch,53,64] or [batch,53,128]

        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
import torch
import torch.nn as nn

# ===== 通道注意力模块 =====
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 对每个通道求平均
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: [B, C, L]
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)   # [B, C]
#         y = self.fc(y).view(b, c, 1)      # [B, C, 1]
#         return x * y                      # 通道加权输出


import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 不融合通道注意力 =====
class ChannelAttention(nn.Module):
    """
    对每个通道独立加权，不做通道融合。
    输入: x [B, F, C]  输出: [B, F, C]
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, F, C]
        w = x.mean(dim=1)           # [B, C]  每个通道全局描述
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))  # [B, C]
        w = w.unsqueeze(1)               # [B, 1, C] 广播乘回原输入
        return x * w

# ===== 完整 CellCNN 模块 =====
class CellCNN1(nn.Module):
    def __init__(self, in_channel=6, feat_dim=128):
        super(CellCNN1, self).__init__()

        # 卷积参数
        max_pool_size = [2, 2, 6]
        drop_rate = 0.2
        kernel_size = [16, 16, 16]

        if in_channel == 6:
            in_channels = [6, 16, 32]
            out_channels = [16, 32, 64]
        else:
            raise ValueError("只支持 6 个通道组学输入")

        # 通道注意力
        self.channel_attn = ChannelAttention(in_channel)

        # 三层 Conv1d + ReLU + Dropout + MaxPool
        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(max_pool_size[0]),

            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(max_pool_size[1]),

            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        # 线性层压缩到 feat_dim
        self.cell_linear = nn.Linear(out_channels[2], feat_dim)

    def forward(self, x):
        # x: [B, F, C] 例如 [batch, 4079, 6]
        x = self.channel_attn(x)          # [B, F, C] 不融合通道
        x = x.transpose(1, 2)             # [B, C, F] 给 Conv1d
        x_cell_embed = self.cell_conv(x)  # [B, out_channels[-1], L']
        x_cell_embed = x_cell_embed.transpose(1, 2)  # [B, L', out_channels[-1]]
        x_cell_embed = self.cell_linear(x_cell_embed) # [B, L', feat_dim]
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


def drug_feat(drug_subs_codes, device, patch, length):
    # 获取药物特征 [bs, 2, 50]
    v = drug_subs_codes
    # 子结构序号  -> [bs, 50]
    subs = v[:, 0].long().to(device)
    # 子结构的掩码 [bs, 50]
    subs_mask = v[:, 1].long().to(device)
    # length < patch的话就需要填充 0    [bs, 1, 50] -> [bs, 165]
    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), 1)
        subs_mask = torch.cat((subs_mask, padding), 1)

    expanded_subs_mask = subs_mask.unsqueeze(1)
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()

    # 预测层

class predictor(torch.nn.Module):
    def __init__(self, out_channels):
        super(predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels, int(out_channels//2)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels//2), int(out_channels//4)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels//4), int(out_channels//8)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels//8), 1),
            )

    def forward(self, cell_embed, drug_embed):
        out = torch.cat((cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        # 是否回归
        out = torch.sigmoid(out.squeeze(dim=1))
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
