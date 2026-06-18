import math
import torch
from torch import nn
from ..model_utiles import LayerNorm
import torch.nn.functional as F


class predictor(torch.nn.Module):
    def __init__(self, out_channels):
        super(predictor, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(out_channels, int(out_channels//2)),
            # nn.Mish(),
            # nn.Dropout(p=0.5),
            nn.Linear(int(out_channels), out_channels//4),
            nn.GELU(),
            nn.Dropout(p=0.3),
            #nn.Linear(int(out_channels//4),1),
            # nn.GELU(),
            # nn.Dropout(p=0.3),
            nn.Linear(int(out_channels//4), int(out_channels//8)),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(out_channels//8), 1),
            )

    def forward(self, cell_embed, drug_embed):
        out = torch.cat((cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        # 是否回归
        out = torch.sigmoid(out.squeeze(dim=1))
        #out = out.squeeze(dim=1)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力的两个全连接层
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (B, F, C)
        w = x.mean(dim=1)              # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)) # (B, C)
        w = w.unsqueeze(1)             # (B, 1, C)
        return x * w                   # 广播到 (B, F, C)

class Fusion0(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim)  # 层归一化
        self.ca = ChannelAttention(channels=2, reduction=1)  # 因为你只有2个通道

    def forward(self, drugA_struct, drugA_sema_repr):
        # 归一化
        a = self.ln(drugA_struct)
        b = self.ln(drugA_sema_repr)
        
        # 将两个通道堆叠成 (B, F, 2)
        x = torch.stack([a, b], dim=-1)   # (B, F, 2)
        # a=a.unsqueeze(1)
        # x=torch.cat([a,b], dim=1)
        # x=x.permute(0, 2, 1)
        
        # 通道注意力加权
        x = self.ca(x)                    # (B, F, 2)
        
        # 输出：经过加权后的 (B, F)
        #out = x.mean(dim=-1)              # 聚合两个通道 (B, F)
        #out = x.max(dim=-1).values # [B, F]
        out=x
        
        return out

class Fusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, feat_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_struct, x_sema):
        x = torch.cat([x_struct, x_sema], dim=-1)  # (B, 2F)

        g = self.sigmoid(self.fc(x))               # (B, F)

        out = g * x_struct + (1 - g) * x_sema      # (B, F)

        return out
