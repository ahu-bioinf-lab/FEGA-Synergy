import torch
import admin_torch

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch_edge_index

from .norm_layers import BN, LN
from .ESA_KIT import SAB, PMA
from .mlp_utils import SmallMLP, GatedMLPMulti
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_batch

class SABComplete1(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="torch",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            if self.pre_or_post == "post":
                self.norm = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm = LN(dim_in, num_elements=self.set_max_items)
        elif norm_type == "BN":
            self.norm = BN(dim_out)

        self.mlp_type = mlp_type
        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out, out_dim=dim_out, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out, out_dim=dim_out, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
        elif norm_type == "BN":
            self.norm_mlp = BN(dim_out)

    def forward(self, X):
        if self.pre_or_post == "pre":
            X = self.norm(X)

        out_attn = self.sab(X)

        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn
        elif self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out
            elif self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out

class SABComplete(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="torch",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            if self.pre_or_post == "post":
                self.norm = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm = LN(dim_in, num_elements=self.set_max_items)
        elif norm_type == "BN":
            self.norm = BN(dim_out)

        self.mlp_type = mlp_type
        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out, out_dim=dim_out, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out, out_dim=dim_out, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
        elif norm_type == "BN":
            self.norm_mlp = BN(dim_out)

    def forward(self, X,):
        if self.pre_or_post == "pre":
            X = self.norm(X)

        # device = next(self.parameters()).device  # 自动获取模型所在设备（比如 cuda:0）#TODO add
        # dtype = torch.float32  # 或模型默认类型（如 torch.float16）

        # # 确保 X 和 adj_mask 与模型设备、类型完全一致
        # X = X.to(device=device, dtype=dtype)
        # if adj_mask is not None:
        #     adj_mask = adj_mask.to(device=device, dtype=dtype)

        out_attn = self.sab(X)

        # 核心修正：将 out_attn 从 [batch_size, 1, dim_out] 压缩为 [batch_size, dim_out]
        # 以便与 X 进行正确的残差连接
        out_attn_squeezed = out_attn.squeeze(1)

        if out_attn_squeezed.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn_squeezed # 使用压缩后的张量进行加法
        elif self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn_squeezed) # 同样使用压缩后的
            out = self.norm(out)

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out
            elif self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out


# --- 3. PMAComplete (多头注意力池化) 模块 (保持不变) ---
class PMAComplete(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type,
        dropout=0,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="torch",
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(PMAComplete, self).__init__()

        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.residual_dropout = residual_dropout
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm = BN(dim_hidden)

        self.mlp_type = mlp_type
        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden, out_dim=dim_hidden, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden, out_dim=dim_hidden, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm_mlp = BN(dim_hidden)

    def forward(self, X_edge_features, edge_index, batch_mapping):
        src_nodes_batch_id = batch_mapping.index_select(0, edge_index[0])
        src_nodes_batch_id = batch_mapping#更改后

        if self.pre_or_post == "pre":
            X_edge_features = self.norm(X_edge_features)
        '''print(f"to_dense_batch 输入 - src_nodes_batch_id 类型: {src_nodes_batch_id.dtype}")
        print(f"to_dense_batch 输入 - src_nodes_batch_id 设备: {src_nodes_batch_id.device}")
        print(f"to_dense_batch 输入 - src_nodes_batch_id 唯一元素数量: {torch.unique(src_nodes_batch_id).numel()}")'''

        pooled_input, _ = to_dense_batch(
            x=X_edge_features,
            batch=src_nodes_batch_id,
            max_num_nodes=None#zuidajiedianshu
        )

        out_attn = self.pma(pooled_input)

        if self.pre_or_post == "pre" and out_attn.shape[-2] == pooled_input.shape[-2]:
            out = out_attn
        elif self.pre_or_post == "post" and out_attn.shape[-2] == pooled_input.shape[-2]:
            out = self.residual_attn(pooled_input, out_attn)
            out = self.norm(out)
        else:
            out = out_attn

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = out_mlp + out
            elif self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out
    
class PMAComplete1(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type="LN",
        dropout=0.1,
        use_mlp=True,
        mlp_hidden_size=64,
        mlp_type="gated_mlp",
        xformers_or_torch_attn="torch",
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(PMAComplete1, self).__init__()

        self.use_mlp = use_mlp
        self.pre_or_post = pre_or_post
        self.residual_dropout = residual_dropout

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            self.norm = LN(dim_hidden)
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm = BN(dim_hidden)
            self.norm_mlp = BN(dim_hidden)

        if self.use_mlp:
            if mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden, out_dim=dim_hidden, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )
            elif mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden, out_dim=dim_hidden, inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout, num_layers=num_mlp_layers, use_ln=use_mlp_ln,
                )

    def forward(self, X_edge_features):
        # X_edge_features: [E, L, D]
        if self.pre_or_post == "pre":
            X_edge_features = self.norm(X_edge_features)

        out = self.pma(X_edge_features)  # [E, num_outputs, D]

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape == out_mlp.shape:
                    out = out + out_mlp
            elif self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                out = self.norm_mlp(out + out_mlp)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out


