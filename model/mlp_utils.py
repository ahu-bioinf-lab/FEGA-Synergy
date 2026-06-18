import torch
from torch import nn
from torch.nn import functional as F


# 替换 flash_attn.ops.activations.swiglu 的手动实现
# 这个函数将根据 Swish-Gated Linear Unit (SwiGLU) 的常见定义来操作。
# 它接收两个张量：gate 和 x。gate 部分会通过 Mish 激活函数处理，
# 然后与 x 进行元素级乘法。
def custom_swiglu(gate_tensor, x_tensor):
    """
    一个手动实现的 swiglu 函数替代。
    根据 Swish-Gated Linear Unit (SwiGLU) 的常见实现，
    门控部分通常通过 Swish (或 Mish) 激活函数处理，
    然后与另一部分进行元素级乘法。
    """
    return x_tensor * F.mish(gate_tensor)


class SmallMLP(nn.Module):
    def __init__(
            self,
            in_dim,
            inter_dim,
            out_dim,
            dropout_p=0.0,
            num_layers=2,
            use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                elif i != num_layers - 1:
                    self.mlp.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                else:
                    self.mlp.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class GatedMLPSingle(nn.Module):
    def __init__(
            self,
            in_dim,
            inter_dim,
            out_dim,
            dropout_p=0.0,
            use_ln=False,
    ):
        super().__init__()

        # Uncomment if you want dropout here
        # self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if self.use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

        # if dropout_p > 0:
        #    self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        if self.use_ln:
            y_pre_chunk = self.ln(self.fc1(x))  # Changed variable name for clarity
        else:
            y_pre_chunk = self.fc1(x)  # Changed variable name for clarity

        # y_value 和 gate_value 是分割后的两部分
        y_value, gate_value = y_pre_chunk.chunk(2, dim=-1)
        # 调用我们自定义的 swiglu 函数
        y = custom_swiglu(gate_value, y_value)

        # if self.dropout_p > 0:
        #    y = self.dropout(y)
        y = self.fc2(y)

        return y


class GatedMLPMulti(nn.Module):
    def __init__(
            self,
            in_dim,
            inter_dim,
            out_dim,
            dropout_p=0.0,
            num_layers=2,
            use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                # GatedMLPSingle 内部现在使用我们自定义的 swiglu
                GatedMLPSingle(in_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=False),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(GatedMLPSingle(in_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                elif i != num_layers - 1:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                else:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=use_ln))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

                # Note: The original code had Mish for num_layers > 1 loop outside the if/elif/else block for i,
                # meaning Mish was appended after every GatedMLPSingle *and* after dropout.
                # This seems like a potential bug or unconventional structure if num_layers is > 1
                # and dropout_p > 0, as you'd have dropout then Mish *then* another GatedMLPSingle in series.
                # I've kept it as per your original logic.
                self.mlp.append(nn.Mish())

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)