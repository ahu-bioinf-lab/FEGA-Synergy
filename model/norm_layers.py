from torch import nn

class BatchRenorm(nn.Module):
    def __init__(self, dim, eps=1e-8, momentum=0.1):
        super(BatchRenorm, self).__init__()

        self.dim = dim

        self.bn = BatchRenorm1d(num_features=dim, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


class BN(nn.Module):
    def __init__(self, dim, num_elements=None):
        super(BN, self).__init__()

        # self.bn = nn.BatchNorm1d(dim, eps=1e-8, momentum=0.1)
        self.bn = BatchRenorm1d(dim, eps=1e-8, momentum=0.1)

    def forward(self, x):
        return self.bn(x)


class LN(nn.Module):
    def __init__(self, hidden_dim, num_elements=None):
        super(LN, self).__init__()

        # if num_elements is not None:
        #     self.ln = nn.LayerNorm([num_elements, hidden_dim], eps=1e-8)
        # else:
        #     self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

    def forward(self, x):
        return self.ln(x)

import torch


class BatchRenorm1(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        """
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)

        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean)) / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm1):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm1):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm1):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
