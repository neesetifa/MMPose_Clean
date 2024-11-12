import torch
import torch.nn as nn
from torch import distributions


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the official implementation of RLE
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    See also `real-nvp-pytorch
    <https://github.com/senya-ashukha/real-nvp-pytorch>`_.
    """

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()

        self.register_buffer('loc', torch.zeros(2))
        self.register_buffer('cov', torch.eye(2))
        self.register_buffer('mask', torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.get_trans_net() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    # ==== Encoding Part, for training ====
    # Reverse process, x->z0
    # This is for training
    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""
        # z[1:d] = x[1:d]
        # z[d:n] = (x[d:n] - t(x[1:d])) * exp(-s(x[1:d]))
        # det_J = Σ(s(x[1:d]))
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    # Loss function, calculate a log value 
    # This is for training
    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det

    # ==== Decoding Part, for generation ====
    # Sample a z from Normal Distribution, then send to forward_p() to generate images
    # RLE里没有用到该函数
    def sample(self, batchSize):
        # batchSize: num of images you want to generate
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    # Forward process, z0->x
    # RLE里没有用到该函数
    def forward_p(self, z):
        # x[1:d] = z[1:d]
        # x[d:n] = z[d:n] * exp(s(z[1:d])) + t(z[1:d])
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    # RLE里没有用到该函数
    def forward(self, x):
        return self.log_prob(x)
