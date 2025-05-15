import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class LayerScale(nn.Module):
    """ LayerScale on tensors with channels in last-dim.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2d(nn.Module):
    """ LayerScale for tensors with torch 2D NCHW layout.
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., init_values=None, attn=None):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        if attn:
            self.msa = attn(feats, head=head, dropout=dropout)
        else:
            self.msa = MultiHeadSoftmaxSelfAttention(feats, head=head, dropout=dropout)
        self.ls1 = LayerScale(feats, init_values=init_values) if init_values else nn.Identity()# add layerscale here
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ls2 = LayerScale(feats, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        out = self.ls1(self.msa(self.la1(x))) + x
        out = self.ls2(self.mlp(self.la2(out))) + out
        return out

"""
traditional softmax attention
sigmoid attention
linear attention
doubly stochastic attention
cosine attention
logits??
"""

"""
https://towardsdatascience.com/linear-attention-is-all-you-need-5fa9c845c1b5
"""
class MultiHeadLinearSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., eps=1e-6):
        super(MultiHeadLinearSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        
        q = self.elu_feature_map(q)
        k = self.elu_feature_map(k)
        kv = torch.einsum("bnfi, bnfj->bnij", k, v)
        # i, j -> self.feats//self.head

        # Compute the normalizer
        z = 1/(torch.einsum("bnld,bnxd->bnlx", q, k.sum(dim=2, keepdim=True)))#+self.eps)
        # print(k.sum(dim=2, keepdim=True).shape)
        # print(z.shape) #torch.Size([1024, 12, 65])
        # Finally compute and return the new values
        v = torch.einsum("bnld,bndk->bnlk", q, kv)
        normed = torch.einsum("bnlk,bnlk->blnk", v, z)
        # print(v.shape) #torch.Size([1024, 12, 32])
        # print(normed.shape)
        return self.dropout(self.o(normed.flatten(2)))


""""
# (Softmax) Attention
out = softmax(q @ k.T / sqrt(d)) @ v

# Sigmoid Attention 
out = sigmoid(q @ k.T / sqrt(d) + b) @ v  # b: scalar
"""
class MultiHeadSigmoidSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSigmoidSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    """
    https://github.com/apple/ml-sigmoid-attention/tree/main/flash_sigmoid -> default 0 bias
    """
    def forward(self, x, bias=0.):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.sigmoid(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d + bias) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h) # transposes 1/2 back?
        # print(attn.shape) -> torch.Size([1024, 65, 12, 32])
        # flattens to [1024, 65, 384]
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class MultiHeadSoftmaxSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSoftmaxSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

"""
From paper: https://arxiv.org/abs/2110.11773
Implementation from https://github.com/michaelsdr/sinkformers/blob/main/vit-pytorch/vit_pytorch/sinkhorn.py
"""
class MultiHeadDoublyStochasticSelfAttention(nn.Module):

    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadDoublyStochasticSelfAttention, self).__init__()
        self.feats = feats
        self.head = head
        self.scale = (feats//head)**-0.5 # scaling factor for stable attention computation

        self.eps = 1
        self.max_iter = 3

        self.q = nn.Linear(feats, feats,bias=False)
        self.k = nn.Linear(feats, feats,bias=False)
        self.v = nn.Linear(feats, feats,bias=False)

        self.o = nn.Linear(feats,feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        # Compute scaled dot-product similarity
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply Sinkhorn normalization to compute doubly stochastic attention
        attention = self.sinkhorn_distance(dots)

        # Weighted sum of values
        out = torch.matmul(attention, v)

        # Combine heads and project back to original feature space
        out = out.transpose(1, 2).contiguous().view(b, n, f)
        out = self.o(out)
        return out

    def sinkhorn_distance(self,c):
        C = -c # higher similarity means lower cost
        batch_size, head, x_points, y_points = C.shape

            # Initialize mu and nu with the appropriate shape
        mu = torch.empty(batch_size, head, x_points, dtype=torch.float,
                        requires_grad=False, device=C.device).fill_(1.0 / x_points)
        nu = torch.empty(batch_size, head, y_points, dtype=torch.float,
                        requires_grad=False, device=C.device).fill_(1.0 / y_points)

        if mu.dim() < 2:
            mu = mu.view(-1, 1)

        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Stopping criterion
        thresh = 1e-6 # originally 1e-12

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) -
                                torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
                # v = v.detach().requires_grad_(False)
                # v[v == float('inf')] = 0.0
                # v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        return pi
    
    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    
class MultiHeadCosineSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        """
        Stabilized Cosine Attention Module.
        
        Args:
        - feats: Dimensionality of input features.
        - head: Number of attention heads.
        - dropout: Dropout rate.
        - init_m: Initial value of the learned stabilization parameter m.
        """
        super(MultiHeadCosineSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.head_dim = feats // head
        self.scale = self.head_dim**0.5

        # Learnable stabilization parameter m
        self.m = nn.Parameter(torch.full((head,), 0.5)) # m initialized to 0.5 as done in paper

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.head_dim).transpose(1, 2)

        # Normalize Q and K for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Compute cosine similarity
        sim = torch.einsum("bhid,bhjd->bhij", q, k)  # (b, h, n, n)

        # Stabilization scaling factor
        sigma_m = self.m.sigmoid().view(1, self.head, 1, 1)
        scaling_factor = sim.size(-1) ** sigma_m

        # Scale similarity scores
        sim_scaled = sim / scaling_factor

        # Apply attention dropout
        attn = self.dropout(sim_scaled)

        # Compute attention output
        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # (b, h, n, d)
        out = out.transpose(1, 2).contiguous().view(b, n, f)  # Merge heads
        out = self.o(out)
        return out

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



