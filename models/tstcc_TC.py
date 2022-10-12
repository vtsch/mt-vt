import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from bunch import Bunch
from einops import rearrange, repeat
import pandas as pd


######################
# Data Augmentations
######################


def DataTransform(sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Data augmentations, from https://arxiv.org/pdf/1706.00527.pdf
    Args:
        sample: sample from the dataset (1 patient)
    Returns:
        sample: weak and strong augmentations of the sample
    '''
    jitter_scale_ratio = 0.001
    jitter_ratio = 0.001
    weak_aug = scaling(sample, jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=3), jitter_ratio)

    return weak_aug, strong_aug


def jitter(x: np.ndarray, sigma=0.8) -> np.ndarray:
    '''
    for strong augmentation
    Args:
        x: sample from the dataset (1 patient)
        sigma: scaling factor
    Returns:
        x: jittered sample
    '''
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x: np.ndarray, sigma=1.1) -> np.ndarray:
    '''
    for weak augmentation
    Args:
        x: sample from the dataset (1 patient)
        sigma: scaling factor
    Returns:
        x: scaled sample
    '''
    factor = np.random.normal(
        loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x: np.ndarray, max_segments=3, seg_mode="random") -> np.ndarray:
    '''
    for strong augmentation
    Args:
        x: sample from the dataset (1 patient)
        max_segments: maximum number of segments to permute
        seg_mode: mode of segment selection, options: random, fixed
    Returns:
        x: permuted sample
    '''
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(6, num_segs[i], replace=True)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            #warp = np.hstack(x).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return ret


########################################################################################
# Transformer Layers
########################################################################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(
                    dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    # add token c to input whose state acts as a reprentative context vector in the output
    # apply features z to linear projection that maps features into hidden dimension
    # output is sent to transformer
    # attach context vector into the features vector such thtat the input features are now (c, z)
    # pass this trhough transformer layers, Attention and MLP
    # finally, re-attach context vector from final output
    # patch_size=self.num_channels+config.context_count_size, dim=config.hidden_dim
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)  # (bs, forward_Seq, hidden_dim)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t


########################################################################################
# Temporal Contrasting
########################################################################################


class TC(nn.Module):
    def __init__(self, config: Bunch) -> None:
        '''
        Temporal Contrastive Learning Module
        Args:
            config: configs
        '''
        super(TC, self).__init__()
        self.config = config
        self.num_channels = config.emb_size
        self.timestep = 3  # want to split max. 3 times here
        self.Wk = nn.ModuleList(
            [nn.Linear(config.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = config.device

        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.emb_size // 2),
            nn.BatchNorm1d(config.emb_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.emb_size // 2, config.emb_size // 4),
        )

        self.seq_transformer = Seq_Transformer(
            patch_size=self.num_channels, dim=config.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1: torch.Tensor, features_aug2: torch.Tensor, context: pd.DataFrame) -> torch.Tensor:
        '''
        Args:
            features_aug1: (bs, seq_len, emb_size)
            features_aug2: (bs, seq_len, emb_size)
            context: (bs, context_count_size)
        Returns:
            loss: (bs, 1)
            projection_head: (bs, emb_size // 4)
        '''
        if self.config.context:
            context = context.unsqueeze(1)
            context = context.repeat(1, features_aug1.shape[1], 1)
            # (batch_size, emb_size, seq_length + context_size)
            features_aug1 = torch.cat((features_aug1, context), dim=2)
            # (batch_size, emb_size, seq_length + context_size)
            features_aug2 = torch.cat((features_aug2, context), dim=2)
            #print("features_aug1 w c", features_aug1.shape)

        z_aug1 = features_aug1  # features are (batch_size, emb_size, seq_len)
        z_aug1 = z_aug1.transpose(1, 2)  # (batch_size, seq_len, emb_size)
        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(self.config.ts_length - self.timestep,
                                  size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty(
            (self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples +
                                           i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        # apply transformer
        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty(
            (self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        # contextual contrasting
        return nce, self.projection_head(c_t)
