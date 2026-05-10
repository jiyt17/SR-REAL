import math
import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class PEConfig(PretrainedConfig):
    model_type = "spatial_pe"

    def __init__(self, pe_type: str = None, **kwargs):
        super().__init__()
        self.pe_type = pe_type


class PosBiasAttention(nn.Module):
    def __init__(self, dim, num_heads=8, pos_mlp_hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.pos_mlp = nn.Sequential(nn.Linear(3, pos_mlp_hidden_dim), nn.ReLU(), nn.Linear(pos_mlp_hidden_dim, 1))

    def forward(self, xyz, feats):
        B, N, C = feats.shape

        Q = (
            self.q_proj(feats).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        )  # (B, heads, N, head_dim)
        K = self.k_proj(feats).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        V = self.v_proj(feats).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Positional bias
        xyz_diff = xyz[:, :, None, :] - xyz[:, None, :, :]  # (B, N, N, 3)
        pos_bias = self.pos_mlp(xyz_diff).squeeze(-1)  # (B, N, N)

        attn_scores = attn_scores + pos_bias.unsqueeze(1)  # broadcast to (B, heads, N, N)

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = attn_probs @ V  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        return self.out_proj(out)


class Abs3DPositionEmbeddingMLP(nn.Module):
    def __init__(self, feature_dim=768, in_channels=3, n_freqs=8, logscale=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)

        self.register_buffer("freq_bands", freq_bands, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.001)

    def frequency_encoding(self, xyz):
        r"""
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz = xyz.to(self.freq_bands.device)
        xyz_n = (xyz).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        # TODO: encoding with 3D position
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding


class PE(PreTrainedModel):
    config_class = PEConfig

    def __init__(self, pe_cfg: PEConfig, config: PretrainedConfig):
        super().__init__(pe_cfg)
        pe_type = pe_cfg.pe_type
        self.pe_type = pe_type

        if config.dynamic_s2 and config.image_aspect_ratio == "dynamic_s2":
            raise ValueError("Dynamic S2 is not supported for PE")
        else:
            feature_dim = config.mm_hidden_size

        if pe_type in ["abs_mlp", "srgpt_local", "srgpt_global+local"]:
            self.layers = Abs3DPositionEmbeddingMLP(feature_dim)
        elif pe_type == "pos_bias_attn":
            self.layers = PosBiasAttention(feature_dim)
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)
