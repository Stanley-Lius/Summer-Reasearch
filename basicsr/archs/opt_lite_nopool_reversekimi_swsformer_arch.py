import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math

from basicsr.archs.arch_util import LayerNorm2d
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.softmax_util import softmax
from basicsr.archs.SWSformer_arch import PEG, FeedForward

try:
    from fla.ops.kda import chunk_kda
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass

from transformers.activations import ACT2FN


class NoPoolBidirectionalKimiDeltaAttention(nn.Module):
    """
    Lite-Kimi-Linear (KDA) adapted for Vision:
    1. Pruned 1D ShortConvolutions (relies on PEG for 2D locality).
    2. Channel-Split Bidirectional Sweep: fused into a single batch-concatenated chunk_kda call
       (bit-for-bit identical to two separate calls, halving Triton kernel launch overhead).
    3. Gating and parameter structure remain 100% identical to the original architecture.
    """
    def __init__(self, c, num_heads):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads

        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.A_log = torch.nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1)
        )

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=1e-6, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x_pooled, x_unpooled):
        B, sub_nW, nW, N, C = x_unpooled.shape
        seq_len = nW * N

        hidden_states = x_unpooled.reshape(B * sub_nW, seq_len, C)

        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v = F.silu(self.v_proj(hidden_states))

        # Gates
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)

        A_log_flat = self.A_log.view(self.num_heads)
        g = fused_kda_gate(g, A_log_flat, dt_bias=self.dt_bias)

        beta = self.b_proj(hidden_states).float().sigmoid()

        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Channel Split (Split head_dim in half)
        q_f, q_b = q.chunk(2, dim=-1)
        k_f, k_b = k.chunk(2, dim=-1)
        v_f, v_b = v.chunk(2, dim=-1)
        g_f, g_b = g.chunk(2, dim=-1)

        # Single-Call Channel-Split Bidirectional Sweep (contiguous batch-concatenation)
        q_bi = torch.cat([q_f, torch.flip(q_b, dims=[1])], dim=0)
        k_bi = torch.cat([k_f, torch.flip(k_b, dims=[1])], dim=0)
        v_bi = torch.cat([v_f, torch.flip(v_b, dims=[1])], dim=0)
        g_bi = torch.cat([g_f, torch.flip(g_b, dims=[1])], dim=0)
        beta_bi = torch.cat([beta, torch.flip(beta, dims=[1])], dim=0)

        out_bi, _ = chunk_kda(
            q=q_bi,
            k=k_bi,
            v=v_bi,
            g=g_bi,
            beta=beta_bi,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_f, out_b = out_bi.chunk(2, dim=0)
        out_b = torch.flip(out_b, dims=[1])

        # Concat the halves back together
        o = torch.cat([out_f, out_b], dim=-1)

        # Output Norm & Projection
        g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = rearrange(g_out, 'b t (h d) -> b t h d', d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class BidirectionalKimiMLAAttention(nn.Module):
    """
    Lite-MLA (Full Attention) adapted for Vision:
    1. Pruned 1D ShortConvolutions.
    2. Optimized Q-Pooling & Output Projection: Computes attention using custom Softmax-Collapse-Safe
       `softmax` and applies `o_proj` BEFORE expanding across N tokens to save N-fold FLOPs/VRAM.
    """
    def __init__(self, c, num_heads):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads

        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.scale = self.head_dim ** -0.5
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x_pooled, x_unpooled):
        B, sub_nW, nW, N, C = x_unpooled.shape
        seq_len = nW * N
        if x_pooled is None:
            x_pooled = x_unpooled.mean(dim=-2)

        q = self.q_proj(x_pooled)  # (B, sub_nW, nW, D)
        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        k = self.k_proj(x_unp_flat)  # (B, sub_nW, seq_len, D)
        v = self.v_proj(x_unp_flat)  # (B, sub_nW, seq_len, D)

        q = q.view(B * sub_nW, nW, -1)
        k = k.view(B * sub_nW, seq_len, -1)
        v = v.view(B * sub_nW, seq_len, -1)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # q @ k^T -> (B*sub_nW, H, nW, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = softmax(attn_weights, dim=-1)

        # attn @ v -> (B*sub_nW, H, nW, D)
        o_pooled = torch.matmul(attn_weights, v)
        o_pooled = rearrange(o_pooled, 'b h t d -> b t (h d)')  # (B*sub_nW, nW, D)

        # Project BEFORE expanding to N tokens (mathematically identical, N-fold fewer FLOPs)
        o_pooled = self.o_proj(o_pooled)  # (B*sub_nW, nW, C)
        o = o_pooled.view(B, sub_nW, nW, 1, C).expand(B, sub_nW, nW, N, C)

        return o


class NoPoolCustomKimiVisionAttention(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, is_full_attn=False):
        super().__init__()
        self.win_size = win_size
        self.sub_win_size = sub_win_size
        self.is_full_attn = is_full_attn

        if is_full_attn:
            self.attn = BidirectionalKimiMLAAttention(c, num_heads)
        else:
            self.attn = NoPoolBidirectionalKimiDeltaAttention(c, num_heads)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        win_h, win_w = max(h // self.win_size, 1), max(w // self.win_size, 1)
        sub_win_h, sub_win_w = min(self.sub_win_size, win_h), min(self.sub_win_size, win_w)
        win_h += (sub_win_h - win_h % sub_win_h) % sub_win_h
        win_w += (sub_win_w - win_w % sub_win_w) % sub_win_w
        mod_pad_h, mod_pad_w = (win_h - h % win_h) % win_h, (win_w - w % win_w) % win_w
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x, (win_h, win_w), (sub_win_h, sub_win_w), x.shape[2] // win_h

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad, win_size, sub_win_size, nWh = self.check_image_size(x)

        sH = win_size[0] // sub_win_size[0]
        sW = win_size[1] // sub_win_size[1]

        # Single-step SWS tokenization & shuffling: (B, sub_nW, nW, N, C)
        x_unpooled = rearrange(
            x_pad,
            'B C (nWh sH wH) (nWw sW wW) -> B (sH sW) (nWh nWw) (wH wW) C',
            sH=sH, sW=sW, wH=sub_win_size[0], wW=sub_win_size[1]
        )
        # Only compute x_pooled when full attention (MLA) needs it
        x_pooled = x_unpooled.mean(dim=-2) if self.is_full_attn else None

        # Apply customized Kimi Attention -> (B, sub_nW, nW, N, C)
        out = self.attn(x_pooled, x_unpooled)

        # Single-step unshuffle and reconstruct image
        out = rearrange(
            out,
            'B (sH sW) (nWh nWw) (wH wW) C -> B C (nWh sH wH) (nWw sW wW)',
            sH=sH, sW=sW, nWh=nWh, wH=sub_win_size[0], wW=sub_win_size[1]
        )

        return out[:, :, :H, :W]


class LiteNoPoolKimiSWSBlock(nn.Module):
    def __init__(self, c, layer_idx, num_heads, win_size, sub_win_size, nwc_k, FFN_Expand=2.66):
        super().__init__()
        self.pos_block = PEG(c)
        is_full_attn = (layer_idx % 4 == 0)  # 3 linear + 1 full
        self.attn = NoPoolCustomKimiVisionAttention(c, num_heads, win_size, sub_win_size, is_full_attn)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k // 2, groups=c, bias=True)
        self.gdfn = FeedForward(c, FFN_Expand, bias=False)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def _ffn_forward(self, y):
        return self.gdfn(self.norm3(y))

    def forward(self, inp):
        y = inp + self.attn(self.pos_block(self.norm1(inp))) * self.beta
        y = y + self.nwc(self.norm2(y)) * self.alpha
        if self.training and y.requires_grad:
            ffn_out = checkpoint(self._ffn_forward, y, use_reentrant=False)
        else:
            ffn_out = self._ffn_forward(y)
        y = y + ffn_out * self.gamma
        return y


@ARCH_REGISTRY.register()
class OptLiteNoPoolReverseKimiSWSformer(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 win_size=16, sub_win_size=8, nwc_ks=[7, 5, 3, 3, 3], num_heads=[1, 2, 4, 8, 16]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        global_layer_idx = 0
        chan = width
        for nB, nwc_k, nH in zip(enc_blk_nums, nwc_ks, num_heads):
            self.encoders.append(nn.Sequential(*[LiteNoPoolKimiSWSBlock(chan, global_layer_idx + i, nH, win_size, sub_win_size, nwc_k) for i in range(nB)]))
            global_layer_idx += nB
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[LiteNoPoolKimiSWSBlock(chan, global_layer_idx + i, num_heads[-1], win_size, sub_win_size, nwc_ks[-1]) for i in range(middle_blk_num)])
        global_layer_idx += middle_blk_num

        for nB, nwc_k, nH in zip(dec_blk_nums, nwc_ks[-2::-1], num_heads[-2::-1]):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[LiteNoPoolKimiSWSBlock(chan, global_layer_idx + i, nH, win_size, sub_win_size, nwc_k) for i in range(nB)]))
            global_layer_idx += nB

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        inp_padded = F.pad(inp, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else inp

        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = decoder(up(x) + enc_skip)
        return self.ending(x)[:, :, :H, :W] + inp
