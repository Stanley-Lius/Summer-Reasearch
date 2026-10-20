import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from basicsr.archs.arch_util import LayerNorm2d
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.SWSformer_arch import PEG, FeedForward

try:
    from fla.ops.kda import chunk_kda
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass

from transformers.activations import ACT2FN


# ---------------------------------------------------------
# SGN (Semantic Guided Neighboring) Components from MambaIRv2
# ---------------------------------------------------------

def index_reverse(index):
    """
    Inverse permutation matching MambaIRv2 implementation.
    """
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    """
    Gathers tokens along the sequence dimension using index matching MambaIRv2.
    """
    dim = index.dim()
    assert x.shape[:dim] == index.shape, f"x ({x.shape}) and index ({index.shape}) shape incompatible"

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class SemanticRouter(nn.Module):
    """
    Semantic Router and Feature Modulation for SGN:
    1. Projects tokens to cluster probabilities (pred_route).
    2. Uses straight-through Gumbel-Softmax for discrete token assignment.
    3. Produces in-graph semantic centroid modulation to provide backward gradient
       flow from the primary pixel loss without modifying KDA.
    """
    def __init__(self, dim, num_tokens=64):
        super().__init__()
        self.num_tokens = num_tokens
        hidden_dim = max(dim // 3, 16)
        self.route = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
            nn.LogSoftmax(dim=-1)
        )
        self.prototype = nn.Embedding(num_tokens, dim)
        self.prototype.weight.data.uniform_(-1 / num_tokens, 1 / num_tokens)
        self.mod_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B_seq, seq_len, C)
        pred_route = self.route(x)  # (B_seq, seq_len, num_tokens)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # (B_seq, seq_len, num_tokens)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False)  # (B_seq, seq_len)
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # In-graph semantic modulation feature: ensures route network receives gradient updates from downstream loss
        semantic_feat = torch.matmul(cls_policy, self.prototype.weight)  # (B_seq, seq_len, C)
        mod = torch.tanh(self.mod_proj(semantic_feat))

        return x_sort_indices, x_sort_indices_reverse, mod


# ---------------------------------------------------------
# Core Attention Mechanisms (KDA remains completely UNCHANGED)
# ---------------------------------------------------------

class NoPoolBidirectionalKimiDeltaAttention(nn.Module):
    """
    Lite-Kimi-Linear (KDA) adapted for Vision:
    1. Pruned 1D ShortConvolutions (relies on PEG for 2D locality).
    2. Channel-Split Bidirectional Sweep: splits head_dim in half, forward handles D/2, backward handles D/2.
    3. Gating remains identical.
    NOTE: Kept 100% UNCHANGED to preserve mathematical integrity and kernel efficiency.
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

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=1e-6, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x_pooled, x_unpooled):
        B, sub_nW, nW, C = x_pooled.shape
        N = x_unpooled.shape[3]
        seq_len = nW * N

        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        
        # Directly use unpooled features for Q in KDA! No average pooling.
        q = F.silu(self.q_proj(x_unp_flat))
        k = F.silu(self.k_proj(x_unp_flat))
        v = F.silu(self.v_proj(x_unp_flat)) 

        # Reshape to sequence view
        q = q.view(B * sub_nW, seq_len, -1)
        k = k.view(B * sub_nW, seq_len, -1)
        v = v.view(B * sub_nW, seq_len, -1)
        hidden_states = x_unp_flat.view(B * sub_nW, seq_len, C)

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

        # Channel-Split Bidirectional Sweep
        out_f, _ = chunk_kda(q=q_f, k=k_f, v=v_f, g=g_f, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=True)

        out_b, _ = chunk_kda(
            q=torch.flip(q_b, dims=[1]),
            k=torch.flip(k_b, dims=[1]),
            v=torch.flip(v_b, dims=[1]),
            g=torch.flip(g_b, dims=[1]),
            beta=torch.flip(beta, dims=[1]),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True
        )
        out_b = torch.flip(out_b, dims=[1])

        # Concat the halves back together
        o = torch.cat([out_f, out_b], dim=-1)

        # Output Norm & Projection
        g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = rearrange(g_out, 'b t (h d) -> b t h d', d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, 'b t (h d) -> b t (h d)')
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class BidirectionalKimiMLAAttention(nn.Module):
    """
    Lite-MLA (Full Attention) adapted for Vision:
    1. Pruned 1D ShortConvolutions.
    2. Optimized Q-Pooling: Computes attention without redundant Q-expansion to save 64x FLOPs/VRAM.
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
        B, sub_nW, nW, C = x_pooled.shape
        N = x_unpooled.shape[3]
        seq_len = nW * N

        q = self.q_proj(x_pooled) # (B, sub_nW, nW, D)
        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        k = self.k_proj(x_unp_flat) # (B, sub_nW, seq_len, D)
        v = self.v_proj(x_unp_flat) # (B, sub_nW, seq_len, D)

        q = q.view(B * sub_nW, nW, -1)
        k = k.view(B * sub_nW, seq_len, -1)
        v = v.view(B * sub_nW, seq_len, -1)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # q @ k^T -> (B*sub_nW, H, nW, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # attn @ v -> (B*sub_nW, H, nW, D)
        o_pooled = torch.matmul(attn_weights, v)
        o_pooled = rearrange(o_pooled, 'b h t d -> b t (h d)') # (B*sub_nW, nW, D)
        
        # Expand output back to seq_len
        o = o_pooled.view(B, sub_nW, nW, 1, -1).expand(B, sub_nW, nW, N, -1)
        
        # Back to standard sequence shape
        o = o.reshape(B * sub_nW, seq_len, -1)
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


# ---------------------------------------------------------
# SGN Vision Attention Wrapper
# ---------------------------------------------------------

class SGNNoPoolCustomKimiVisionAttention(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, is_full_attn=False, num_tokens=64):
        super().__init__()
        self.win_size = win_size
        self.sub_win_size = sub_win_size
        self.is_full_attn = is_full_attn
        self.num_tokens = num_tokens

        if is_full_attn:
            self.attn = BidirectionalKimiMLAAttention(c, num_heads)
            self.router = None
        else:
            self.attn = NoPoolBidirectionalKimiDeltaAttention(c, num_heads)
            self.router = SemanticRouter(c, num_tokens=num_tokens)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        win_h, win_w = max(h // self.win_size, 1), max(w // self.win_size, 1)
        sub_win_h, sub_win_w = min(self.sub_win_size, win_h), min(self.sub_win_size, win_w)
        win_h += (sub_win_h - win_h % sub_win_h) % sub_win_h
        win_w += (sub_win_w - win_w % sub_win_w) % sub_win_w
        mod_pad_h, mod_pad_w = (win_h - h % win_h) % win_h, (win_w - w % win_w) % win_w
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x, (win_h, win_w), (sub_win_h, sub_win_w), x.shape[2] // win_h

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad, win_size, sub_win_size, nWh = self.check_image_size(x)

        # Tokenization & Shuffling (SWS mechanism)
        win_x = rearrange(x_pad, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_x = rearrange(win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_win_x = sub_win_x.transpose(1, 2)

        # x_unpooled: (B, sub_nW, nW, N, C)
        x_unpooled = shuffled_sub_win_x
        # x_pooled: (B, sub_nW, nW, C)
        x_pooled = x_unpooled.mean(dim=-2)

        if not self.is_full_attn and self.router is not None:
            # SGN Mechanism for KDA
            _, sub_nW, nW, N, _ = x_unpooled.shape
            seq_len = nW * N
            x_flat = x_unpooled.reshape(B * sub_nW, seq_len, C)

            # Predict semantic route and sort indices
            x_sort_indices, x_sort_indices_reverse, mod = self.router(x_flat)

            # SGN-unfold
            semantic_x_flat = semantic_neighbor(x_flat, x_sort_indices)
            semantic_x_unpooled = semantic_x_flat.view(B, sub_nW, nW, N, C)

            # Apply KDA with semantic ordering
            out = self.attn(x_pooled, semantic_x_unpooled)

            # SGN-fold
            out_flat = out.reshape(B * sub_nW, seq_len, C)
            out_restored_flat = semantic_neighbor(out_flat, x_sort_indices_reverse)

            # In-graph semantic modulation for downstream gradient flow back to self.router
            out_restored_flat = out_restored_flat + out_restored_flat * mod
            out = out_restored_flat.view(B, sub_nW, nW, N, C)
        else:
            # MLA Attention (Bypasses SGN to maintain local Q-pooling spatial semantics)
            out = self.attn(x_pooled, x_unpooled)

        # Unshuffle and reconstruct image
        out = out.transpose(1, 2) # B, nW, sub_nW, N, C
        out = rearrange(out, 'B nW (nWh nWw) (wH wW) C -> B nW (nWh wH) (nWw wW) C',
                        nWh=win_size[0] // sub_win_size[0], wH=sub_win_size[0], wW=sub_win_size[1])
        out = rearrange(out, 'B (nWh nWw) wH wW C -> B C (nWh wH) (nWw wW)',
                        nWh=nWh, wH=win_size[0], wW=win_size[1])

        return out[:, :, :H, :W]


# ---------------------------------------------------------
# SGN Block and Network
# ---------------------------------------------------------

class SGNLiteNoPoolKimiSWSBlock(nn.Module):
    def __init__(self, c, layer_idx, num_heads, win_size, sub_win_size, nwc_k, FFN_Expand=2.66, num_tokens=64):
        super().__init__()
        self.pos_block = PEG(c)
        is_full_attn = (layer_idx % 4 == 3) # 3 linear + 1 full
        self.attn = SGNNoPoolCustomKimiVisionAttention(c, num_heads, win_size, sub_win_size, is_full_attn, num_tokens=num_tokens)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k//2, groups=c, bias=True)
        self.gdfn = FeedForward(c, FFN_Expand, bias=False)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        y = inp + self.attn(self.pos_block(self.norm1(inp))) * self.beta
        y = y + self.nwc(self.norm2(y)) * self.alpha
        y = y + self.gdfn(self.norm3(y)) * self.gamma
        return y


@ARCH_REGISTRY.register()
class SGNLiteNoPoolKimiSWSformer(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 win_size=16, sub_win_size=8, nwc_ks=[7, 5, 3, 3, 3], num_heads=[1, 2, 4, 8, 16],
                 num_tokens=64):
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
            self.encoders.append(nn.Sequential(*[
                SGNLiteNoPoolKimiSWSBlock(chan, global_layer_idx+i, nH, win_size, sub_win_size, nwc_k, num_tokens=num_tokens)
                for i in range(nB)
            ]))
            global_layer_idx += nB
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[
            SGNLiteNoPoolKimiSWSBlock(chan, global_layer_idx+i, num_heads[-1], win_size, sub_win_size, nwc_ks[-1], num_tokens=num_tokens)
            for i in range(middle_blk_num)
        ])
        global_layer_idx += middle_blk_num

        for nB, nwc_k, nH in zip(dec_blk_nums, nwc_ks[-2::-1], num_heads[-2::-1]):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[
                SGNLiteNoPoolKimiSWSBlock(chan, global_layer_idx+i, nH, win_size, sub_win_size, nwc_k, num_tokens=num_tokens)
                for i in range(nB)
            ]))
            global_layer_idx += nB

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.intro(F.pad(inp, (0, (self.padder_size - W % self.padder_size) % self.padder_size, 0, (self.padder_size - H % self.padder_size) % self.padder_size)))
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = decoder(up(x) + enc_skip)
        return (self.ending(x) + inp)[:, :, :H, :W]
