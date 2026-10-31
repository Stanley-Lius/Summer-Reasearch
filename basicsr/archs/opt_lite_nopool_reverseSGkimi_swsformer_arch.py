import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.archs.arch_util import LayerNorm2d
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.SWSformer_arch import PEG

try:
    from fla.ops.kda import chunk_kda
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NoPoolBidirectionalKimiDeltaAttention(nn.Module):
    """
    Optimized Lite-Kimi-Linear (KDA) adapted for Vision:
    1. Pruned 1D ShortConvolutions (relies on PEG for 2D locality).
    2. Fused QKV and FGB projections to minimize HBM memory reads and CUDA kernel launch overhead.
    3. Channel-Split Bidirectional Sweep: splits head_dim in half with independent forward/backward beta gates.
    """
    def __init__(self, c, num_heads):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads

        projection_size = self.head_dim * self.num_heads
        self.projection_size = projection_size

        # Fused Q, K, V linear projection
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * projection_size, bias=False)

        self.A_log = torch.nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1)
        )

        # Fused f_a, g_a, and bidirectional beta (2 * num_heads) low-rank/gate input projection
        self.fgb_proj = nn.Linear(self.hidden_size, 2 * self.head_dim + 2 * self.num_heads, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))

        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=1e-6, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x_pooled, x_unpooled):
        B, sub_nW, nW, N, C = x_unpooled.shape
        seq_len = nW * N

        hidden_states = x_unpooled.reshape(B * sub_nW, seq_len, C)

        # 1. Fused Q, K, V projection + SiLU
        qkv = F.silu(self.qkv_proj(hidden_states))
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Fused Gates (f_a, g_a, bidirectional beta)
        f_a, g_a, beta_raw = torch.split(
            self.fgb_proj(hidden_states),
            [self.head_dim, self.head_dim, 2 * self.num_heads],
            dim=-1
        )

        g = self.f_b_proj(f_a)
        g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)
        A_log_flat = self.A_log.view(self.num_heads)
        g = fused_kda_gate(g, A_log_flat, dt_bias=self.dt_bias)

        beta = beta_raw.float().sigmoid()
        beta_f, beta_b = beta.chunk(2, dim=-1)

        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # 3. Channel Split (Split head_dim in half)
        q_f, q_b = q.chunk(2, dim=-1)
        k_f, k_b = k.chunk(2, dim=-1)
        v_f, v_b = v.chunk(2, dim=-1)
        g_f, g_b = g.chunk(2, dim=-1)

        # 4. Channel-Split Bidirectional Sweep
        out_f, _ = chunk_kda(
            q=q_f, k=k_f, v=v_f, g=g_f, beta=beta_f,
            output_final_state=False, use_qk_l2norm_in_kernel=True
        )

        out_b, _ = chunk_kda(
            q=torch.flip(q_b, dims=[1]),
            k=torch.flip(k_b, dims=[1]),
            v=torch.flip(v_b, dims=[1]),
            g=torch.flip(g_b, dims=[1]),
            beta=torch.flip(beta_b, dims=[1]),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True
        )
        out_b = torch.flip(out_b, dims=[1])

        # Concat the halves back together
        o = torch.cat([out_f, out_b], dim=-1)

        # 5. Output Norm & Projection
        g_out = self.g_b_proj(g_a)
        g_out = rearrange(g_out, 'b t (h d) -> b t h d', d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class BidirectionalKimiMLAAttention(nn.Module):
    """
    Optimized Lite-MLA (Full Softmax Attention) adapted for Vision:
    1. Pruned 1D ShortConvolutions.
    2. Fused KV projection and PyTorch native SDPA (FlashAttention / Memory-Efficient Attention).
    3. Projects pooled output `o_pooled` BEFORE `.expand()`, reducing `o_proj` FLOPs by N-fold (e.g. 64x).
    """
    def __init__(self, c, num_heads):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads

        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 2 * projection_size, bias=False)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(self, x_pooled, x_unpooled):
        B, sub_nW, nW, N, C = x_unpooled.shape
        seq_len = nW * N
        if x_pooled is None:
            x_pooled = x_unpooled.mean(dim=-2)

        q = self.q_proj(x_pooled)  # (B, sub_nW, nW, D)
        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        k, v = self.kv_proj(x_unp_flat).chunk(2, dim=-1)  # (B, sub_nW, seq_len, D)

        q = rearrange(q, 'B s t (h d) -> (B s) h t d', h=self.num_heads)
        k = rearrange(k, 'B s t (h d) -> (B s) h t d', h=self.num_heads)
        v = rearrange(v, 'B s t (h d) -> (B s) h t d', h=self.num_heads)

        # Exact softmax attention via FlashAttention / SDPA
        o_pooled = F.scaled_dot_product_attention(q, k, v)
        o_pooled = rearrange(o_pooled, '(B s) h t d -> B s t (h d)', B=B)

        # Apply o_proj BEFORE expanding along N to save N-fold FLOPs and memory copy
        o_pooled = self.o_proj(o_pooled)  # (B, sub_nW, nW, C)
        o = o_pooled.unsqueeze(-2).expand(B, sub_nW, nW, N, C)

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
            pad_mode = 'reflect' if (mod_pad_w < w and mod_pad_h < h) else 'replicate'
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=pad_mode)
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
    def __init__(self, c, layer_idx, num_heads, win_size, sub_win_size, nwc_k, FFN_Expand=2):
        super().__init__()
        self.pos_block = PEG(c)
        is_full_attn = (layer_idx % 4 == 0)  # 3 linear + 1 full
        self.attn = NoPoolCustomKimiVisionAttention(c, num_heads, win_size, sub_win_size, is_full_attn)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k // 2, groups=c, bias=True)

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        y = inp + self.attn(self.pos_block(self.norm1(inp))) * self.beta
        y = y + self.nwc(self.norm2(y)) * self.alpha
        x = self.conv4(self.norm3(y))
        x = self.sg(x)
        x = self.conv5(x)
        y = y + x * self.gamma
        return y


@ARCH_REGISTRY.register()
class OptLiteNoPoolReverseSGKimiSWSformer(nn.Module):
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
        if pad_h > 0 or pad_w > 0:
            pad_mode = 'reflect' if (pad_w < W and pad_h < H) else 'replicate'
            inp_padded = F.pad(inp, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            inp_padded = inp

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
