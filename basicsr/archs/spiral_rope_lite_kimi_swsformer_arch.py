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
    from fla.modules import FusedRMSNormGated
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass


class SpiralRoPE2D(nn.Module):
    def __init__(self, dim, num_heads, base_dirs=8, dir_indices=None, min_freq=math.pi/16, max_freq=4*math.pi):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.base_dirs = base_dirs
        
        if dir_indices is None:
            dir_indices = list(range(base_dirs))
        self.dir_indices = dir_indices
        self.num_used_dirs = len(dir_indices)
        
        self.freqs_per_dir = (dim // 2) // self.num_used_dirs
        assert (dim // 2) % self.num_used_dirs == 0, f"dim/2 ({dim//2}) must be divisible by num_used_dirs ({self.num_used_dirs})"
        
        all_dirs = torch.arange(base_dirs).float() * (math.pi / base_dirs)
        if base_dirs > 1:
            all_freqs = min_freq * (max_freq / min_freq) ** (torch.arange(base_dirs).float() / (base_dirs - 1))
        else:
            all_freqs = torch.tensor([min_freq])
            
        self.register_buffer("all_dirs", all_dirs, persistent=False)
        self.register_buffer("all_freqs", all_freqs, persistent=False)
        
    def _get_freqs(self, h):
        m = torch.arange(self.base_dirs)
        idx = (m + 2 * h) % self.base_dirs
        return self.all_freqs[idx]

    def forward(self, x, grid_y, grid_x):
        freqs = torch.stack([self._get_freqs(h) for h in range(self.num_heads)]) # [H, base_dirs]
        
        th = self.all_dirs[self.dir_indices] # [used_dirs]
        freqs = freqs[:, self.dir_indices] # [H, used_dirs]
        
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)
        
        x_proj = grid_x.unsqueeze(-1) * cos_th
        y_proj = grid_y.unsqueeze(-1) * sin_th
        proj = x_proj + y_proj # [..., used_dirs]
        
        phase = proj.unsqueeze(-2) * freqs # [..., H, used_dirs]
        phase = phase.repeat_interleave(self.freqs_per_dir, dim=-1) # [..., H, dim/2]
        phase = phase.repeat_interleave(2, dim=-1) # [..., H, dim]
        
        cos = torch.cos(phase).to(x.dtype)
        sin = torch.sin(phase).to(x.dtype)
        
        x_reshaped = x.unflatten(-1, (self.dim // 2, 2))
        x1, x2 = x_reshaped.unbind(-1)
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        
        return x * cos + x_rot * sin


class BidirectionalKimiDeltaAttention(nn.Module):
    def __init__(self, c, num_heads, rope_in_kda=True, rope_min_freq=math.pi/16, rope_max_freq=4*math.pi):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.rope_in_kda = rope_in_kda

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
        
        if self.rope_in_kda:
            full_dim = self.head_dim
            K = 1
            for k in range(4, 0, -1):
                if full_dim % (4 * k) == 0:
                    K = k
                    break
            base_dirs = 2 * K
            fwd_indices = list(range(0, base_dirs, 2))
            bwd_indices = list(range(1, base_dirs, 2))
            dim_per_half = full_dim // 2
            
            self.rope_fwd = SpiralRoPE2D(dim_per_half, num_heads, base_dirs=base_dirs, dir_indices=fwd_indices, min_freq=rope_min_freq, max_freq=rope_max_freq)
            self.rope_bwd = SpiralRoPE2D(dim_per_half, num_heads, base_dirs=base_dirs, dir_indices=bwd_indices, min_freq=rope_min_freq, max_freq=rope_max_freq)

    def forward(self, x_pooled, x_unpooled, q_grid_y, q_grid_x, kv_grid_y, kv_grid_x):
        B, sub_nW, nW, C = x_pooled.shape
        N = x_unpooled.shape[3]
        seq_len = nW * N

        q = self.q_proj(x_pooled) 
        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        k = self.k_proj(x_unp_flat) 
        v = self.v_proj(x_unp_flat) 

        q = q.unsqueeze(3).expand(B, sub_nW, nW, N, -1)

        g = self.f_b_proj(self.f_a_proj(x_unp_flat))
        g = rearrange(g, 'b s t (h d) -> (b s) t h d', h=self.num_heads)
        
        A_log_flat = self.A_log.view(self.num_heads)
        g = fused_kda_gate(g, A_log_flat, dt_bias=self.dt_bias)
        g = rearrange(g, '(b s) t h d -> b s t h d', b=B, s=sub_nW)
        
        beta = self.b_proj(x_unp_flat).float().sigmoid()

        q = rearrange(q, 'b s w n (h d) -> b s w n h d', h=self.num_heads)
        k = rearrange(k, 'b s t (h d) -> b s t h d', h=self.num_heads)
        v = rearrange(v, 'b s t (h d) -> b s t h d', h=self.num_heads)

        q_f, q_b = q.chunk(2, dim=-1)
        k_f, k_b = k.chunk(2, dim=-1)
        v_f, v_b = v.chunk(2, dim=-1)
        g_f, g_b = g.chunk(2, dim=-1)

        if self.rope_in_kda:
            kv_grid_y_flat = kv_grid_y.reshape(1, sub_nW, seq_len)
            kv_grid_x_flat = kv_grid_x.reshape(1, sub_nW, seq_len)
            q_f = self.rope_fwd(q_f, q_grid_y, q_grid_x)
            q_b = self.rope_bwd(q_b, q_grid_y, q_grid_x)
            k_f = self.rope_fwd(k_f, kv_grid_y_flat, kv_grid_x_flat)
            k_b = self.rope_bwd(k_b, kv_grid_y_flat, kv_grid_x_flat)

        q_f = q_f.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        k_f = k_f.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        v_f = v_f.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        g_f = g_f.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        
        q_b = q_b.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        k_b = k_b.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        v_b = v_b.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        g_b = g_b.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        
        beta_flat = beta.view(B * sub_nW, seq_len, self.num_heads)

        out_f, _ = chunk_kda(q=q_f, k=k_f, v=v_f, g=g_f, beta=beta_flat, output_final_state=False, use_qk_l2norm_in_kernel=True)

        out_b, _ = chunk_kda(
            q=torch.flip(q_b, dims=[1]),
            k=torch.flip(k_b, dims=[1]),
            v=torch.flip(v_b, dims=[1]),
            g=torch.flip(g_b, dims=[1]),
            beta=torch.flip(beta_flat, dims=[1]),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True
        )
        out_b = torch.flip(out_b, dims=[1])

        o = torch.cat([out_f, out_b], dim=-1)

        g_out = self.g_b_proj(self.g_a_proj(x_unp_flat))
        g_out = rearrange(g_out, 'b s t (h d) -> (b s) t h d', d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, '(b s) t h d -> b s t (h d)', b=B, s=sub_nW)
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class BidirectionalKimiMLAAttention(nn.Module):
    def __init__(self, c, num_heads, rope_in_mla=True, rope_min_freq=math.pi/16, rope_max_freq=4*math.pi):
        super().__init__()
        self.hidden_size = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.rope_in_mla = rope_in_mla

        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        
        self.scale = self.head_dim ** -0.5
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
        
        if self.rope_in_mla:
            full_dim = self.head_dim
            K = 1
            for k in range(4, 0, -1):
                if full_dim % (4 * k) == 0:
                    K = k
                    break
            base_dirs = 2 * K
            all_indices = list(range(base_dirs))
            self.rope = SpiralRoPE2D(full_dim, num_heads, base_dirs=base_dirs, dir_indices=all_indices, min_freq=rope_min_freq, max_freq=rope_max_freq)

    def forward(self, x_pooled, x_unpooled, q_grid_y, q_grid_x, kv_grid_y, kv_grid_x):
        B, sub_nW, nW, C = x_pooled.shape
        N = x_unpooled.shape[3]
        seq_len = nW * N

        q = self.q_proj(x_pooled) 
        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)
        k = self.k_proj(x_unp_flat) 
        v = self.v_proj(x_unp_flat) 

        q = rearrange(q, 'b s w (h d) -> b s w h d', h=self.num_heads)
        k = rearrange(k, 'b s t (h d) -> b s t h d', h=self.num_heads)
        v = rearrange(v, 'b s t (h d) -> b s t h d', h=self.num_heads)

        if self.rope_in_mla:
            q = self.rope(q, q_grid_y, q_grid_x)
            
            kv_grid_y_flat = kv_grid_y.reshape(1, sub_nW, seq_len)
            kv_grid_x_flat = kv_grid_x.reshape(1, sub_nW, seq_len)
            k = self.rope(k, kv_grid_y_flat, kv_grid_x_flat)

        q = q.reshape(B * sub_nW, nW, self.num_heads, -1)
        k = k.reshape(B * sub_nW, seq_len, self.num_heads, -1)
        v = v.reshape(B * sub_nW, seq_len, self.num_heads, -1)

        q = rearrange(q, 'bt w h d -> bt h w d')
        k = rearrange(k, 'bt t h d -> bt h t d')
        v = rearrange(v, 'bt t h d -> bt h t d')

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        o_pooled = torch.matmul(attn_weights, v)
        o_pooled = rearrange(o_pooled, 'bt h w d -> bt w (h d)') 
        
        o = o_pooled.view(B, sub_nW, nW, 1, -1).expand(B, sub_nW, nW, N, -1)
        o = o.reshape(B * sub_nW, seq_len, -1)
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class CustomKimiVisionAttention(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, is_full_attn=False, rope_in_kda=True, rope_in_mla=True, rope_min_freq=math.pi/16, rope_max_freq=4*math.pi):
        super().__init__()
        self.win_size = win_size
        self.sub_win_size = sub_win_size
        self.is_full_attn = is_full_attn

        if is_full_attn:
            self.attn = BidirectionalKimiMLAAttention(c, num_heads, rope_in_mla=rope_in_mla, rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq)
        else:
            self.attn = BidirectionalKimiDeltaAttention(c, num_heads, rope_in_kda=rope_in_kda, rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq)

    def get_coords(self, H, W, win_size, sub_win_size, device):
        key = (H, W, win_size, sub_win_size, device)
        if hasattr(self, 'coord_cache') and key in self.coord_cache:
            return self.coord_cache[key]
            
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), 
                                        torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        grid_y = grid_y / win_size[0]
        grid_x = grid_x / win_size[1]
        
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        win_y = rearrange(grid_y, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_y = rearrange(win_y, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_y = sub_win_y.transpose(1, 2)
        
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        win_x = rearrange(grid_x, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_x = rearrange(win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_x = sub_win_x.transpose(1, 2)
        
        kda_grid_y = shuffled_sub_y.squeeze(-1) 
        kda_grid_x = shuffled_sub_x.squeeze(-1)
        
        mla_grid_y = kda_grid_y.mean(dim=-1) 
        mla_grid_x = kda_grid_x.mean(dim=-1)
        
        if not hasattr(self, 'coord_cache'):
            self.coord_cache = {}
        if len(self.coord_cache) > 4:
            self.coord_cache.clear()
            
        res = (kda_grid_y, kda_grid_x, mla_grid_y, mla_grid_x)
        self.coord_cache[key] = res
        return res

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
        pad_H, pad_W = x_pad.shape[2], x_pad.shape[3]

        win_x = rearrange(x_pad, 'B C (nWh wH) (nWw wW) -> B (nWh nWw) wH wW C', wH=win_size[0], wW=win_size[1])
        sub_win_x = rearrange(win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1])
        shuffled_sub_win_x = sub_win_x.transpose(1, 2)

        x_unpooled = shuffled_sub_win_x
        x_pooled = x_unpooled.mean(dim=-2)

        kda_y, kda_x, mla_y, mla_x = self.get_coords(pad_H, pad_W, win_size, sub_win_size, x.device)

        if self.is_full_attn:
            out = self.attn(x_pooled, x_unpooled, q_grid_y=mla_y, q_grid_x=mla_x, kv_grid_y=kda_y, kv_grid_x=kda_x)
        else:
            out = self.attn(x_pooled, x_unpooled, q_grid_y=kda_y, q_grid_x=kda_x, kv_grid_y=kda_y, kv_grid_x=kda_x)

        out = out.transpose(1, 2)
        out = rearrange(out, 'B nW (nWh nWw) (wH wW) C -> B nW (nWh wH) (nWw wW) C',
                        nWh=win_size[0] // sub_win_size[0], wH=sub_win_size[0], wW=sub_win_size[1])
        out = rearrange(out, 'B (nWh nWw) wH wW C -> B C (nWh wH) (nWw wW)',
                        nWh=nWh, wH=win_size[0], wW=win_size[1])

        return out[:, :, :H, :W]


class LiteKimiSWSBlock(nn.Module):
    def __init__(self, c, layer_idx, num_heads, win_size, sub_win_size, nwc_k, FFN_Expand=2.66, 
                 use_peg=True, rope_in_kda=True, rope_in_mla=True, rope_min_freq=math.pi/16, rope_max_freq=4*math.pi):
        super().__init__()
        self.use_peg = use_peg
        if use_peg:
            self.pos_block = PEG(c)
        is_full_attn = (layer_idx % 4 == 3) 
        self.attn = CustomKimiVisionAttention(c, num_heads, win_size, sub_win_size, is_full_attn,
                                              rope_in_kda=rope_in_kda, rope_in_mla=rope_in_mla,
                                              rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k//2, groups=c, bias=True)
        self.gdfn = FeedForward(c, FFN_Expand, bias=False)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        if self.use_peg:
            x = self.pos_block(x)
        y = inp + self.attn(x) * self.beta
        y = y + self.nwc(self.norm2(y)) * self.alpha
        y = y + self.gdfn(self.norm3(y)) * self.gamma
        return y


@ARCH_REGISTRY.register()
class LiteKimiSWSformerSpiralRoPE(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 win_size=16, sub_win_size=8, nwc_ks=[7, 5, 3, 3, 3], num_heads=[1, 2, 4, 8, 16],
                 use_peg=True, rope_in_kda=True, rope_in_mla=True, rope_min_freq=math.pi/16, rope_max_freq=4*math.pi):
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
            self.encoders.append(nn.Sequential(*[LiteKimiSWSBlock(chan, global_layer_idx+i, nH, win_size, sub_win_size, nwc_k, 
                                                                  use_peg=use_peg, rope_in_kda=rope_in_kda, rope_in_mla=rope_in_mla,
                                                                  rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq) for i in range(nB)]))
            global_layer_idx += nB
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[LiteKimiSWSBlock(chan, global_layer_idx+i, num_heads[-1], win_size, sub_win_size, nwc_ks[-1], 
                                                            use_peg=use_peg, rope_in_kda=rope_in_kda, rope_in_mla=rope_in_mla,
                                                            rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq) for i in range(middle_blk_num)])
        global_layer_idx += middle_blk_num

        for nB, nwc_k, nH in zip(dec_blk_nums, nwc_ks[-2::-1], num_heads[-2::-1]):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[LiteKimiSWSBlock(chan, global_layer_idx+i, nH, win_size, sub_win_size, nwc_k, 
                                                                  use_peg=use_peg, rope_in_kda=rope_in_kda, rope_in_mla=rope_in_mla,
                                                                  rope_min_freq=rope_min_freq, rope_max_freq=rope_max_freq) for i in range(nB)]))
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
