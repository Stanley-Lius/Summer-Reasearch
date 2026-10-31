import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.archs.arch_util import LayerNorm2d
from basicsr.archs.SWSformer_arch import PEG, FeedForward
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from fla.ops.kda import chunk_kda
    from fla.modules import FusedRMSNormGated
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass


class LoopNoPoolBidirectionalKimiDeltaAttention(nn.Module):
    """
    L-Module Core Attention: Lite-Kimi-Linear (KDA) adapted for Vision:
    1. Unpooled pixel-specific queries (no Query Pooling in KDA).
    2. Eliminated redundant x_pooled mean reduction in the KDA path.
    3. Channel-Split Bidirectional Sweep: splits head_dim in half (forward D/2, backward D/2).
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

        self.A_log = nn.Parameter(
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

    def forward(self, x_unpooled):
        B, sub_nW, nW, N, C = x_unpooled.shape
        seq_len = nW * N

        x_unp_flat = x_unpooled.reshape(B, sub_nW, seq_len, C)

        q = F.silu(self.q_proj(x_unp_flat))
        k = F.silu(self.k_proj(x_unp_flat))
        v = F.silu(self.v_proj(x_unp_flat))

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
        out_f, _ = chunk_kda(
            q=q_f, k=k_f, v=v_f, g=g_f, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=True
        )

        out_b, _ = chunk_kda(
            q=torch.flip(q_b, dims=[1]),
            k=torch.flip(k_b, dims=[1]),
            v=torch.flip(v_b, dims=[1]),
            g=torch.flip(g_b, dims=[1]),
            beta=torch.flip(beta, dims=[1]),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        out_b = torch.flip(out_b, dims=[1])

        o = torch.cat([out_f, out_b], dim=-1)

        # Output Norm & Projection
        g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = rearrange(g_out, 'b t (h d) -> b t h d', d=self.head_dim)
        o = self.o_norm(o, g_out)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class LoopBidirectionalKimiMLAAttention(nn.Module):
    """
    H-Module Core Attention: Lite-MLA (Window Full Attention) adapted for Vision:
    Uses Q-Pooling over sub-windows to save 64x FLOPs/VRAM while integrating global window context.
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

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        o_pooled = torch.matmul(attn_weights, v)
        o_pooled = rearrange(o_pooled, 'b h t d -> b t (h d)')

        o = o_pooled.view(B, sub_nW, nW, 1, -1).expand(B, sub_nW, nW, N, -1)
        o = o.reshape(B * sub_nW, seq_len, -1)
        o = self.o_proj(o)

        return o.view(B, sub_nW, nW, N, C)


class LoopNoPoolCustomKimiVisionAttention(nn.Module):
    def __init__(self, c, num_heads, win_size, sub_win_size, is_full_attn=False):
        super().__init__()
        self.win_size = win_size
        self.sub_win_size = sub_win_size
        self.is_full_attn = is_full_attn

        if is_full_attn:
            self.attn = LoopBidirectionalKimiMLAAttention(c, num_heads)
        else:
            self.attn = LoopNoPoolBidirectionalKimiDeltaAttention(c, num_heads)

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
        sub_win_x = rearrange(
            win_x, 'B nW (nWh wH) (nWw wW) C -> B nW (nWh nWw) (wH wW) C', wH=sub_win_size[0], wW=sub_win_size[1]
        )
        x_unpooled = sub_win_x.transpose(1, 2)  # (B, sub_nW, nW, N, C)

        if self.is_full_attn:
            x_pooled = x_unpooled.mean(dim=-2)  # (B, sub_nW, nW, C)
            out = self.attn(x_pooled, x_unpooled)
        else:
            out = self.attn(x_unpooled)

        # Unshuffle and reconstruct image
        out = out.transpose(1, 2)  # B, nW, sub_nW, N, C
        out = rearrange(
            out,
            'B nW (nWh nWw) (wH wW) C -> B nW (nWh wH) (nWw wW) C',
            nWh=win_size[0] // sub_win_size[0],
            wH=sub_win_size[0],
            wW=sub_win_size[1],
        )
        out = rearrange(
            out, 'B (nWh nWw) wH wW C -> B C (nWh wH) (nWw wW)', nWh=nWh, wH=win_size[0], wW=win_size[1]
        )

        return out[:, :, :H, :W]


class VisualAnchor2D(nn.Module):
    """
    2D adaptation of LoopVL's VisualAnchor for image restoration.
    Controls cycle-specific re-injection strength of the initial stage anchor features.
    """

    def __init__(self, H_cycles: int = 2, raw_init: float = -2.0, max_scale: float = 0.5):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.full((H_cycles,), float(raw_init)))
        self.max_scale = float(max_scale)

    def alpha(self, cycle: int) -> torch.Tensor:
        idx = min(cycle, self.raw_alpha.shape[0] - 1)
        return self.max_scale * torch.sigmoid(self.raw_alpha[idx])

    def forward(self, visual_anchor: torch.Tensor, gates: torch.Tensor, cycle: int) -> torch.Tensor:
        alpha = self.alpha(cycle).to(dtype=visual_anchor.dtype)
        return alpha * gates * visual_anchor


class LoopAware2DVisualGate(nn.Module):
    """
    2D adaptation of LoopVL's LoopAwareVisualGate without text instructions.
    Computes a spatial token-wise gate from the current high-level state z_H and initial anchor x_anchor.
    Zero-initialized output head ensures neutral gating (gate = 1.0) at initialization.
    """

    def __init__(self, hidden_size: int, gate_dim: int = 32):
        super().__init__()
        self.visual_proj = nn.Conv2d(hidden_size, gate_dim, kernel_size=1, bias=False)
        self.state_proj = nn.Conv2d(hidden_size, gate_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(gate_dim, 1, kernel_size=1, bias=True)
        self.reset_output_head()

    def reset_output_head(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, current_state: torch.Tensor, visual_anchor: torch.Tensor) -> torch.Tensor:
        output_dtype = current_state.dtype
        param_dtype = self.visual_proj.weight.dtype
        with torch.autocast(device_type=current_state.device.type, enabled=False):
            visual = self.visual_proj(visual_anchor.to(param_dtype))
            state = self.state_proj(current_state.to(param_dtype))
            score = self.out_proj(F.silu(visual + state))
            bounded_score = score.clamp(-8.0, 8.0)
            surrogate_score = score + (bounded_score - score).detach()
            raw_gate = 2.0 * torch.sigmoid(score)
            surrogate_gate = 2.0 * torch.sigmoid(surrogate_score)
            gate = raw_gate.detach() + surrogate_gate - surrogate_gate.detach()
        return gate.to(output_dtype)


class StepConditionedSWSBlock(nn.Module):
    """
    Shared SWSBlock (either L=KDA or H=MLA) with step-indexed channel scaling (beta, alpha, gamma).
    Provides `forward_delta` to return pure feature increments Delta(u) for coupled state updates,
    preventing additive identity explosion across recurrent loops.
    """

    def __init__(self, c, num_heads, win_size, sub_win_size, nwc_k, is_full_attn=False, num_steps=3, FFN_Expand=2.66):
        super().__init__()
        self.num_steps = max(int(num_steps), 1)
        self.pos_block = PEG(c)
        self.attn = LoopNoPoolCustomKimiVisionAttention(c, num_heads, win_size, sub_win_size, is_full_attn=is_full_attn)
        self.nwc = nn.Conv2d(c, c, kernel_size=nwc_k, stride=1, padding=nwc_k // 2, groups=c, bias=True)
        self.gdfn = FeedForward(c, FFN_Expand, bias=False)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((self.num_steps, 1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((self.num_steps, 1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((self.num_steps, 1, c, 1, 1)), requires_grad=True)

    def forward_delta(self, u: torch.Tensor, step_idx: int = 0) -> torch.Tensor:
        idx = min(step_idx, self.num_steps - 1)
        beta = self.beta[idx]
        alpha = self.alpha[idx]
        gamma = self.gamma[idx]

        delta1 = self.attn(self.pos_block(self.norm1(u))) * beta
        y1 = u + delta1
        delta2 = self.nwc(self.norm2(y1)) * alpha
        y2 = y1 + delta2
        delta3 = self.gdfn(self.norm3(y2)) * gamma
        return delta1 + delta2 + delta3

    def forward(self, inp: torch.Tensor, step_idx: int = 0) -> torch.Tensor:
        return inp + self.forward_delta(inp, step_idx=step_idx)


class LoopLiteNoPoolKimiSWSBlock(nn.Module):
    """
    LoopVL-inspired [L x L_cycles -> H] x H_cycles Recurrent SWS Stage Block.
    Instantiates exactly ONE L_module (KDA) and ONE H_module (MLA Window Attention),
    reusing them across inner and outer cycles over coupled states (z_L, z_H).
    """

    def __init__(
        self,
        c,
        num_heads,
        win_size,
        sub_win_size,
        nwc_k,
        H_cycles=2,
        L_cycles=3,
        FFN_Expand=2.66,
        use_visual_anchor=True,
        use_selective_backprop=True,
    ):
        super().__init__()
        self.H_cycles = int(H_cycles)
        self.L_cycles = int(L_cycles)
        self.use_visual_anchor = bool(use_visual_anchor)
        self.use_selective_backprop = bool(use_selective_backprop)

        # Shared L (KDA) and H (MLA Window Attention) modules
        self.L_module = StepConditionedSWSBlock(
            c,
            num_heads,
            win_size,
            sub_win_size,
            nwc_k,
            is_full_attn=False,
            num_steps=self.L_cycles,
            FFN_Expand=FFN_Expand,
        )
        self.H_module = StepConditionedSWSBlock(
            c,
            num_heads,
            win_size,
            sub_win_size,
            nwc_k,
            is_full_attn=True,
            num_steps=self.H_cycles,
            FFN_Expand=FFN_Expand,
        )

        # Persistent low-level initial state z_L_init
        self.z_L_init = nn.Parameter(torch.zeros(1, c, 1, 1))

        if self.use_visual_anchor and self.H_cycles > 1:
            gate_dim = max(c // 4, 16)
            self.visual_gate = LoopAware2DVisualGate(c, gate_dim=gate_dim)
            self.visual_anchor = VisualAnchor2D(self.H_cycles, raw_init=-2.0, max_scale=0.5)
        else:
            self.visual_gate = None
            self.visual_anchor = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        z_H = inp
        z_L = self.z_L_init.to(dtype=z_H.dtype, device=z_H.device).expand_as(z_H).contiguous()
        x_anchor = inp

        for h_idx in range(self.H_cycles):
            # Re-inject 2D gated visual anchor starting from the 2nd outer cycle (h_idx >= 1)
            if h_idx >= 1 and self.visual_anchor is not None and self.visual_gate is not None:
                gates = self.visual_gate(z_H, x_anchor)
                z_H = z_H + self.visual_anchor(x_anchor, gates, h_idx)

            # LoopVL Selective Backpropagation:
            # Exclude earlier outer cycles' L-calls (e.g., L1->L2->L3) from autograd graph during training,
            # while keeping H1 -> L4 -> L5 -> L6 -> H2 in the active backward graph.
            no_grad_l = (
                self.training
                and self.use_selective_backprop
                and (self.H_cycles > 1)
                and (h_idx < self.H_cycles - 1)
            )

            for l_idx in range(self.L_cycles):
                if no_grad_l:
                    with torch.no_grad():
                        delta_L = self.L_module.forward_delta(z_L + z_H, step_idx=l_idx)
                    z_L = z_L + delta_L
                else:
                    delta_L = self.L_module.forward_delta(z_L + z_H, step_idx=l_idx)
                    z_L = z_L + delta_L

            # High-level window attention update (always in active backward graph)
            delta_H = self.H_module.forward_delta(z_H + z_L, step_idx=h_idx)
            z_H = z_H + delta_H

        return z_H


def _infer_cycles_from_blk_num(blk_num: int) -> tuple[int, int]:
    """Fallback mapper from legacy block count to (H_cycles, L_cycles)."""
    if blk_num >= 8:
        return 2, 3
    elif blk_num >= 4:
        return 1, 3
    else:
        return 1, 1


@ARCH_REGISTRY.register()
class LoopLiteNoPoolKimiSWSformer(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=64,
        middle_blk_num=8,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        enc_h_cycles=None,
        enc_l_cycles=None,
        mid_h_cycles=None,
        mid_l_cycles=None,
        dec_h_cycles=None,
        dec_l_cycles=None,
        win_size=16,
        sub_win_size=8,
        nwc_ks=[7, 5, 3, 3, 3],
        num_heads=[1, 2, 4, 8, 16],
        use_visual_anchor=True,
        use_selective_backprop=True,
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        num_enc_stages = len(enc_blk_nums) if enc_blk_nums else len(enc_h_cycles)
        if enc_h_cycles is None or enc_l_cycles is None:
            inferred_enc = [_infer_cycles_from_blk_num(n) for n in enc_blk_nums]
            enc_h_cycles = [h for h, _ in inferred_enc]
            enc_l_cycles = [l for _, l in inferred_enc]

        if mid_h_cycles is None or mid_l_cycles is None:
            mid_h_cycles, mid_l_cycles = _infer_cycles_from_blk_num(middle_blk_num)

        if dec_h_cycles is None or dec_l_cycles is None:
            inferred_dec = [_infer_cycles_from_blk_num(n) for n in dec_blk_nums]
            dec_h_cycles = [h for h, _ in inferred_dec]
            dec_l_cycles = [l for _, l in inferred_dec]

        chan = width
        for idx in range(num_enc_stages):
            nwc_k = nwc_ks[idx]
            nH = num_heads[idx]
            h_cyc = enc_h_cycles[idx]
            l_cyc = enc_l_cycles[idx]
            self.encoders.append(
                LoopLiteNoPoolKimiSWSBlock(
                    chan,
                    nH,
                    win_size,
                    sub_win_size,
                    nwc_k,
                    H_cycles=h_cyc,
                    L_cycles=l_cyc,
                    use_visual_anchor=use_visual_anchor,
                    use_selective_backprop=use_selective_backprop,
                )
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = LoopLiteNoPoolKimiSWSBlock(
            chan,
            num_heads[-1],
            win_size,
            sub_win_size,
            nwc_ks[-1],
            H_cycles=mid_h_cycles,
            L_cycles=mid_l_cycles,
            use_visual_anchor=use_visual_anchor,
            use_selective_backprop=use_selective_backprop,
        )

        num_dec_stages = len(dec_h_cycles)
        dec_nwc_ks = nwc_ks[-2::-1]
        dec_num_heads = num_heads[-2::-1]
        for idx in range(num_dec_stages):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(
                LoopLiteNoPoolKimiSWSBlock(
                    chan,
                    dec_num_heads[idx],
                    win_size,
                    sub_win_size,
                    dec_nwc_ks[idx],
                    H_cycles=dec_h_cycles[idx],
                    L_cycles=dec_l_cycles[idx],
                    use_visual_anchor=use_visual_anchor,
                    use_selective_backprop=use_selective_backprop,
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        x = self.intro(F.pad(inp, (0, pad_w, 0, pad_h)))
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = decoder(up(x) + enc_skip)
        return (self.ending(x) + inp)[:, :, :H, :W]
