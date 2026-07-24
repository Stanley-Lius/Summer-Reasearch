# ------------------------------------------------------------------------
# Vectorized Dynamic Cascaded Gaze Network (VectorizedDynamicCGNet)
#
# `dynamic_cgnet_arch.py` (DynamicCGNet) decides a per-image receptive-field size by
# walking an attention-derived diagonal until it drops below a learnable threshold `T`,
# then physically re-slices the feature map into that many ragged sub-patches, and
# recurses the same procedure into those sub-patches for 2 more hierarchy levels.
# Two things make that expensive in a way that gets worse the more "content-adaptive"
# it tries to be:
#   1. The threshold walk is a Python double loop (`for b in range(B): for s in range(...):
#      if A[b, 0, ...] > T`). Comparing a single CUDA tensor element inside a Python `if`
#      forces an implicit `.item()`-style synchronization on every iteration.
#   2. `DynamicGCE` feeds layer 1's *patches* into layer 2, and layer 2's patches into
#      layer 3 (see its forward()), so the batch dimension the threshold walk loops over
#      multiplies across levels: level 3's loop runs `B * (patches from level 1) *
#      (patches from level 2)` times. Since the patch count is itself decided by the
#      dynamic threshold, a content decision toward smaller patches directly multiplies
#      the number of synchronizing loop iterations at the next level.
#
# `OmniDirDynamicCGNet` solved this by removing the dynamic decision entirely (fixed
# 3/5/7 causal convs, static shapes). That fixes the performance problem but throws away
# the useful idea: letting the network decide, per image, how much context a region
# needs.
#
# This file keeps that idea but implements it so the *shape* of every tensor and the
# *number* of ops is fixed at construction time and independent of the data - only the
# *weights* mixing a small set of fixed-shape candidate receptive fields are
# data-dependent:
#   - The diagonal threshold walk is replaced by one gather + cumprod per direction
#     (no Python loop over batch, no recursion, no per-element GPU->CPU sync).
#   - Instead of re-slicing the tensor into a dynamically-sized patch, several
#     fixed-shape candidate branches (increasing dilation, plus one strided branch for
#     a cheap large receptive field, in the spirit of CGNet's original
#     GlobalContextExtractor) are all run on the full batch, and the attention-derived
#     signal only decides a soft mixing weight between them.
#   - The hard `>` threshold comparison (non-differentiable, so `T` in the original
#     code essentially never received a useful gradient) is replaced with a sigmoid, so
#     `T` is now actually trainable.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_arch import LayerNorm2d, SimpleGate, NAFBlock0


class DilatedContextBranch(nn.Module):
    """One fixed-shape candidate receptive field: a dilated depthwise conv + pointwise mix.

    Larger `dilation` widens the receptive field without growing the kernel or the
    parameter count - the branch's output shape never depends on its input content.
    """

    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv2d(c, c, kernel_size, padding=padding, dilation=dilation, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        return F.gelu(self.pw(self.dw(x)))


class StridedContextBranch(nn.Module):
    """Cheap large-receptive-field candidate branch.

    Reuses CGNet's original `GlobalContextExtractor` idea (downsample with a strided
    conv to cover a large area cheaply, then upsample back) as one fixed-shape option
    among several, rather than the network's only source of wide context.
    """

    def __init__(self, c, kernel_size=5, stride=4):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(c, c, kernel_size, stride=stride, padding=padding, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = F.gelu(self.pw(self.dw(x)))
        return F.interpolate(out, size=(h, w), mode='nearest')


class VectorizedDynamicContext(nn.Module):
    """Content-adaptive mixture over fixed-shape receptive fields.

    Computes a per-image "reach" score the same way `DynamicGCELayer` did - walk two
    diagonals from a coarse attention map's corner tokens and see how far the attention
    stays above a threshold - but as a single vectorized gather + cumprod instead of a
    nested Python loop, and uses that score to softly weight a fixed set of candidate
    branches instead of physically re-slicing the tensor into a dynamically-sized patch.
    Every tensor shape here is fixed at construction time; only the mixing weights are
    data-dependent, so cost cannot blow up with recursion depth or patch count.
    """

    def __init__(self, c, num_scales=4, max_radius=4, pattern_stride=3):
        super().__init__()
        assert num_scales >= 2, 'need at least a local and a wide branch'
        self.c = c
        self.max_radius = max_radius
        self.pattern_stride = pattern_stride
        self.tau = 1.0

        dilations = [2 ** i for i in range(num_scales - 1)]  # e.g. [1, 2, 4] for num_scales=4
        self.branches = nn.ModuleList(
            [DilatedContextBranch(c, kernel_size=3, dilation=d) for d in dilations] +
            [StridedContextBranch(c, kernel_size=5, stride=4)]
        )
        self.num_scales = len(self.branches)

        # Coarse pattern extraction + attention, used only to compute the "reach" score.
        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=pattern_stride, padding=0, groups=c, bias=False)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.T = nn.Parameter(torch.tensor(0.0))

        self.gate_mlp = nn.Linear(1, self.num_scales)
        # Learned fallback gate for inputs too small to form a coarse attention grid.
        self.fallback_gate_logits = nn.Parameter(torch.zeros(self.num_scales))

        self.fuse_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def _compute_gate(self, x):
        b, c, h, w = x.shape
        if h < self.pattern_stride * 2 or w < self.pattern_stride * 2:
            return F.softmax(self.fallback_gate_logits, dim=0).expand(b, -1)

        pad_h = (self.pattern_stride - h % self.pattern_stride) % self.pattern_stride
        pad_w = (self.pattern_stride - w % self.pattern_stride) % self.pattern_stride
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x

        pattern = self.pattern_conv(x_pad)  # (B, C, Hp, Wp)
        hp, wp = pattern.shape[-2:]
        n = hp * wp
        if hp < 2 or wp < 2:
            return F.softmax(self.fallback_gate_logits, dim=0).expand(b, -1)

        q = self.q_conv(pattern).flatten(2)  # (B, C, N)
        k = self.k_conv(pattern).flatten(2)
        attn = torch.bmm(q.transpose(1, 2), k) / (c ** 0.5)  # (B, N, N)

        r = max(1, min(self.max_radius, hp - 1, wp - 1))
        s = torch.arange(1, r + 1, device=x.device)
        # Anchor at top-left token (index 0), walking toward bottom-right.
        idx_tl = torch.clamp(s * (wp + 1), max=n - 1)
        # Anchor at top-right token (index wp - 1), walking toward bottom-left.
        idx_tr = torch.clamp((wp - 1) + s * (wp - 1), max=n - 1)

        valid_tl = torch.sigmoid((attn[:, 0, idx_tl] - self.T) / self.tau)
        valid_tr = torch.sigmoid((attn[:, wp - 1, idx_tr] - self.T) / self.tau)
        reach_tl = torch.cumprod(valid_tl, dim=1).sum(dim=1)
        reach_tr = torch.cumprod(valid_tr, dim=1).sum(dim=1)
        reach = torch.maximum(reach_tl, reach_tr) / r  # (B,), roughly in [0, 1]

        gate_logits = self.gate_mlp(reach.unsqueeze(-1))  # (B, num_scales)
        return F.softmax(gate_logits, dim=-1)

    def forward(self, x):
        b = x.shape[0]
        branch_outs = torch.stack([branch(x) for branch in self.branches], dim=1)  # (B, K, C, H, W)
        gate = self._compute_gate(x).view(b, self.num_scales, 1, 1, 1)
        fused = (branch_outs * gate).sum(dim=1)
        return self.fuse_conv(fused)


class VectorizedDynamicGazeBlock(nn.Module):
    """CGNet-style block with `VectorizedDynamicContext` in place of a fixed or
    recursively-dynamic GCE module."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 context_num_scales=4, context_max_radius=4):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1,
                                groups=self.dw_channel, bias=True)
        self.sg1 = SimpleGate()

        context_channels = self.dw_channel // 2  # matches SimpleGate's channel halving
        self.context = VectorizedDynamicContext(context_channels, num_scales=context_num_scales,
                                                 max_radius=context_max_radius)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv_ffn_dw = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, padding=1,
                                      groups=ffn_channel, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)

        x = self.context(x)
        x = x * self.sca(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        ffn = self.conv4(self.norm2(y))
        ffn = self.conv_ffn_dw(ffn)
        ffn = self.sg2(ffn)
        ffn = self.conv5(ffn)
        ffn = self.dropout2(ffn)

        return y + ffn * self.gamma


@ARCH_REGISTRY.register()
class VectorizedDynamicCGNet(nn.Module):
    """Vectorized Dynamic CGNet.

    Same U-Net skeleton as `OmniDirDynamicCGNet` (drop-in replacement in configs), but
    following the family's own convention of using the "special" block only in some
    stages: `VectorizedDynamicGazeBlock` is used only in the deepest encoder stage, the
    middle blocks, and the shallowest decoder stage - where spatial resolution is
    smallest and the dynamic-context branches are cheapest - while shallower encoder
    and decoder stages use plain `NAFBlock0` for fast local detail processing.
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1],
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 context_num_scales=4, context_max_radius=4):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        def make_stage(chan, num_blocks, use_context):
            if use_context:
                return nn.Sequential(*[
                    VectorizedDynamicGazeBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                                                drop_out_rate=drop_out_rate,
                                                context_num_scales=context_num_scales,
                                                context_max_radius=context_max_radius)
                    for _ in range(num_blocks)
                ])
            return nn.Sequential(*[
                NAFBlock0(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate)
                for _ in range(num_blocks)
            ])

        chan = width
        for i, num in enumerate(enc_blk_nums):
            is_deepest = (i == len(enc_blk_nums) - 1)
            self.encoders.append(make_stage(chan, num, use_context=is_deepest))
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = make_stage(chan, middle_blk_num, use_context=True)

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan = chan // 2
            is_shallowest = (i == 0)
            self.decoders.append(make_stage(chan, num, use_context=is_shallowest))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)

        x = self.intro(inp_padded)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x
