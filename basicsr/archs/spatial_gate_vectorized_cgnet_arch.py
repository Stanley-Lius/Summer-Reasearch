# ------------------------------------------------------------------------
# SpatialGateVectorizedCGNet
#
# Improved version of `vectorized_dynamic_cgnet_arch.py` (VectorizedDynamicCGNet).
# The class name carries the main fix; the full list of updates (keywords in
# brackets):
#
# 1. [SpatialGate] The original derives ONE scalar "reach" per image and mixes
#    the candidate receptive-field branches with a single image-wide weight
#    vector, so mixed-content images (flat sky + dense texture) get one
#    compromise receptive field everywhere. A lightweight conv head now
#    produces a per-pixel gate map (B, K, H, W), letting every region pick its
#    own branch mixture; the global signal survives as a per-image bias.
# 2. [AnchorRow] The original computes the full (B, N, N) attention matrix via
#    `bmm(q^T, k)` but only ever reads TWO rows of it (the top-left and
#    top-right anchor tokens). Attention is now computed for the two anchor
#    queries only — (B, 2, N) — dropping the cost from O(N^2) to O(N) with
#    bit-identical semantics.
# 3. [StatCond] The per-image gate bias is conditioned on per-channel global
#    mean/std pooled statistics in addition to the reach scalar, instead of a
#    single scalar through Linear(1, K).
# 4. [LearnableTau] The sigmoid temperature of the soft threshold walk was a
#    fixed 1.0; it is now a learnable log-parameterized value, so the walk can
#    sharpen or soften as training demands.
# 5. [Bilinear] The strided wide-context branch upsampled with `nearest`
#    (blocky artifacts); it now uses bilinear interpolation.
#
# The original files are left untouched. Components shared with them are
# COPIED below (see the "PRE-EXISTING COMPONENTS" section) so this file is
# fully self-contained; the U-Net skeleton comes from `cgnet_unet_base.py`.
# The heavy blocks keep the original placement: deepest encoder stage,
# bottleneck, and the lowest-resolution decoder stage.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_unet_base import CGNetUNetBase, pad_safe


# ==========================================================================
# PRE-EXISTING COMPONENTS — copied unchanged from `cgnet_arch.py` and
# `vectorized_dynamic_cgnet_arch.py`.
# 原檔既有元件（原樣複製，非本次新增）。
# ==========================================================================

class LayerNormFunction(torch.autograd.Function):
    """Custom autograd function for 2D Layer Normalization across channels.

    (Copied from `cgnet_arch.py`.)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """2D Layer Normalization across spatial dimensions (N, C, H, W).

    (Copied from `cgnet_arch.py`.)
    """

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple Gate parameter-free activation block.

    (Copied from `cgnet_arch.py`.)
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock0(nn.Module):
    """Standard NAFNet block used in middle and decoding stages.

    (Copied from `cgnet_arch.py`.)
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class DilatedContextBranch(nn.Module):
    """One fixed-shape candidate receptive field: a dilated depthwise conv + pointwise mix.

    Larger `dilation` widens the receptive field without growing the kernel or the
    parameter count - the branch's output shape never depends on its input content.

    (Copied from `vectorized_dynamic_cgnet_arch.py`.)
    """

    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv2d(c, c, kernel_size, padding=padding, dilation=dilation, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        return F.gelu(self.pw(self.dw(x)))


# ==========================================================================
# NEW COMPONENTS — introduced by SpatialGateVectorizedCGNet.
# 本次新增元件。
# ==========================================================================

class BilinearStridedContextBranch(nn.Module):
    """[Bilinear] Cheap wide-receptive-field candidate branch.

    Same as the original `StridedContextBranch` (strided depthwise + pointwise,
    upsample back), but with bilinear instead of nearest interpolation to avoid
    blocky reconstruction artifacts.
    """

    def __init__(self, c, kernel_size=5, stride=4):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(c, c, kernel_size, stride=stride, padding=padding, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = F.gelu(self.pw(self.dw(x)))
        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)


class SpatialGateDynamicContext(nn.Module):
    """Per-pixel content-adaptive mixture over fixed-shape receptive fields.

    Keeps the original design's key property — every tensor shape is fixed at
    construction time, only mixing WEIGHTS are data-dependent — while upgrading
    where the weights come from:

      gate logits = spatial logits (conv head on the features, per pixel)
                  + global bias    (MLP on [channel mean, channel std, reach],
                                    per image)
      gate        = softmax over the K candidate branches.

    The "reach" scalar is computed exactly as in the original (soft diagonal
    walk on a coarse attention map, vectorized sigmoid + cumprod), but with
    [AnchorRow] anchor-query-only attention and a [LearnableTau] temperature.

    Debugging: set `record_gate_stats = True` to have each forward pass store
    the per-branch mean gate weight in `last_gate_mean` (detached, shape (K,)),
    e.g. to verify that all branches stay in use during training.
    """

    def __init__(self, c, num_scales=4, max_radius=4, pattern_stride=3):
        super().__init__()
        if num_scales < 2:
            raise ValueError('need at least a local and a wide branch')
        self.c = c
        self.max_radius = max_radius
        self.pattern_stride = pattern_stride

        dilations = [2 ** i for i in range(num_scales - 1)]  # e.g. [1, 2, 4] for num_scales=4
        self.branches = nn.ModuleList(
            [DilatedContextBranch(c, kernel_size=3, dilation=d) for d in dilations] +
            [BilinearStridedContextBranch(c, kernel_size=5, stride=4)]
        )
        self.num_scales = len(self.branches)

        # [SpatialGate] per-pixel gate head.
        gate_hidden = max(8, c // 4)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c, gate_hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_hidden, self.num_scales, kernel_size=3, padding=1, bias=True),
        )

        # Coarse pattern extraction + anchor attention for the "reach" score.
        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=pattern_stride, padding=0,
                                      groups=c, bias=False)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.T = nn.Parameter(torch.tensor(0.0))
        # [LearnableTau] tau = exp(log_tau), initialized to 1.0 like the original.
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        # Learned reach for inputs too small to form a coarse attention grid.
        self.fallback_reach_logit = nn.Parameter(torch.tensor(0.0))

        # [StatCond] per-image gate bias from pooled statistics + reach.
        self.cond_mlp = nn.Linear(2 * c + 1, self.num_scales)

        self.fuse_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

        # Debug switch; see class docstring.
        self.record_gate_stats = False
        self.last_gate_mean = None

    def _compute_reach(self, x):
        """Per-image receptive-field demand score, roughly in [0, 1]. Shape (B,)."""
        b, c, h, w = x.shape
        if h < self.pattern_stride * 2 or w < self.pattern_stride * 2:
            return torch.sigmoid(self.fallback_reach_logit).expand(b)

        pad_h = (self.pattern_stride - h % self.pattern_stride) % self.pattern_stride
        pad_w = (self.pattern_stride - w % self.pattern_stride) % self.pattern_stride
        x_pad = pad_safe(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x

        pattern = self.pattern_conv(x_pad)  # (B, C, Hp, Wp)
        hp, wp = pattern.shape[-2:]
        n = hp * wp
        if hp < 2 or wp < 2:
            return torch.sigmoid(self.fallback_reach_logit).expand(b)

        q = self.q_conv(pattern).flatten(2)  # (B, C, N)
        k = self.k_conv(pattern).flatten(2)  # (B, C, N)

        # [AnchorRow] attention for the two anchor queries only: O(N), not O(N^2).
        q_anchors = q[:, :, [0, wp - 1]]  # top-left and top-right tokens, (B, C, 2)
        attn = torch.bmm(q_anchors.transpose(1, 2), k) / (c ** 0.5)  # (B, 2, N)

        r = max(1, min(self.max_radius, hp - 1, wp - 1))
        s = torch.arange(1, r + 1, device=x.device)
        # Anchor at top-left token (index 0), walking toward bottom-right.
        idx_tl = torch.clamp(s * (wp + 1), max=n - 1)
        # Anchor at top-right token (index wp - 1), walking toward bottom-left.
        idx_tr = torch.clamp((wp - 1) + s * (wp - 1), max=n - 1)

        tau = self.log_tau.exp().clamp(min=1e-2)
        valid_tl = torch.sigmoid((attn[:, 0, idx_tl] - self.T) / tau)
        valid_tr = torch.sigmoid((attn[:, 1, idx_tr] - self.T) / tau)
        reach_tl = torch.cumprod(valid_tl, dim=1).sum(dim=1)
        reach_tr = torch.cumprod(valid_tr, dim=1).sum(dim=1)
        return torch.maximum(reach_tl, reach_tr) / r  # (B,)

    def _compute_gate(self, x):
        """Per-pixel branch mixture. Shape (B, K, H, W), softmax over K."""
        b = x.shape[0]

        # [StatCond] per-image conditioning: channel-wise mean/std + reach.
        mean = x.mean(dim=(2, 3))                                      # (B, C)
        std = x.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt()       # (B, C)
        reach = self._compute_reach(x)                                 # (B,)
        cond = torch.cat([mean, std, reach.unsqueeze(1)], dim=1)       # (B, 2C + 1)
        global_bias = self.cond_mlp(cond).view(b, self.num_scales, 1, 1)

        logits = self.spatial_gate(x) + global_bias                    # (B, K, H, W)
        return F.softmax(logits, dim=1)

    def forward(self, x):
        branch_outs = torch.stack([branch(x) for branch in self.branches], dim=1)  # (B, K, C, H, W)
        gate = self._compute_gate(x)                                               # (B, K, H, W)

        if self.record_gate_stats:
            self.last_gate_mean = gate.mean(dim=(0, 2, 3)).detach()

        fused = (branch_outs * gate.unsqueeze(2)).sum(dim=1)                       # (B, C, H, W)
        return self.fuse_conv(fused)

    def extra_repr(self):
        return (f'c={self.c}, num_scales={self.num_scales}, '
                f'max_radius={self.max_radius}, pattern_stride={self.pattern_stride}')


class SpatialGateDynamicGazeBlock(nn.Module):
    """CGNet-style block with `SpatialGateDynamicContext` as its context module.

    Structure (identical to `VectorizedDynamicGazeBlock` apart from the context):
        LayerNorm -> 1x1 expand (2c) -> 3x3 depthwise -> SimpleGate (-> c)
        -> spatial-gated dynamic context -> SE-style channel attention
        -> residual (beta) -> GDFN-style FFN with SimpleGate -> residual (gamma)
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 context_num_scales=4, context_max_radius=4):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1,
                               groups=self.dw_channel, bias=True)
        self.sg1 = SimpleGate()

        context_channels = self.dw_channel // 2  # matches SimpleGate's channel halving
        self.context = SpatialGateDynamicContext(context_channels, num_scales=context_num_scales,
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
class SpatialGateVectorizedCGNet(CGNetUNetBase):
    """VectorizedDynamicCGNet with SpatialGate / AnchorRow / StatCond /
    LearnableTau / Bilinear fixes.

    Stage layout (unchanged from the original — heavy blocks only where
    resolution is small):
        encoders: NAFBlock0, except the deepest stage -> SpatialGateDynamicGazeBlock,
        middle:   SpatialGateDynamicGazeBlock,
        decoders: NAFBlock0, except the lowest-resolution stage (index 0,
                  right after the bottleneck) -> SpatialGateDynamicGazeBlock.

    Config-compatible with VectorizedDynamicCGNet (same constructor signature).
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1),
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 context_num_scales=4, context_max_radius=4):

        def dynamic_stage(chan, num_blocks):
            return nn.Sequential(*[
                SpatialGateDynamicGazeBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                                            drop_out_rate=drop_out_rate,
                                            context_num_scales=context_num_scales,
                                            context_max_radius=context_max_radius)
                for _ in range(num_blocks)
            ])

        def naf_stage(chan, num_blocks):
            return nn.Sequential(*[
                NAFBlock0(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                          drop_out_rate=drop_out_rate)
                for _ in range(num_blocks)
            ])

        def make_encoder_stage(index, num_stages, chan, num_blocks):
            is_deepest = (index == num_stages - 1)
            return dynamic_stage(chan, num_blocks) if is_deepest else naf_stage(chan, num_blocks)

        def make_middle_stage(chan, num_blocks):
            return dynamic_stage(chan, num_blocks)

        def make_decoder_stage(index, num_stages, chan, num_blocks):
            is_lowest_resolution = (index == 0)
            return dynamic_stage(chan, num_blocks) if is_lowest_resolution else naf_stage(chan, num_blocks)

        super().__init__(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
                         make_encoder_stage=make_encoder_stage,
                         make_middle_stage=make_middle_stage,
                         make_decoder_stage=make_decoder_stage,
                         pad_mode='reflect')
