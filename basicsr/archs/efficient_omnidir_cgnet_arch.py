# ------------------------------------------------------------------------
# EfficientOmniDirCGNet
#
# Improved version of `omnidir_dynamic_cgnet_arch.py` (OmniDirDynamicCGNet).
# The class name carries the main fix; the full list of updates (keywords in
# brackets):
#
# 1. [Efficient] The original network places its heavy 4-direction block in
#    EVERY stage, including the full-resolution first encoder stage, which
#    dominates VRAM and runtime. Heavy blocks are now used only where spatial
#    resolution is small — the deepest encoder stage, the bottleneck, and the
#    lowest-resolution decoder stage — while all other stages use plain
#    NAFBlock0 (the family's own convention, cf. VectorizedDynamicCGNet).
# 2. [SharedFlip] The original learns 4 independent depthwise convs whose
#    quadrant windows heavily overlap. Only 2 directional convs (top-left,
#    top-right) are learned now; the bottom-right / bottom-left responses are
#    obtained by running the same convs on the 180-degree-flipped input and
#    flipping back — genuine directional receptive fields at half the
#    depthwise parameters. The 4 per-direction pointwise convs are also merged
#    into a single grouped 1x1 conv.
# 3. [GlobalBranch] The original's largest receptive field is a 3+5+7 causal
#    cascade (~12 px one-sided) — it lost CGNet's cheap wide context entirely.
#    The third GCE level is replaced by a strided global-context layer
#    (downsample -> conv -> bilinear upsample), restoring wide-area noise
#    statistics at negligible cost.
# 4. [Residual] The original GCE stacks 3 gated+GELU layers with no residual,
#    hurting gradient flow. Each GCE level now adds its input back.
# 5. SimpleGate uses the chunk-based semantics of `cgnet_arch.py` instead of
#    the original file's divergent interleaved-slicing redefinition, so the
#    whole family shares one gate semantics.
#
# The original files are left untouched. Components shared with them are
# COPIED below (see the "PRE-EXISTING COMPONENTS" section) so this file is
# fully self-contained; the U-Net skeleton comes from `cgnet_unet_base.py`.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_unet_base import CGNetUNetBase, pad_safe


# ==========================================================================
# PRE-EXISTING COMPONENTS — copied unchanged from `cgnet_arch.py` and
# `omnidir_dynamic_cgnet_arch.py`.
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

    (Copied from `cgnet_arch.py` — chunk-based, NOT the interleaved variant
    that `omnidir_dynamic_cgnet_arch.py` redefines.)
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


class FactorizedAdaptiveSpatialChannelGatedFusion(nn.Module):
    """
    Factorized Adaptive Spatial-Channel Gated Fusion (Factorized ASC-Gate) module.
    Fuses 4 directional feature maps (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    using factorized spatial gating (4 channels at full resolution) and channel gating
    (4*c channels at 1x1 resolution). Cuts intermediate backward activation memory by >90%
    compared to dense 4*c spatial convolution while retaining independent multi-directional
    feature selection per spatial coordinate and per channel.

    (Copied from `omnidir_dynamic_cgnet_arch.py`.)
    """
    def __init__(self, c):
        super().__init__()
        # Lightweight spatial directional gating (4 channels at spatial resolution HxW)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4 * c, 4, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        # Lightweight channel directional gating (4*c channels at 1x1 resolution)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * c, max(1, c), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(1, c), 4 * c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, feats):
        # feats: list of 4 feature maps [tl, tr, bl, br], each of shape (B, C, H, W)
        cat_feats = torch.cat(feats, dim=1)  # (B, 4*C, H, W)

        s_gates = self.spatial_gate(cat_feats).chunk(4, dim=1)  # tuple of 4 tensors of (B, 1, H, W)
        c_gates = self.channel_gate(cat_feats).chunk(4, dim=1)  # tuple of 4 tensors of (B, C, 1, 1)

        out = (feats[0] * s_gates[0] * c_gates[0] +
               feats[1] * s_gates[1] * c_gates[1] +
               feats[2] * s_gates[2] * c_gates[2] +
               feats[3] * s_gates[3] * c_gates[3])
        return self.proj(out)


# ==========================================================================
# NEW COMPONENTS — introduced by EfficientOmniDirCGNet.
# 本次新增元件。
# ==========================================================================

class SharedFlipFourDirDWConv(nn.Module):
    """Four directional receptive fields from two learned depthwise convs.

    Learns depthwise kernels for the top-left and top-right causal windows
    only. The bottom-right (resp. bottom-left) response is the top-left
    (resp. top-right) conv applied to the input flipped in both spatial dims,
    flipped back afterwards — the shared kernel is effectively rotated 180
    degrees, so each direction still has a distinct receptive field while the
    depthwise parameter count is halved. A single grouped 1x1 conv replaces
    the original's four separate pointwise convs.

    Input (B, C, H, W) -> list of 4 tensors [tl, tr, bl, br], each (B, C, H, W).
    """

    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_tl = nn.Conv2d(c, c, kernel_size, padding=0, groups=c, bias=False)
        self.dw_tr = nn.Conv2d(c, c, kernel_size, padding=0, groups=c, bias=False)
        # One 1x1 conv over the concatenated directions, grouped so each
        # direction is mixed independently (equivalent to 4 convs, one kernel).
        self.pw = nn.Conv2d(4 * c, 4 * c, kernel_size=1, groups=4, bias=True)

    def _directional(self, x, dw_conv, pad):
        """Causal-pad with `pad` = (left, right, top, bottom), then convolve."""
        return dw_conv(pad_safe(x, pad))

    def forward(self, x):
        k = self.kernel_size
        x_flipped = torch.flip(x, dims=(-2, -1))

        tl = self._directional(x, self.dw_tl, (k - 1, 0, k - 1, 0))
        tr = self._directional(x, self.dw_tr, (0, k - 1, k - 1, 0))
        # 180-degree weight sharing: TL kernel on the flipped image, flipped
        # back, is a genuine bottom-right causal response (and TR -> BL).
        br = torch.flip(self._directional(x_flipped, self.dw_tl, (k - 1, 0, k - 1, 0)), dims=(-2, -1))
        bl = torch.flip(self._directional(x_flipped, self.dw_tr, (0, k - 1, k - 1, 0)), dims=(-2, -1))

        out = F.gelu(self.pw(torch.cat([tl, tr, bl, br], dim=1)))
        return list(out.chunk(4, dim=1))

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, directions=4 (2 learned + 180-degree flip sharing)'


class EfficientOmniDirGCELayer(nn.Module):
    """One directional context level: 4-direction extraction + ASC-Gate fusion."""

    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.four_dir = SharedFlipFourDirDWConv(c, kernel_size=kernel_size)
        self.fusion = FactorizedAdaptiveSpatialChannelGatedFusion(c)

    def forward(self, x):
        return self.fusion(self.four_dir(x))


class GlobalStridedContextLayer(nn.Module):
    """[GlobalBranch] Cheap wide-receptive-field level.

    Strided depthwise + pointwise conv covers a large area at 1/stride^2 cost,
    then bilinear upsampling restores the input resolution. Replaces the
    original third directional level (kernel 7), whose receptive field was
    still local while being the most expensive level of the three.
    """

    def __init__(self, c, kernel_size=5, stride=4):
        super().__init__()
        self.pad = kernel_size // 2
        self.dw = nn.Conv2d(c, c, kernel_size, stride=stride, padding=0, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = pad_safe(x, (self.pad, self.pad, self.pad, self.pad))
        out = F.gelu(self.pw(self.dw(out)))
        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)


class EfficientOmniDirGCE(nn.Module):
    """Hierarchical context module: two directional levels + one global level.

    [Residual] Each level adds its input back, so gradients reach the shallow
    levels directly and stacked gating cannot over-smooth the features.
    """

    def __init__(self, c):
        super().__init__()
        self.layer1 = EfficientOmniDirGCELayer(c, kernel_size=3)
        self.layer2 = EfficientOmniDirGCELayer(c, kernel_size=5)
        self.layer3 = GlobalStridedContextLayer(c, kernel_size=5, stride=4)

    def forward(self, x):
        p1 = self.layer1(x) + x
        p2 = self.layer2(p1) + p1
        p3 = self.layer3(p2) + p2
        return p1, p2, p3


class EfficientOmniDirGazeBlock(nn.Module):
    """OmniDirDynamicGazeBlock with the SharedFlip / GlobalBranch / Residual fixes.

    Structure (unchanged apart from the fixes):
        LayerNorm -> 1x1 expand (2c) -> 3x3 depthwise -> SimpleGate (-> c)
        -> 3-level GCE -> concat [x, p1, p2, p3] (4c) -> 1x1 aggregate
        -> SE-style channel attention -> residual (beta)
        -> GDFN-style FFN with SimpleGate -> residual (gamma)
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1,
                               groups=self.dw_channel, bias=True)
        self.sg1 = SimpleGate()

        gce_channels = self.dw_channel // 2  # SimpleGate halves the channels back to c
        self.GCE = EfficientOmniDirGCE(gce_channels)

        # Aggregate the gated trunk + 3 hierarchical context outputs.
        self.agg_conv = nn.Conv2d(gce_channels * 4, c, kernel_size=1, bias=True)

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
        x = self.sg1(x)  # (B, c, H, W)

        p1, p2, p3 = self.GCE(x)
        x_agg = self.agg_conv(torch.cat([x, p1, p2, p3], dim=1))

        x_out = x_agg * self.sca(x_agg)
        x_out = self.dropout1(x_out)

        y = inp + x_out * self.beta

        ffn = self.conv4(self.norm2(y))
        ffn = self.conv_ffn_dw(ffn)
        ffn = self.sg2(ffn)
        ffn = self.conv5(ffn)
        ffn = self.dropout2(ffn)

        return y + ffn * self.gamma


@ARCH_REGISTRY.register()
class EfficientOmniDirCGNet(CGNetUNetBase):
    """OmniDirDynamicCGNet with Efficient / SharedFlip / GlobalBranch / Residual fixes.

    Stage layout ([Efficient] — heavy blocks only where resolution is small):
        encoders: NAFBlock0, except the deepest stage -> EfficientOmniDirGazeBlock,
        middle:   EfficientOmniDirGazeBlock,
        decoders: NAFBlock0, except the lowest-resolution stage (index 0,
                  right after the bottleneck) -> EfficientOmniDirGazeBlock.

    Config-compatible with OmniDirDynamicCGNet (same constructor signature).
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1),
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):

        def omnidir_stage(chan, num_blocks):
            return nn.Sequential(*[
                EfficientOmniDirGazeBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                                          drop_out_rate=drop_out_rate)
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
            return omnidir_stage(chan, num_blocks) if is_deepest else naf_stage(chan, num_blocks)

        def make_middle_stage(chan, num_blocks):
            return omnidir_stage(chan, num_blocks)

        def make_decoder_stage(index, num_stages, chan, num_blocks):
            is_lowest_resolution = (index == 0)
            return omnidir_stage(chan, num_blocks) if is_lowest_resolution else naf_stage(chan, num_blocks)

        super().__init__(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
                         make_encoder_stage=make_encoder_stage,
                         make_middle_stage=make_middle_stage,
                         make_decoder_stage=make_decoder_stage,
                         pad_mode='reflect')
