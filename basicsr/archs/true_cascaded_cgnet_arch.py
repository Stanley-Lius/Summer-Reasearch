# ------------------------------------------------------------------------
# TrueCascadeCGNet
#
# Improved version of `cgnet_arch.py` (CGNet / CascadedGaze). The class name
# carries the main fix; the full list of updates (keywords in brackets):
#
# 1. [TrueCascade] `GlobalContextExtractor` in the original file applies every
#    strided conv to the SAME input (`x` is never reassigned), so the "cascade"
#    that gives the architecture its name never happens and the effective
#    receptive field is a single stride level. Here each conv consumes the
#    previous conv's output, so strides multiply (e.g. 2 -> 2*3 -> 2*3*4) and
#    the intended large receptive field is restored.
# 2. [Reflect] All context convs previously used `padding=0`, silently
#    truncating borders; the model-level size padding used zero padding.
#    Both now use symmetric reflect padding, removing systematic quality loss
#    at image borders.
# 3. [Bilinear] Low-resolution context maps were upsampled with `nearest`,
#    producing blocky artifacts; bilinear interpolation is used instead.
# 4. [DecoderContext] The original decoder was NAFBlock0-only, so the
#    reconstruction path had no access to global context. The full-resolution
#    (shallowest) decoder stage now starts with one context block.
#
# The original file is left untouched. Components shared with it are COPIED
# below (see the "PRE-EXISTING COMPONENTS" section) so this file is fully
# self-contained; the U-Net skeleton comes from `cgnet_unet_base.py`.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_unet_base import CGNetUNetBase, pad_safe


# ==========================================================================
# PRE-EXISTING COMPONENTS — copied unchanged from `cgnet_arch.py`.
# 原檔既有元件（自 cgnet_arch.py 原樣複製，非本次新增）。
# ==========================================================================

class LayerNormFunction(torch.autograd.Function):
    """Custom autograd function for 2D Layer Normalization across channels."""

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
    """2D Layer Normalization across spatial dimensions (N, C, H, W)."""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple Gate parameter-free activation block."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock0(nn.Module):
    """Standard NAFNet block used in middle and decoding stages."""

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


# ==========================================================================
# NEW COMPONENTS — introduced by TrueCascadeCGNet.
# 本次新增元件。
# ==========================================================================

class ReflectStridedDWSeparableConv(nn.Module):
    """Strided depthwise-separable conv with reflect padding (no border truncation).

    The symmetric `kernel_size // 2` padding guarantees the output covers the
    whole input area: output size is ceil-like `floor((H + 2*(k//2) - k) / s) + 1`
    instead of dropping up to `k - 1` border rows/cols as `padding=0` did.
    """

    def __init__(self, channels, kernel_size, stride):
        super().__init__()
        self.pad = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, stride=stride,
                                   padding=0, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = pad_safe(x, (self.pad, self.pad, self.pad, self.pad))
        return self.pointwise(self.depthwise(x))


class TrueCascadeGlobalContextExtractor(nn.Module):
    """Multi-scale context extractor with genuine cascading.

    Each conv consumes the PREVIOUS conv's output (`x = conv(x)`), so the
    downsampling strides compound across levels and the deepest output sees a
    receptive field of `prod(strides)` — the behaviour the "CascadedGaze" name
    promises. Returns one feature map per level, coarsest last.
    """

    def __init__(self, c, kernel_sizes=(3, 3), strides=(2, 3)):
        super().__init__()
        if len(kernel_sizes) != len(strides):
            raise ValueError(f'kernel_sizes {kernel_sizes} and strides {strides} must match in length')
        self.convs = nn.ModuleList([
            ReflectStridedDWSeparableConv(c, k, s) for k, s in zip(kernel_sizes, strides)
        ])
        self.strides = tuple(strides)

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            x = F.gelu(conv(x))  # cascaded: next level consumes this output
            outputs.append(x)
        return outputs

    def extra_repr(self):
        return f'strides={self.strides} (compounding)'


class TrueCascadeGazeBlock(nn.Module):
    """CascadedGazeBlock with the TrueCascade / Reflect / Bilinear fixes applied.

    Structure (identical to the original block apart from the fixes):
        LayerNorm -> 1x1 expand (2c) -> 3x3 depthwise -> GELU
        -> halves summed -> cascaded GCE -> bilinear upsample of each level
        -> concat with trunk -> SCA -> 1x1 project -> residual (beta)
        -> LayerNorm -> 1x1 expand -> SimpleGate -> 1x1 project -> residual (gamma)
    """

    def __init__(self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1,
                               groups=self.dw_channel, bias=True)

        if GCE_Conv == 3:
            self.GCE = TrueCascadeGlobalContextExtractor(c, kernel_sizes=(3, 3, 5), strides=(2, 3, 4))
        elif GCE_Conv == 2:
            self.GCE = TrueCascadeGlobalContextExtractor(c, kernel_sizes=(3, 3), strides=(2, 3))
        else:
            raise ValueError(f'GCE_Conv must be 2 or 3, got {GCE_Conv}')

        # Concatenation: trunk (2c) + one c-channel map per GCE level.
        cat_channels = self.dw_channel + GCE_Conv * c
        self.project_out = nn.Conv2d(cat_channels, c, kernel_size=1)
        # NAFNet-style simplified channel attention (deliberately no sigmoid),
        # matching the original block.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cat_channels, cat_channels, kernel_size=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        b, c, h, w = inp.shape

        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        # Cascaded global context on the summed halves of the expanded trunk.
        x_1, x_2 = x.chunk(2, dim=1)
        context = self.GCE(x_1 + x_2)
        context = [
            F.interpolate(level, size=(h, w), mode='bilinear', align_corners=False)
            for level in context
        ]
        x = torch.cat([x, *context], dim=1)

        x = self.sca(x) * x
        x = self.project_out(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # Channel-mixing FFN.
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class TrueCascadeCGNet(CGNetUNetBase):
    """CGNet with TrueCascade / Reflect / Bilinear / DecoderContext fixes.

    Stage layout (same as the original CGNet, plus DecoderContext):
        encoders: TrueCascadeGazeBlock everywhere,
        middle:   NAFBlock0,
        decoders: NAFBlock0 everywhere, except the full-resolution stage whose
                  first block is a TrueCascadeGazeBlock.

    Config-compatible with CGNet (`GCE_CONVS_nums` chooses 2 or 3 GCE levels
    per encoder stage, defaulting to 2 when unspecified).
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1),
                 GCE_CONVS_nums=(), DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        gce_convs_nums = list(GCE_CONVS_nums)

        def gaze_block(chan, gce_convs):
            return TrueCascadeGazeBlock(chan, GCE_Conv=gce_convs, DW_Expand=DW_Expand,
                                        FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate)

        def naf_block(chan):
            return NAFBlock0(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                             drop_out_rate=drop_out_rate)

        def make_encoder_stage(index, num_stages, chan, num_blocks):
            gce_convs = gce_convs_nums[index] if index < len(gce_convs_nums) else 2
            return nn.Sequential(*[gaze_block(chan, gce_convs) for _ in range(num_blocks)])

        def make_middle_stage(chan, num_blocks):
            return nn.Sequential(*[naf_block(chan) for _ in range(num_blocks)])

        def make_decoder_stage(index, num_stages, chan, num_blocks):
            is_full_resolution = (index == num_stages - 1)
            if is_full_resolution and num_blocks >= 1:
                # [DecoderContext] give the reconstruction path one context block.
                blocks = [gaze_block(chan, 2)] + [naf_block(chan) for _ in range(num_blocks - 1)]
            else:
                blocks = [naf_block(chan) for _ in range(num_blocks)]
            return nn.Sequential(*blocks)

        super().__init__(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
                         make_encoder_stage=make_encoder_stage,
                         make_middle_stage=make_middle_stage,
                         make_decoder_stage=make_decoder_stage,
                         pad_mode='reflect')