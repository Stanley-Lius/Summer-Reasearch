# ------------------------------------------------------------------------
# Hybrid U-Net:
#   - encoder: use CascadedGazeBlock
#   - deepest encoder / middle / shallowest decoder: keep original VectorizedDynamicGazeBlock
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.cascadedgaze_arch import CascadedGazeBlock, NAFBlock0
from basicsr.archs.cgnet_arch import LayerNorm2d, SimpleGate
from basicsr.utils.registry import ARCH_REGISTRY


class DilatedContextBranch(nn.Module):
    """Fixed-shape candidate receptive field implemented with a dilated convolution."""

    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv2d(c, c, kernel_size, padding=padding, dilation=dilation, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        return F.gelu(self.pw(self.dw(x)))


class StridedContextBranch(nn.Module):
    """Fixed-shape large-receptive-field candidate using downsample and upsample."""

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
    """Content-adaptive mixture of fixed-shape receptive-field branches."""

    def __init__(self, c, num_scales=4, max_radius=4, pattern_stride=3):
        super().__init__()
        assert num_scales >= 2, 'need at least a local and a wide branch'
        self.c = c
        self.max_radius = max_radius
        self.pattern_stride = pattern_stride
        self.tau = 1.0

        dilations = [2 ** i for i in range(num_scales - 1)]
        self.branches = nn.ModuleList(
            [DilatedContextBranch(c, kernel_size=3, dilation=d) for d in dilations] +
            [StridedContextBranch(c, kernel_size=5, stride=4)]
        )
        self.num_scales = len(self.branches)

        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=pattern_stride, padding=0, groups=c, bias=False)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.T = nn.Parameter(torch.tensor(0.0))
        self.gate_mlp = nn.Linear(1, self.num_scales)
        self.fallback_gate_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.fuse_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def _compute_gate(self, x):
        b, c, h, w = x.shape
        fallback = F.softmax(self.fallback_gate_logits, dim=0).expand(b, -1)
        if h < self.pattern_stride * 2 or w < self.pattern_stride * 2:
            return fallback

        pad_h = (self.pattern_stride - h % self.pattern_stride) % self.pattern_stride
        pad_w = (self.pattern_stride - w % self.pattern_stride) % self.pattern_stride
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x

        pattern = self.pattern_conv(x_pad)
        hp, wp = pattern.shape[-2:]
        n = hp * wp
        if hp < 2 or wp < 2:
            return fallback

        q = self.q_conv(pattern).flatten(2)
        k = self.k_conv(pattern).flatten(2)

        r = max(1, min(self.max_radius, hp - 1, wp - 1))
        s = torch.arange(1, r + 1, device=x.device)
        idx_tl = torch.clamp(s * (wp + 1), max=n - 1)
        idx_tr = torch.clamp((wp - 1) + s * (wp - 1), max=n - 1)

        q_tl = q[:, :, 0]
        q_tr = q[:, :, wp - 1]
        k_tl = k.index_select(2, idx_tl)
        k_tr = k.index_select(2, idx_tr)
        score_tl = (q_tl.unsqueeze(-1) * k_tl).sum(dim=1) / (c ** 0.5)
        score_tr = (q_tr.unsqueeze(-1) * k_tr).sum(dim=1) / (c ** 0.5)

        valid_tl = torch.sigmoid((score_tl - self.T) / self.tau)
        valid_tr = torch.sigmoid((score_tr - self.T) / self.tau)
        reach_tl = torch.cumprod(valid_tl, dim=1).sum(dim=1)
        reach_tr = torch.cumprod(valid_tr, dim=1).sum(dim=1)
        reach = torch.maximum(reach_tl, reach_tr) / r

        return F.softmax(self.gate_mlp(reach.unsqueeze(-1)), dim=-1)

    def forward(self, x):
        b = x.shape[0]
        branch_outs = torch.stack([branch(x) for branch in self.branches], dim=1)
        gate = self._compute_gate(x).view(b, self.num_scales, 1, 1, 1)
        return self.fuse_conv((branch_outs * gate).sum(dim=1))


class VectorizedDynamicGazeBlock(nn.Module):
    """CGNet-style block with sparse-query vectorized dynamic context."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                 context_num_scales=4, context_max_radius=4):
        super().__init__()
        self.dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1,
                               groups=self.dw_channel, bias=True)
        self.sg1 = SimpleGate()
        context_channels = self.dw_channel // 2
        self.context = VectorizedDynamicContext(context_channels, num_scales=context_num_scales,
                                                max_radius=context_max_radius)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, kernel_size=1, bias=True), nn.GELU(),
            nn.Conv2d(c, c, kernel_size=1, bias=True), nn.Sigmoid())
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
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg1(x)
        x = self.context(x)
        x = self.dropout1(x * self.sca(x))
        y = inp + x * self.beta
        ffn = self.conv5(self.sg2(self.conv_ffn_dw(self.conv4(self.norm2(y)))))
        return y + self.dropout2(ffn) * self.gamma


@ARCH_REGISTRY.register()
class CascadedGazeEncoderVectorizedDynamicCGNet(nn.Module):
    """Hybrid CGNet + VectorizedDynamicCGNet.

    - Encoder stages use CascadedGazeBlock to insert CG-style context in the encoder.
    - Original VectorizedDynamicGazeBlock is retained at the deepest encoder stage,
      the middle blocks, and the shallowest decoder stage, matching the original
      VectorizedDynamicCGNet design philosophy.
    """

    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1],
                 dec_blk_nums=[1, 1, 1, 1],
                 GCE_CONVS_nums=[2, 2, 2, 2],
                 DW_Expand=2,
                 FFN_Expand=2,
                 drop_out_rate=0.,
                 context_num_scales=4,
                 context_max_radius=4):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        def make_cascaded_stage(chan, num_blocks, gce_convs):
            return nn.Sequential(*[
                CascadedGazeBlock(
                    chan,
                    GCE_Conv=gce_convs,
                    DW_Expand=DW_Expand,
                    FFN_Expand=FFN_Expand,
                    drop_out_rate=drop_out_rate
                ) for _ in range(num_blocks)
            ])

        def make_vectorized_stage(chan, num_blocks):
            return nn.Sequential(*[
                VectorizedDynamicGazeBlock(
                    chan,
                    DW_Expand=DW_Expand,
                    FFN_Expand=FFN_Expand,
                    drop_out_rate=drop_out_rate,
                    context_num_scales=context_num_scales,
                    context_max_radius=context_max_radius
                ) for _ in range(num_blocks)
            ])

        def make_naf_stage(chan, num_blocks):
            return nn.Sequential(*[
                NAFBlock0(
                    chan,
                    DW_Expand=DW_Expand,
                    FFN_Expand=FFN_Expand,
                    drop_out_rate=drop_out_rate
                ) for _ in range(num_blocks)
            ])

        chan = width
        for i, num in enumerate(enc_blk_nums):
            is_deepest = (i == len(enc_blk_nums) - 1)
            gce_convs = GCE_CONVS_nums[i] if i < len(GCE_CONVS_nums) else 2

            if is_deepest:
                stage = make_vectorized_stage(chan, num)
            else:
                stage = make_cascaded_stage(chan, num, gce_convs)

            self.encoders.append(stage)
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = make_vectorized_stage(chan, middle_blk_num)

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            is_shallowest = (i == 0)
            if is_shallowest:
                self.decoders.append(make_vectorized_stage(chan, num))
            else:
                self.decoders.append(make_naf_stage(chan, num))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _, _, H, W = inp.shape
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


@ARCH_REGISTRY.register()
class CascadedGazeEncoderCGNet(CascadedGazeEncoderVectorizedDynamicCGNet):
    """Alias for compatibility."""
    pass