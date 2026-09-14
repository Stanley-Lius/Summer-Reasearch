# ------------------------------------------------------------------------
# Shared U-Net scaffolding for the CGNet architecture family.
#
# All CGNet variants (CGNet / OmniDirDynamicCGNet / VectorizedDynamicCGNet and
# their improved versions) share the exact same macro structure:
#     intro 3x3 conv
#       -> [encoder stage -> stride-2 down] * L
#       -> middle blocks
#       -> [PixelShuffle up -> additive skip -> decoder stage] * L
#     -> ending 3x3 conv -> global residual
# Only the *blocks inside each stage* differ. This module factors that skeleton
# out once so each new variant only has to describe its stage contents, keeping
# the per-arch files short and the forward pass identical (and comparable)
# across variants.
#
# NOTE: this file intentionally does NOT end with `_arch.py`, so it is not
# scanned by `basicsr.archs.__init__` and registers nothing by itself.
# ------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


def pad_safe(x, pad, preferred_mode='reflect'):
    """`F.pad` that degrades gracefully instead of raising on small inputs.

    `reflect` padding requires every pad amount to be strictly smaller than the
    corresponding spatial dimension. Deep U-Net stages can easily produce 4x4
    (or smaller) feature maps where a large directional pad violates that, so
    we fall back to `replicate` (which only needs a non-empty input) in that
    case rather than crashing mid-training.

    Args:
        x: input tensor of shape (B, C, H, W).
        pad: 4-tuple (left, right, top, bottom), same convention as `F.pad`.
        preferred_mode: padding mode to use whenever the input is large enough.

    Returns:
        The padded tensor.
    """
    left, right, top, bottom = pad
    if left == right == top == bottom == 0:
        return x
    h, w = x.shape[-2:]
    mode = preferred_mode
    if mode == 'reflect' and (max(left, right) >= w or max(top, bottom) >= h):
        mode = 'replicate'
    return F.pad(x, pad, mode=mode)


class CGNetUNetBase(nn.Module):
    """U-Net skeleton with pluggable stage factories.

    Subclasses do not override `forward`; they only supply three callables that
    build the blocks of each stage. This guarantees every variant runs the very
    same encoder/skip/decoder wiring, so PSNR differences between variants can
    be attributed to the blocks alone.

    Args:
        img_channel: number of image channels (3 for RGB).
        width: base channel width; doubled after every encoder stage.
        middle_blk_num: number of blocks in the bottleneck stage.
        enc_blk_nums: blocks per encoder stage, shallow -> deep.
        dec_blk_nums: blocks per decoder stage, deep -> shallow
            (index 0 is the lowest-resolution decoder stage, right after the
            bottleneck; the last index is the full-resolution stage).
        make_encoder_stage: callable (stage_index, num_stages, channels,
            num_blocks) -> nn.Module building one encoder stage.
        make_middle_stage: callable (channels, num_blocks) -> nn.Module.
        make_decoder_stage: callable (stage_index, num_stages, channels,
            num_blocks) -> nn.Module building one decoder stage.
        pad_mode: padding mode used by `check_image_size` to round the input
            up to a multiple of the total downsampling factor. 'reflect'
            avoids the dark-border artifacts of zero padding.
    """

    def __init__(self, *, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1),
                 make_encoder_stage, make_middle_stage, make_decoder_stage,
                 pad_mode='reflect'):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError(
                f'enc_blk_nums and dec_blk_nums must have the same length, '
                f'got {len(enc_blk_nums)} vs {len(dec_blk_nums)}')

        self.img_channel = img_channel
        self.pad_mode = pad_mode

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        num_stages = len(enc_blk_nums)
        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(make_encoder_stage(i, num_stages, chan, num))
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = make_middle_stage(chan, middle_blk_num)

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan = chan // 2
            self.decoders.append(make_decoder_stage(i, num_stages, chan, num))

        self.padder_size = 2 ** num_stages

    def forward(self, inp):
        if inp.dim() != 4 or inp.shape[1] != self.img_channel:
            raise ValueError(
                f'{self.__class__.__name__} expects input of shape '
                f'(B, {self.img_channel}, H, W), got {tuple(inp.shape)}')

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
        """Pad the input so H and W are multiples of the total downsampling factor."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = pad_safe(x, (0, mod_pad_w, 0, mod_pad_h), preferred_mode=self.pad_mode)
        return x

    def extra_repr(self):
        return f'img_channel={self.img_channel}, padder_size={self.padder_size}, pad_mode={self.pad_mode}'