import numpy as np
import random
import torch
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


class MixingAugment:
    """Mixup augmentation for image restoration."""

    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


@MODEL_REGISTRY.register()
class RestormerModel(SRModel):
    """Restormer Model for Image Restoration (Denoising, Deblurring, etc.).

    Integrates progressive patch size training and mixup data augmentation.
    """

    def __init__(self, opt):
        super(RestormerModel, self).__init__(opt)
        train_opt = self.opt.get('train', {})
        mixing_augs = train_opt.get('mixing_augs', {})
        self.mixing_flag = mixing_augs.get('mixup', False)
        if self.mixing_flag:
            mixup_beta = mixing_augs.get('mixup_beta', 1.2)
            use_identity = mixing_augs.get('use_identity', False)
            self.mixing_augmentation = MixingAugment(mixup_beta, use_identity, self.device)

    def optimize_parameters(self, current_iter):
        # 1. Progressive patch size & mini batch size cropping
        train_opt_ds = self.opt.get('datasets', {}).get('train', {})
        iters = train_opt_ds.get('iters')
        mini_gt_sizes = train_opt_ds.get('gt_sizes')
        mini_batch_sizes = train_opt_ds.get('mini_batch_sizes')
        scale = self.opt.get('scale', 1)

        if iters is not None and mini_gt_sizes is not None and mini_batch_sizes is not None:
            groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if mini_batch_size < self.lq.size(0):
                indices = random.sample(range(0, self.lq.size(0)), k=mini_batch_size)
                self.lq = self.lq[indices]
                if hasattr(self, 'gt'):
                    self.gt = self.gt[indices]

            h_lq, w_lq = self.lq.size(-2), self.lq.size(-1)
            mini_lq_size = mini_gt_size // scale
            if mini_lq_size < h_lq and mini_lq_size < w_lq:
                x0 = int((h_lq - mini_lq_size) * random.random())
                y0 = int((w_lq - mini_lq_size) * random.random())
                x1 = x0 + mini_lq_size
                y1 = y0 + mini_lq_size
                self.lq = self.lq[:, :, x0:x1, y0:y1]
                if hasattr(self, 'gt'):
                    self.gt = self.gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]

        # 2. Mixing Augmentation (Mixup)
        if self.mixing_flag and hasattr(self, 'gt'):
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

        # 3. Optimize parameters via base class logic
        super().optimize_parameters(current_iter)

    def test(self):
        """Override test to support arbitrary resolution images via reflection padding."""
        window_size = self.opt.get('val', {}).get('window_size', 8)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g

        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        with torch.no_grad():
            preds = net(img)
            if isinstance(preds, list):
                preds = preds[-1]
            output = preds

        _, _, out_h, out_w = output.size()
        self.output = output[:, :, 0:out_h - mod_pad_h * scale, 0:out_w - mod_pad_w * scale]
        self.net_g.train()


@MODEL_REGISTRY.register()
class ImageCleanModel(RestormerModel):
    """Alias for RestormerModel to maintain compatibility with official Restormer configs."""
    pass
