# ------------------------------------------------------------------------
# CascadedGaze Model (CGNetModel)
# Restructured for high readability, maintainability, and modularity.
# Supports single-head and multi-head restoration training workflows.
# ------------------------------------------------------------------------

import random
import torch
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


class MixupAugmentation:
    """Beta distribution mixup data augmentation helper for image restoration."""

    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity

    def __call__(self, target, input_):
        if self.use_identity and random.random() < 0.5:
            return target, input_

        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_


@MODEL_REGISTRY.register()
class CGNetModel(SRModel):
    """CascadedGaze Model for Image Restoration.

    Supports both single-output (CGNet) and multi-head (GCENetMH) architectures,
    optional gradient clipping, and mixup augmentation.
    """

    def __init__(self, opt):
        super(CGNetModel, self).__init__(opt)
        train_opt = self.opt.get('train', {})

        # Setup optional Mixup Augmentation
        mixing_augs = train_opt.get('mixing_augs', {})
        self.mixing_flag = mixing_augs.get('mixup', False) or train_opt.get('mixup', False)
        if self.mixing_flag:
            mixup_beta = mixing_augs.get('mixup_beta', 1.2)
            use_identity = mixing_augs.get('use_identity', True)
            self.mixup_helper = MixupAugmentation(mixup_beta, use_identity, self.device)

        # Gradient clipping settings
        self.use_grad_clip = train_opt.get('use_grad_clip', True)
        self.grad_clip_norm = train_opt.get('grad_clip_norm', 0.01)

    def optimize_parameters(self, current_iter):
        if not self.use_grad_accum or (current_iter - 1) % self.accum_iter == 0:
            self.optimizer_g.zero_grad()

        # Apply Mixup Augmentation if configured
        if self.mixing_flag and self.gt is not None:
            self.gt, self.lq = self.mixup_helper(self.gt, self.lq)

        preds = self.net_g(self.lq)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        # Use the final head/prediction as the primary output for logging/metrics
        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        # Calculate Pixel Loss across all predictions (for multi-head supervision)
        if self.cri_pix:
            l_pix = 0.
            for idx, pred in enumerate(preds):
                l_pix += self.cri_pix(pred, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Calculate Perceptual Loss on final prediction
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        loss_to_backward = l_total / self.accum_iter if self.use_grad_accum else l_total
        loss_to_backward.backward()

        total_iter = self.opt.get('train', {}).get('total_iter', -1)
        if not self.use_grad_accum or current_iter % self.accum_iter == 0 or current_iter == total_iter:
            # Apply gradient clipping if enabled
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)

            self.optimizer_g.step()
            if self.use_grad_accum:
                self.optimizer_g.zero_grad()
            if getattr(self, 'ema_decay', 0) > 0:
                self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        """Test forward pass with automatic extraction of final head if output is a list, 
        and arbitrary resolution reflection padding."""
        window_size = self.opt.get('val', {}).get('window_size', 0)
        
        self.net_g.eval()
        
        if window_size > 0:
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
                
            img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        else:
            img = self.lq

        with torch.no_grad():
            preds = self.net_g(img)
            if isinstance(preds, (list, tuple)):
                output = preds[-1]
            else:
                output = preds

        if window_size > 0:
            _, _, out_h, out_w = output.size()
            self.output = output[:, :, 0:out_h - mod_pad_h * scale, 0:out_w - mod_pad_w * scale]
        else:
            self.output = output
            
        self.net_g.train()


@MODEL_REGISTRY.register()
class ImageRestorationModel(CGNetModel):
    """Alias for CGNetModel matching official CascadedGaze training configurations."""
    pass


@MODEL_REGISTRY.register()
class MultiHeadImageRestorationModel(CGNetModel):
    """Alias for CGNetModel matching multi-head training configurations."""
    pass
