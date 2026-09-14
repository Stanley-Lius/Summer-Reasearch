import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.restormer_model import RestormerModel

@MODEL_REGISTRY.register()
class OmniRestormerModel(RestormerModel):
    """
    OmniRestormer Model for Image Restoration.
    
    Inherits from RestormerModel to reuse the progressive patch training 
    and mixup augmentation logic, but overrides the test function 
    because OmniRestormerNet handles its own reflection padding internally.
    """
    def __init__(self, opt):
        super(OmniRestormerModel, self).__init__(opt)

    def test(self):
        """Override test because OmniRestormerNet handles padding automatically."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g

        with torch.no_grad():
            preds = net(self.lq)
            if isinstance(preds, (list, tuple)):
                self.output = preds[-1]
            else:
                self.output = preds

        self.net_g.train()

