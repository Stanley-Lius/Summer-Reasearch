import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger

@MODEL_REGISTRY.register()
class LatentDistillationCGNetModel(SRModel):
    """Latent Distillation Image Restoration Model.
    
    Uses a frozen pre-trained Teacher (e.g., CGNet) to guide a Student 
    (e.g., DistilledVectorizedDynamicCGNet) via latent space feature alignment.
    """

    def __init__(self, opt):
        super(LatentDistillationCGNetModel, self).__init__(opt)
        self.teacher_latent = None

    def init_training_settings(self):
        super(LatentDistillationCGNetModel, self).init_training_settings()
        
        # Initialize Teacher Network
        self.net_teacher = build_network(self.opt['network_teacher'])
        self.net_teacher = self.model_to_device(self.net_teacher)
        self.print_network(self.net_teacher)
        
        # Load Teacher weights
        load_path = self.opt['path'].get('pretrain_network_teacher', None)
        if load_path is not None:
            logger = get_root_logger()
            logger.info(f'Loading teacher model from {load_path}')
            self.load_network(self.net_teacher, load_path, True, 'params_ema')
        else:
            raise ValueError("pretrain_network_teacher path is required for distillation.")
        
        # Freeze Teacher parameters
        for param in self.net_teacher.parameters():
            param.requires_grad = False
        self.net_teacher.eval()
        
        # Register Forward Hook on the Teacher's middle_blks to capture latent features
        def hook_fn(module, input, output):
            self.teacher_latent = output
        
        self.hook_handle = self.net_teacher.middle_blks.register_forward_hook(hook_fn)
        
        # Read latent loss weight from options
        train_opt = self.opt['train']
        self.latent_loss_weight = train_opt.get('latent_opt', {}).get('loss_weight', 0.1)
        logger = get_root_logger()
        logger.info(f'Using Latent Distillation (Cosine Similarity Loss) with weight: {self.latent_loss_weight}')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # 1. Forward Student (Returns both image and latent feature during training)
        out_img, student_latent = self.net_g(self.lq)
        self.output = out_img
        
        # 2. Forward Teacher (No grad)
        with torch.no_grad():
            self.net_teacher(self.gt)
            # The forward hook will populate self.teacher_latent
            teacher_latent = self.teacher_latent
            
        l_total = 0
        loss_dict = OrderedDict()
        
        # 3. Calculate Image-space Reconstruction Loss (Pixel Loss)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # 4. Calculate Latent Feature Loss (Cosine Similarity)
        # Cosine similarity is computed along the channel dimension (dim=1)
        # We want to minimize the distance, so the loss is 1 - cos_sim
        cos_sim = F.cosine_similarity(student_latent, teacher_latent, dim=1)
        l_latent = (1.0 - cos_sim).mean() * self.latent_loss_weight
        
        l_total += l_latent
        loss_dict['l_latent'] = l_latent

        # 5. Backward and Step
        l_total.backward()
        
        if self.opt['train'].get('use_grad_clip', False):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train'].get('grad_clip_norm', 0.01))
            
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
