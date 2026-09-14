import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger

@LOSS_REGISTRY.register()
class CosineSimilarityLoss(nn.Module):
    """Cosine Similarity Loss for Latent Distillation."""
    def __init__(self, loss_weight=1.0):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, pred, target):
        # Cosine similarity is computed along the channel dimension (dim=1)
        # Minimize the distance (1 - cos_sim)
        return (1.0 - F.cosine_similarity(pred, target, dim=1)).mean() * self.loss_weight

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
        # Calculate latent channel sizes to handle mismatched width between Student and Teacher
        student_w = self.opt['network_g'].get('width', 16)
        student_encs = len(self.opt['network_g'].get('enc_blk_nums', []))
        student_channels = student_w * (2 ** student_encs)
        
        teacher_w = self.opt['network_teacher'].get('width', 16)
        teacher_encs = len(self.opt['network_teacher'].get('enc_blk_nums', []))
        teacher_channels = teacher_w * (2 ** teacher_encs)
        
        # Create a 1x1 Conv projection layer if channels mismatch
        if student_channels != teacher_channels:
            self.feature_proj = nn.Conv2d(student_channels, teacher_channels, kernel_size=1).to(self.device)
            get_root_logger().info(f'Created 1x1 projection layer from {student_channels} to {teacher_channels} channels.')
        else:
            self.feature_proj = None

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
        
        # Build Latent Loss dynamically
        train_opt = self.opt['train']
        if train_opt.get('latent_opt'):
            self.cri_latent = build_loss(train_opt['latent_opt']).to(self.device)
            logger = get_root_logger()
            logger.info(f'Initialized Latent Loss: {self.cri_latent.__class__.__name__}')
        else:
            self.cri_latent = None
            logger = get_root_logger()
            logger.warning('No latent_opt provided. Latent Distillation is disabled.')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
                
        # Add feature projection parameters to optimizer
        if hasattr(self, 'feature_proj') and self.feature_proj is not None:
            for k, v in self.feature_proj.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # 1. Forward Student (Returns both image and latent feature during training)
        out_img, student_latent = self.net_g(self.lq)
        self.output = out_img
        
        # 2. Forward Teacher (No grad)
        with torch.no_grad():
            self.net_teacher(self.gt)
            teacher_latent = self.teacher_latent
            
        l_total = 0
        loss_dict = OrderedDict()
        
        # 3. Calculate Image-space Reconstruction Loss (Pixel Loss)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # 4. Calculate Latent Feature Loss dynamically
        if self.cri_latent:
            if hasattr(self, 'feature_proj') and self.feature_proj is not None:
                student_latent = self.feature_proj(student_latent)
            l_latent = self.cri_latent(student_latent, teacher_latent)
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
