import sys
import yaml
import torch
import argparse
from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs # to trigger registrations
from basicsr.utils.flop_util import calculate_flops

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_net(opt):
    net_opt = opt['network_g'].copy()
    net_type = net_opt.pop('type')
    net = ARCH_REGISTRY.get(net_type)(**net_opt)
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yml', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=False)
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256], help='Height Width')
    args = parser.parse_args()

    opt = load_yaml(args.yml)
    net = build_net(opt)
    
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        # typical basicsr checkpoints have 'params' or 'params_ema'
        state_dict = checkpoint.get('params', checkpoint)
        try:
            net.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: strict load failed, trying non-strict: {e}")
            net.load_state_dict(state_dict, strict=False)
            
    net = net.cuda(); net.eval()
    
    print(f"Measuring {args.yml} with input size {args.size[0]}x{args.size[1]}...")
    macs, params = calculate_flops(net, input_shape=(3, args.size[0], args.size[1]), print_per_layer_stat=False)
    print(f"Result for [{args.yml}] at {args.size[0]}x{args.size[1]}: MACs={macs}, Params={params}")

if __name__ == '__main__':
    main()
