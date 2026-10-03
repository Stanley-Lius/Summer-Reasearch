import torch
from ptflops import get_model_complexity_info

def calculate_flops(net, input_shape=(3, 256, 256), verbose=False, print_per_layer_stat=True):
    """Calculate FLOPs/MACs and parameters of a model.
    
    Args:
        net (nn.Module): The network model.
        input_shape (tuple): The shape of the input tensor (C, H, W).
        verbose (bool): Whether to print verbose information in ptflops.
        print_per_layer_stat (bool): Whether to print statistics per layer.
        
    Returns:
        tuple: (macs_str, params_str) as formatted strings.
    """
    try:
        # Check if CUDA is available, as ptflops runs faster on GPU sometimes,
        # but model needs to be on the same device as what ptflops creates inside
        device = next(net.parameters()).device
        macs, params = get_model_complexity_info(
            net, 
            input_shape, 
            as_strings=True, 
            print_per_layer_stat=print_per_layer_stat, 
            verbose=verbose,
            ost=None # Use default print
        )
        print(f"\\n[FLOPs/MACs Info] MACs: {macs}, Params: {params}")
        return macs, params
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None, None
