import torch
from torchview import draw_graph
from basicsr.archs.irregular_cgnet_arch import IrregularCGNet, IrregularGCELayer
from basicsr.archs.dynamic_cgnet_arch import DynamicCGNet, DynamicGCELayer

def generate_torchview_graphs():
    print("Generating Torchview graph for IrregularGCELayer...")
    layer_irr = IrregularGCELayer(c=32, kernel_size=7)
    model_graph_irr = draw_graph(
        layer_irr, input_size=(1, 32, 64, 64),
        graph_name='IrregularGCELayer Computation Graph',
        save_graph=True, filename='assets/architecture_diagrams/torchview_IrregularGCELayer'
    )
    
    print("Generating Torchview graph for DynamicGCELayer...")
    layer_dyn = DynamicGCELayer(c=32, kernel_size=7)
    model_graph_dyn = draw_graph(
        layer_dyn, input_size=(1, 32, 64, 64),
        graph_name='DynamicGCELayer Computation Graph',
        save_graph=True, filename='assets/architecture_diagrams/torchview_DynamicGCELayer'
    )

    print("Generating Torchview graph for Full IrregularCGNet (depth=2)...")
    net_irr = IrregularCGNet()
    full_graph_irr = draw_graph(
        net_irr, input_size=(1, 3, 64, 64),
        depth=2,
        graph_name='Full IrregularCGNet (Depth=2)',
        save_graph=True, filename='assets/architecture_diagrams/torchview_Full_IrregularCGNet'
    )

    print("Generating Torchview graph for Full DynamicCGNet (depth=2)...")
    net_dyn = DynamicCGNet()
    full_graph_dyn = draw_graph(
        net_dyn, input_size=(1, 3, 64, 64),
        depth=2,
        graph_name='Full DynamicCGNet (Depth=2)',
        save_graph=True, filename='assets/architecture_diagrams/torchview_Full_DynamicCGNet'
    )
    print("All Torchview graphs generated successfully!")

if __name__ == '__main__':
    generate_torchview_graphs()
