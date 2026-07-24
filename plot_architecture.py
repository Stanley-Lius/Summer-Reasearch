import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import numpy as np

def create_box(ax, xy, width, height, boxstyle, facecolor, edgecolor, lw=2, label="", fontsize=11, fontcolor='white'):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=boxstyle,
        facecolor=facecolor, edgecolor=edgecolor, lw=lw
    )
    ax.add_patch(box)
    if label:
        lines = label.split('\n')
        y_center = xy[1] + height / 2
        ax.text(xy[0] + width / 2, y_center, label,
                ha='center', va='center', fontsize=fontsize, color=fontcolor, weight='bold')
    return box

def draw_arrow(ax, start, end, color='#333333', lw=2, style='->'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0))

def plot_dynamic_cgnet():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.title("DynamicCGNet Architecture & Uniform Patching Mechanism", fontsize=18, weight='bold', pad=20, color='#1e293b')

    # Main U-Net Backbone
    create_box(ax, (0.5, 6.8), 2.2, 1.2, "round,pad=0.1", "#3b82f6", "#1d4ed8", label="Input LQ\nImage")
    create_box(ax, (3.5, 6.8), 2.5, 1.2, "round,pad=0.1", "#10b981", "#047857", label="Encoder Stages\n(DynamicGazeBlocks)")
    create_box(ax, (6.8, 6.8), 2.0, 1.2, "round,pad=0.1", "#8b5cf6", "#6d28d9", label="Downsampling\n(Conv 2x2)")
    create_box(ax, (9.5, 6.8), 2.2, 1.2, "round,pad=0.1", "#f59e0b", "#b45309", label="Middle Block\n(DynamicGazeBlocks)")
    
    draw_arrow(ax, (2.7, 7.4), (3.5, 7.4))
    draw_arrow(ax, (6.0, 7.4), (6.8, 7.4))
    draw_arrow(ax, (8.8, 7.4), (9.5, 7.4))

    # Down arrow to decoder
    draw_arrow(ax, (10.6, 6.8), (10.6, 5.5))

    create_box(ax, (9.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#ec4899", "#be185d", label="Upsampling\n(PixelShuffle)")
    create_box(ax, (6.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#10b981", "#047857", label="Decoder Stages\n(DynamicGazeBlocks)")
    create_box(ax, (3.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#3b82f6", "#1d4ed8", label="Output HQ\nRestored Image")

    draw_arrow(ax, (9.5, 4.9), (8.7, 4.9))
    draw_arrow(ax, (6.5, 4.9), (5.7, 4.9))

    # Skip connection
    draw_arrow(ax, (4.7, 6.8), (4.7, 5.7), color='#64748b', style='->', lw=1.5)
    draw_arrow(ax, (4.7, 5.7), (7.6, 5.7), color='#64748b', style='-', lw=1.5)
    draw_arrow(ax, (7.6, 5.7), (7.6, 5.5), color='#64748b', style='->', lw=1.5)
    ax.text(6.1, 5.85, "Skip Connection (+)", fontsize=10, color='#475569', weight='bold')

    # Zoom in block on DynamicGCELayer
    create_box(ax, (0.5, 0.5), 13.0, 3.2, "round,pad=0.2", "#f8fafc", "#cbd5e1", lw=1.5, fontcolor='#0f172a')
    ax.text(1.0, 3.3, "Deep Dive: DynamicGCELayer (Uniform Dynamic Grid)", fontsize=14, weight='bold', color='#1e293b')

    # Dynamic GCE steps
    create_box(ax, (1.0, 1.3), 2.2, 1.4, "round,pad=0.1", "#38bdf8", "#0284c7", label="Input Feature X\n(B, C, H, W)")
    create_box(ax, (3.8, 1.3), 2.5, 1.4, "round,pad=0.1", "#818cf8", "#4f46e5", label="Attention Q/K Conv\nPredict Single Size S_x")
    create_box(ax, (7.0, 1.3), 2.5, 1.4, "round,pad=0.1", "#34d399", "#059669", label="Uniform Patchify\nAll Patches = S_x * S_x")
    create_box(ax, (10.2, 1.3), 2.8, 1.4, "round,pad=0.1", "#fb923c", "#ea580c", label="Local DW + PW Conv\n& Unpatchify")

    draw_arrow(ax, (3.2, 2.0), (3.8, 2.0))
    draw_arrow(ax, (6.3, 2.0), (7.0, 2.0))
    draw_arrow(ax, (9.5, 2.0), (10.2, 2.0))

    # Grid visual inside Uniform patchify
    ax.text(8.25, 0.8, "Uniform 3x3 or 9x9 Grid", fontsize=10, color='#047857', weight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig("assets/architecture_diagrams/DynamicCGNet_arch.png", bbox_inches='tight')
    plt.close()

def plot_irregular_cgnet():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.title("IrregularCGNet Architecture & Recursive k-d Tree Patching Mechanism", fontsize=18, weight='bold', pad=20, color='#1e293b')

    # Main U-Net Backbone
    create_box(ax, (0.5, 6.8), 2.2, 1.2, "round,pad=0.1", "#3b82f6", "#1d4ed8", label="Input LQ\nImage")
    create_box(ax, (3.5, 6.8), 2.5, 1.2, "round,pad=0.1", "#10b981", "#047857", label="Encoder Stages\n(IrregularGazeBlocks)")
    create_box(ax, (6.8, 6.8), 2.0, 1.2, "round,pad=0.1", "#8b5cf6", "#6d28d9", label="Downsampling\n(Conv 2x2)")
    create_box(ax, (9.5, 6.8), 2.2, 1.2, "round,pad=0.1", "#f59e0b", "#b45309", label="Middle Block\n(IrregularGazeBlocks)")
    
    draw_arrow(ax, (2.7, 7.4), (3.5, 7.4))
    draw_arrow(ax, (6.0, 7.4), (6.8, 7.4))
    draw_arrow(ax, (8.8, 7.4), (9.5, 7.4))

    # Down arrow to decoder
    draw_arrow(ax, (10.6, 6.8), (10.6, 5.5))

    create_box(ax, (9.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#ec4899", "#be185d", label="Upsampling\n(PixelShuffle)")
    create_box(ax, (6.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#10b981", "#047857", label="Decoder Stages\n(IrregularGazeBlocks)")
    create_box(ax, (3.5, 4.3), 2.2, 1.2, "round,pad=0.1", "#3b82f6", "#1d4ed8", label="Output HQ\nRestored Image")

    draw_arrow(ax, (9.5, 4.9), (8.7, 4.9))
    draw_arrow(ax, (6.5, 4.9), (5.7, 4.9))

    # Skip connection
    draw_arrow(ax, (4.7, 6.8), (4.7, 5.7), color='#64748b', style='->', lw=1.5)
    draw_arrow(ax, (4.7, 5.7), (7.6, 5.7), color='#64748b', style='-', lw=1.5)
    draw_arrow(ax, (7.6, 5.7), (7.6, 5.5), color='#64748b', style='->', lw=1.5)
    ax.text(6.1, 5.85, "Skip Connection (+)", fontsize=10, color='#475569', weight='bold')

    # Zoom in block on IrregularGCELayer
    create_box(ax, (0.5, 0.3), 13.0, 3.6, "round,pad=0.2", "#fff1f2", "#f43f5e", lw=1.5, fontcolor='#0f172a')
    ax.text(1.0, 3.5, "Deep Dive: IrregularGCELayer (Adaptive Recursive k-d Tree Partitioning)", fontsize=14, weight='bold', color='#9f1239')

    # Irregular GCE steps
    create_box(ax, (1.0, 1.3), 2.2, 1.5, "round,pad=0.1", "#f43f5e", "#be123c", label="1x1 Feature Map\nExtract Complexity")
    create_box(ax, (3.7, 1.3), 2.8, 1.5, "round,pad=0.1", "#a855f7", "#7e22ce", label="Variance vs Threshold\nCheck Spatial Var")
    create_box(ax, (7.0, 1.3), 2.8, 1.5, "round,pad=0.1", "#06b6d4", "#0e7490", label="Recursive k-d Tree Split\n(Bounded by Kernel Size)")
    create_box(ax, (10.3, 1.3), 2.7, 1.5, "round,pad=0.1", "#10b981", "#047857", label="Independent DW/PW\n& Attention Modulate")

    draw_arrow(ax, (3.2, 2.05), (3.7, 2.05))
    draw_arrow(ax, (6.5, 2.05), (7.0, 2.05))
    draw_arrow(ax, (9.8, 2.05), (10.3, 2.05))

    ax.text(8.4, 0.7, "Non-overlapping 4x5, 3x3, 6x6 Patches", fontsize=10, color='#0e7490', weight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig("assets/architecture_diagrams/IrregularCGNet_arch.png", bbox_inches='tight')
    plt.close()

def plot_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Left: DynamicCGNet Uniform Grid
    ax1 = axes[0]
    ax1.set_title("DynamicCGNet: Uniform Grid Patching", fontsize=14, weight='bold', color='#1d4ed8')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Draw uniform grid
    for i in range(1, 10, 3):
        for j in range(1, 10, 3):
            rect = patches.Rectangle((i, j), 3, 3, linewidth=2, edgecolor='#1d4ed8', facecolor='#dbeafe')
            ax1.add_patch(rect)
            ax1.text(i + 1.5, j + 1.5, "S_x * S_x\nFixed", ha='center', va='center', color='#1e3a8a', fontsize=11, weight='bold')
    ax1.text(5, 0.2, "All patches share uniform size predicted by Q/K Attention", ha='center', fontsize=11, color='#334155')

    # Right: IrregularCGNet k-d Tree
    ax2 = axes[1]
    ax2.set_title("IrregularCGNet: Adaptive k-d Tree Patching", fontsize=14, weight='bold', color='#be123c')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Draw irregular grid like user diagram
    # Outer box (1, 1) to (10, 10) -> size 9x9
    # Left top: 4x5
    ax2.add_patch(patches.Rectangle((1, 5), 4, 5, linewidth=2, edgecolor='#be123c', facecolor='#ffe4e6'))
    ax2.text(3, 7.5, "Smooth Area\n4 x 5", ha='center', va='center', color='#881337', fontsize=11, weight='bold')

    # Right top: 5x5
    ax2.add_patch(patches.Rectangle((5, 5), 5, 5, linewidth=2, edgecolor='#be123c', facecolor='#fce7f3'))
    ax2.text(7.5, 7.5, "Smooth Area\n5 x 5", ha='center', va='center', color='#831843', fontsize=11, weight='bold')

    # Bottom left: 3x4
    ax2.add_patch(patches.Rectangle((1, 1), 3, 4, linewidth=2, edgecolor='#be123c', facecolor='#f3e8ff'))
    ax2.text(2.5, 3, "3 x 4", ha='center', va='center', color='#581c87', fontsize=11, weight='bold')

    # Bottom middle fine splits: edges
    ax2.add_patch(patches.Rectangle((4, 1), 3, 2, linewidth=2, edgecolor='#be123c', facecolor='#e0e7ff'))
    ax2.text(5.5, 2, "3 x 2", ha='center', va='center', color='#312e81', fontsize=10, weight='bold')

    ax2.add_patch(patches.Rectangle((4, 3), 3, 2, linewidth=2, edgecolor='#be123c', facecolor='#c7d2fe'))
    ax2.text(5.5, 4, "3 x 2", ha='center', va='center', color='#312e81', fontsize=10, weight='bold')

    # Bottom right strip
    ax2.add_patch(patches.Rectangle((7, 1), 3, 4, linewidth=2, edgecolor='#be123c', facecolor='#ffedd5'))
    ax2.text(8.5, 3, "Edge Strip\n3 x 4", ha='center', va='center', color='#7c2d12', fontsize=11, weight='bold')

    ax2.text(5, 0.2, "Patches split recursively based on variance until uniform or kernel_size", ha='center', fontsize=11, color='#334155')

    plt.tight_layout()
    plt.savefig("assets/architecture_diagrams/Patching_Comparison.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_dynamic_cgnet()
    plot_irregular_cgnet()
    plot_comparison()
    print("Successfully generated all diagrams in assets/architecture_diagrams/")
