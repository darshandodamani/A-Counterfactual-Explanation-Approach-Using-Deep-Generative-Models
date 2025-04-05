import matplotlib.pyplot as plt
import numpy as np

def draw_vae_fc_encoder_diagram():
    fig, ax = plt.subplots(figsize=(20, 8))

    # Simulated visual sizes for large layers
    layer_sizes = [70, 40, 25, 25]  # visually scaled down for 18432 → 1024 → 128 → 128
    labels = [
        r"Input Layer $\in \mathbb{R}^{18432}$",
        r"Hidden Layer $\in \mathbb{R}^{1024}$",
        r"$\mu \in \mathbb{R}^{128}$",
        r"logvar $\in \mathbb{R}^{128}$"
    ]

    v_spacing = 1.3
    h_spacing = 3.0
    radius = 0.15

    positions = []
    for i, size in enumerate(layer_sizes):
        x = i * h_spacing
        y_positions = np.linspace(-(size - 1) * v_spacing / 2, (size - 1) * v_spacing / 2, size)
        layer_pos = [(x, y) for y in y_positions]
        positions.append(layer_pos)

        # Draw nodes
        for pos in layer_pos:
            circle = plt.Circle(pos, radius, color='white', ec='black', zorder=3)
            ax.add_patch(circle)

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        for pos1 in positions[i]:
            for pos2 in positions[i + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', lw=0.3, alpha=0.4)

    # Add labels
    for i, label in enumerate(labels):
        x = i * h_spacing
        ax.text(x, -layer_sizes[i] * v_spacing / 2 - 1.0, label,
                fontsize=14, ha='center', va='center', style='italic')

    ax.axis('off')
    ax.set_xlim(-1, len(layer_sizes) * h_spacing)
    ax.set_ylim(-max(layer_sizes) * v_spacing / 1.5, max(layer_sizes) * v_spacing / 1.5)
    plt.title("VAE Encoder Fully Connected Layers", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("vae_encoder.svg", format='svg')

draw_vae_fc_encoder_diagram()
