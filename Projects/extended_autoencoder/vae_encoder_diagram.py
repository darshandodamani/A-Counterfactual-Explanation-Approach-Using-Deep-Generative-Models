import matplotlib.pyplot as plt

def draw_vae_encoder_blocks():
    fig, ax = plt.subplots(figsize=(16, 12))

    layers = [
        "Input\n[3×80×160]",
        "Conv2D(3→64)\nKernel=4, Stride=2",
        "Conv2D(64→128)\nKernel=3, Stride=2\n+ BatchNorm",
        "Conv2D(128→256)\nKernel=4, Stride=2",
        "Conv2D(256→512)\nKernel=3, Stride=2\n+ BatchNorm",
        "Flatten → [18432]",
        "Hidden Linear Layer\n(18432 → 1024)",
        "Separate Linear Layers:",
        "→ μ ∈ ℝ¹²⁸",
        "→ logvar ∈ ℝ¹²⁸",
        "Reparameterization:\nz = μ + σ·ε"
    ]

    y_positions = list(range(len(layers)))[::-1]

    for i, (layer, y) in enumerate(zip(layers, y_positions)):
        # Highlight latent components
        if "μ" in layer or "logvar" in layer or "z" in layer:
            color = 'lightyellow'
        elif "Separate Linear" in layer:
            color = 'lavender'
        else:
            color = 'lightblue'

        ax.text(0.5, y, layer,
                fontsize=11.5, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor=color))

        # Draw arrows between boxes
        if i < len(layers) - 1:
            ax.annotate('', xy=(0.5, y - 1), xytext=(0.5, y - 0.3),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(layers) + 1)
    ax.axis("off")
    ax.set_title("VAE Encoder Architecture (Detailed Latent Flow)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("vae_encoder_detailed.png", dpi=300)
    plt.show()

draw_vae_encoder_blocks()
