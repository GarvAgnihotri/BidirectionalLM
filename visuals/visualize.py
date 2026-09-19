import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from matplotlib.colors import LinearSegmentedColormap


plt.style.use("dark_background")

NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#B44FFF"
NEON_GREEN = "#00FF88"
NEON_ORANGE = "#FF6B35"
NEON_PINK = "#FF2D9B"
DARK_BG = "#0D0D1A"
CARD_BG = "#13132A"


def plot_training_loss(training_history_path, save_path="visuals/training_loss.png"):
    with open(training_history_path) as f:
        history = json.load(f)

    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]
    learning_rates = [entry["lr"] for entry in history]

    smoothed_losses = []
    window = 10
    for i in range(len(losses)):
        start = max(0, i - window)
        smoothed_losses.append(np.mean(losses[start:i+1]))

    figure, (top_axis, bottom_axis) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
    figure.suptitle("Training Progress", fontsize=20, color="white", fontweight="bold", y=0.98)

    top_axis.set_facecolor(CARD_BG)
    top_axis.plot(steps, losses, color=NEON_BLUE, alpha=0.3, linewidth=1, label="Raw Loss")
    top_axis.plot(steps, smoothed_losses, color=NEON_GREEN, linewidth=2.5, label="Smoothed Loss")
    top_axis.set_ylabel("Cross Entropy Loss", color="white", fontsize=12)
    top_axis.set_xlabel("Training Step", color="white", fontsize=12)
    top_axis.tick_params(colors="white")
    top_axis.spines[:].set_color("#333355")
    top_axis.legend(facecolor=CARD_BG, labelcolor="white", fontsize=11)
    top_axis.grid(alpha=0.15, color="white")

    bottom_axis.set_facecolor(CARD_BG)
    bottom_axis.plot(steps, learning_rates, color=NEON_PURPLE, linewidth=2)
    bottom_axis.set_ylabel("Learning Rate", color="white", fontsize=12)
    bottom_axis.set_xlabel("Training Step", color="white", fontsize=12)
    bottom_axis.tick_params(colors="white")
    bottom_axis.spines[:].set_color("#333355")
    bottom_axis.grid(alpha=0.15, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_heatmap(attention_weights, tokens, layer_number, head_number, save_path="visuals/attention_heatmap.png"):
    figure, axis = plt.subplots(figsize=(12, 10), facecolor=DARK_BG)
    axis.set_facecolor(CARD_BG)

    custom_colormap = LinearSegmentedColormap.from_list(
        "neon_heat",
        [DARK_BG, "#1A1A4E", NEON_PURPLE, NEON_BLUE, NEON_GREEN, "white"]
    )

    heatmap = axis.imshow(attention_weights, cmap=custom_colormap, aspect="auto", vmin=0, vmax=attention_weights.max())

    colorbar = plt.colorbar(heatmap, ax=axis, shrink=0.8)
    colorbar.ax.yaxis.set_tick_params(color="white")
    colorbar.outline.set_edgecolor("white")
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="white")
    colorbar.set_label("Attention Weight", color="white", fontsize=11)

    axis.set_xticks(range(len(tokens)))
    axis.set_yticks(range(len(tokens)))
    axis.set_xticklabels(tokens, rotation=45, ha="right", color="white", fontsize=10)
    axis.set_yticklabels(tokens, color="white", fontsize=10)

    axis.set_title(
        f"Attention Heatmap — Layer {layer_number}, Head {head_number}",
        color="white", fontsize=16, fontweight="bold", pad=15
    )
    axis.set_xlabel("Key Tokens (attending to)", color="white", fontsize=12)
    axis.set_ylabel("Query Tokens (from)", color="white", fontsize=12)

    for row in range(len(tokens)):
        for col in range(len(tokens)):
            value = attention_weights[row, col]
            text_color = "black" if value > attention_weights.max() * 0.6 else "white"
            axis.text(col, row, f"{value:.2f}", ha="center", va="center",
                     color=text_color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confidence_generation(generation_steps, save_path="visuals/confidence_generation.png"):
    number_of_steps = len(generation_steps)
    figure = plt.figure(figsize=(16, number_of_steps * 1.8 + 3), facecolor=DARK_BG)
    figure.suptitle("Bidirectional Confidence-Guided Generation", fontsize=18, color="white", fontweight="bold")

    all_positions = set()
    for step in generation_steps:
        for revealed in step["revealed_tokens"]:
            all_positions.add(revealed["position"])

    total_positions = max(all_positions) + 1 if all_positions else 10

    revealed_so_far = {}

    for step_index, step in enumerate(generation_steps):
        axis = figure.add_subplot(number_of_steps, 1, step_index + 1)
        axis.set_facecolor(CARD_BG)
        axis.set_xlim(-0.5, total_positions - 0.5)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xticks([])
        axis.spines[:].set_color("#333355")

        for revealed in step["revealed_tokens"]:
            revealed_so_far[revealed["position"]] = revealed

        for position in range(total_positions):
            box_x = position
            if position in revealed_so_far:
                token_info = revealed_so_far[position]
                confidence = token_info["confidence"]
                side = token_info.get("side", "left")

                if side == "left":
                    box_color = NEON_BLUE
                elif side == "right":
                    box_color = NEON_PURPLE
                else:
                    box_color = NEON_GREEN

                box = mpatches.FancyBboxPatch(
                    (box_x - 0.4, 0.15), 0.8, 0.7,
                    boxstyle="round,pad=0.05",
                    facecolor=box_color, edgecolor="white", linewidth=1.5, alpha=0.9
                )
                axis.add_patch(box)
                axis.text(box_x, 0.52, token_info["token"][:6],
                         ha="center", va="center", color="black", fontsize=9, fontweight="bold")
                axis.text(box_x, 0.22, f"{confidence:.2f}",
                         ha="center", va="center", color="black", fontsize=7)
            else:
                box = mpatches.FancyBboxPatch(
                    (box_x - 0.4, 0.15), 0.8, 0.7,
                    boxstyle="round,pad=0.05",
                    facecolor="#1A1A3E", edgecolor="#444466", linewidth=1, alpha=0.8
                )
                axis.add_patch(box)
                axis.text(box_x, 0.5, "?",
                         ha="center", va="center", color="#666688", fontsize=12)

        axis.set_ylabel(f"Step {step_index + 1}", color="white", fontsize=10, rotation=0, labelpad=40, va="center")

    left_patch = mpatches.Patch(color=NEON_BLUE, label="Left cursor")
    right_patch = mpatches.Patch(color=NEON_PURPLE, label="Right cursor")
    forced_patch = mpatches.Patch(color=NEON_GREEN, label="Forced reveal")
    figure.legend(handles=[left_patch, right_patch, forced_patch],
                 loc="lower center", ncol=3, facecolor=CARD_BG,
                 labelcolor="white", fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_token_probability_distribution(model, tokenizer, sentence, device, save_path="visuals/token_probabilities.png"):
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    token_words = [tokenizer.decode([t]) for t in tokens]
    token_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(token_tensor)

    probabilities = F.softmax(logits[0], dim=-1).cpu().numpy()
    top_k = 8

    figure, axes = plt.subplots(1, len(tokens), figsize=(len(tokens) * 3.5, 6), facecolor=DARK_BG)
    figure.suptitle(f'Token Probability Distribution\n"{sentence}"',
                   fontsize=14, color="white", fontweight="bold")

    if len(tokens) == 1:
        axes = [axes]

    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, NEON_PINK,
              "#FFD700", "#00FFFF", "#FF69B4"]

    for position_index, axis in enumerate(axes):
        axis.set_facecolor(CARD_BG)
        axis.spines[:].set_color("#333355")

        top_indices = np.argsort(probabilities[position_index])[-top_k:][::-1]
        top_probs = probabilities[position_index][top_indices]
        top_words = [tokenizer.decode([idx]).strip() for idx in top_indices]

        bars = axis.barh(range(top_k), top_probs, color=colors[:top_k], alpha=0.85)

        for bar, word, prob in zip(bars, top_words, top_probs):
            axis.text(prob + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{prob:.3f}", va="center", color="white", fontsize=8)
            axis.text(0.005, bar.get_y() + bar.get_height()/2,
                     word[:10], va="center", color="black", fontsize=8, fontweight="bold")

        axis.set_yticks([])
        axis.set_xlabel("Probability", color="white", fontsize=9)
        axis.tick_params(colors="white")
        axis.set_title(f'"{token_words[position_index].strip()}"',
                      color=NEON_BLUE, fontsize=11, fontweight="bold")
        axis.set_xlim(0, min(1.0, top_probs[0] * 1.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_architecture_diagram(save_path="visuals/architecture.png"):
    figure, axis = plt.subplots(figsize=(14, 10), facecolor=DARK_BG)
    axis.set_facecolor(DARK_BG)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 12)
    axis.axis("off")

    figure.suptitle("Bidirectional LM — Architecture Overview",
                   fontsize=18, color="white", fontweight="bold")

    def draw_box(x, y, width, height, label, color, fontsize=10):
        box = mpatches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85
        )
        axis.add_patch(box)
        axis.text(x, y, label, ha="center", va="center",
                 color="white", fontsize=fontsize, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, color="white"):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2))

    draw_box(5, 11, 8, 0.7, "Input Text  +  Reversed Text (Training Batches)", NEON_BLUE, fontsize=11)

    draw_box(5, 9.8, 6, 0.6, "Token Embedding  +  Positional Encoding", "#2A2A5E", fontsize=10)
    draw_arrow(5, 10.65, 5, 10.1)

    for layer_index in range(4):
        layer_y = 8.6 - layer_index * 1.1
        draw_box(5, layer_y, 6, 0.75, f"Transformer Block {layer_index + 1}  (Bidirectional Attention)", "#1E1E4E")
        if layer_index > 0:
            draw_arrow(5, layer_y + 1.1 - 0.375, 5, layer_y + 0.375)

    draw_box(2, 3.8, 2.5, 0.65, "Left Cursor\n(Forward)", NEON_BLUE, fontsize=9)
    draw_box(8, 3.8, 2.5, 0.65, "Right Cursor\n(Reversed)", NEON_PURPLE, fontsize=9)

    draw_arrow(5, 5.47, 2, 4.12)
    draw_arrow(5, 5.47, 8, 4.12)

    draw_box(5, 2.8, 5, 0.65, "Confidence Scorer  (0.97, 0.91, 0.88 ...)", NEON_GREEN, fontsize=9)
    draw_arrow(2, 3.47, 5, 3.12)
    draw_arrow(8, 3.47, 5, 3.12)

    draw_box(5, 1.8, 6, 0.65, "Reveal Highest Confidence Tokens First", NEON_ORANGE, fontsize=9)
    draw_arrow(5, 2.47, 5, 2.12)

    draw_box(5, 0.8, 7, 0.65,
             "My(0.99) → name(0.91) → is(0.88) → Garv(0.97)",
             "#2A1A0A", fontsize=10)
    draw_arrow(5, 1.47, 5, 1.12)

    dots_y = 5.0
    for dot_x in [4.5, 5.0, 5.5]:
        axis.plot(dot_x, dots_y, 'o', color="white", markersize=4, alpha=0.6)

    axis.text(8.5, 5.0, "×12 layers", color="#AAAACC", fontsize=9, ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def generate_demo_visuals():
    os.makedirs("visuals", exist_ok=True)

    print("Generating architecture diagram...")
    plot_model_architecture_diagram()

    print("Generating demo training loss curve...")
    demo_steps = list(range(100, 50100, 100))
    demo_losses = [4.5 * np.exp(-i/15000) + 0.8 + np.random.normal(0, 0.05) for i in range(len(demo_steps))]
    demo_lrs = [1e-4 * (i/500) if i < 500 else 1e-4 * (0.5 + 0.5 * np.cos(np.pi * (i-500)/49500)) for i in range(len(demo_steps))]
    demo_history = [{"step": s, "loss": l, "lr": lr} for s, l, lr in zip(demo_steps, demo_losses, demo_lrs)]
    with open("visuals/demo_history.json", "w") as f:
        json.dump(demo_history, f)
    plot_training_loss("visuals/demo_history.json", "visuals/training_loss.png")

    print("Generating demo attention heatmap...")
    demo_tokens = ["My", "name", "is", "Garv", "and", "I", "build", "AI"]
    demo_attention = np.random.dirichlet(np.ones(8) * 0.5, size=8)
    demo_attention = (demo_attention + demo_attention.T) / 2
    plot_attention_heatmap(demo_attention, demo_tokens, layer_number=6, head_number=3)

    print("Generating confidence generation visualization...")
    demo_generation_steps = [
        {"step": 1, "revealed_tokens": [
            {"position": 0, "token": "My", "confidence": 0.99, "side": "left"},
            {"position": 7, "token": "Garv", "confidence": 0.97, "side": "right"}
        ]},
        {"step": 2, "revealed_tokens": [
            {"position": 1, "token": "name", "confidence": 0.91, "side": "left"},
            {"position": 6, "token": "is", "confidence": 0.88, "side": "right"}
        ]},
        {"step": 3, "revealed_tokens": [
            {"position": 2, "token": "and", "confidence": 0.85, "side": "left"},
            {"position": 5, "token": "I", "confidence": 0.83, "side": "right"}
        ]},
        {"step": 4, "revealed_tokens": [
            {"position": 3, "token": "build", "confidence": 0.79, "side": "left"},
            {"position": 4, "token": "things", "confidence": 0.76, "side": "right"}
        ]},
    ]
    plot_confidence_generation(demo_generation_steps)

    print("\nAll visuals saved to visuals/ folder!")
    print("Files:")
    for filename in os.listdir("visuals"):
        if filename.endswith(".png"):
            print(f"  visuals/{filename}")


if __name__ == "__main__":
    generate_demo_visuals()
