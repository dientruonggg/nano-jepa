import json

with open("visualize_features.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_source = """# Plot original frames side-by-side with their corresponding PCA feature maps
import matplotlib.pyplot as plt

# We have T_out temporal blocks. Each block corresponds to `tubelet_size` frames.
# We will take the first frame of each tubelet for the original video row.
fig, axes = plt.subplots(2, T_out, figsize=(4 * T_out, 8), dpi=150)

for i in range(T_out):
    # Original Frame (Row 0)
    frame_idx = i * tubelet_size
    ax_orig = axes[0, i]
    # Ensure frames are uint8 for imshow or normalize if they are floats
    img = frames[frame_idx]
    if img.max() <= 1.0:
        ax_orig.imshow(img)
    else:
        ax_orig.imshow(img.astype('uint8'))
    ax_orig.set_title(f"Orig Frame {frame_idx}")
    ax_orig.axis('off')
    
    # Feature Map (Row 1)
    ax_feat = axes[1, i]
    ax_feat.imshow(features_rgb[i], interpolation='bicubic')
    ax_feat.set_title(f"PCA Block {i+1}")
    ax_feat.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.show()"""

# Create a new notebook cell
new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "new_plot_cell",
   "metadata": {},
   "outputs": [],
   "source": [line + ("\n" if j < len(new_cell_source.split("\n"))-1 else "") for j, line in enumerate(new_cell_source.split("\n"))]
}

# Append the new cell to the notebook
nb["cells"].append(new_cell)

with open("visualize_features.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook appended successfully.")
