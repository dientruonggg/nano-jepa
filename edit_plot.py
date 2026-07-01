import json

with open("visualize_features.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_str = "".join(cell["source"])
        
        # Replace `img = frames[frame_idx]` with `img = frames_resized[frame_idx]`
        if "img = frames[frame_idx]" in source_str:
            source_str = source_str.replace("img = frames[frame_idx]", "img = frames_resized[frame_idx] # Dùng ảnh đã resize vuông để khớp với Feature Map")
            cell["source"] = [line + ("\n" if i < len(source_str.split("\n"))-1 else "") for i, line in enumerate(source_str.split("\n"))]

with open("visualize_features.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook plot updated to use frames_resized.")
