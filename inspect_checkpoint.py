import torch
import sys

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n--- Checkpoint Keys ---")
for key in checkpoint.keys():
    print(f"- {key}")

print("\n--- Details ---")
if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")
if 'loss' in checkpoint:
    print(f"Loss: {checkpoint['loss']}")
if 'lr' in checkpoint:
    print(f"Learning Rate: {checkpoint['lr']}")
if 'batch_size' in checkpoint:
    print(f"Batch Size: {checkpoint['batch_size']}")

print("\n--- Encoder State Dict Keys (first 5) ---")
if 'encoder' in checkpoint:
    keys = list(checkpoint['encoder'].keys())
    for k in keys[:5]:
        print(f"  {k}: {checkpoint['encoder'][k].shape}")
    print("  ...")
