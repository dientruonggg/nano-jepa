import argparse
import yaml
import torch
from app.vjepa.utils import init_video_model
import os

def run_inference(checkpoint_path, config_path):
    print(f"[*] Reading config from: {config_path}")
    with open(config_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    # Extract model parameters from config
    cfgs_model = params['model']
    model_name = cfgs_model['model_name']
    pred_depth = cfgs_model['pred_depth']
    pred_embed_dim = cfgs_model['pred_embed_dim']
    uniform_power = cfgs_model['uniform_power']
    use_mask_tokens = cfgs_model['use_mask_tokens']
    zero_init_mask_tokens = cfgs_model['zero_init_mask_tokens']
    
    cfgs_data = params['data']
    num_frames = cfgs_data.get('num_frames', 16)
    tubelet_size = cfgs_data.get('tubelet_size', 2)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size', 16)
    
    cfgs_mask = params.get('mask', [])
    use_sdpa = params['meta'].get('use_sdpa', False)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing model on device: {device}")
    
    encoder, _ = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    
    # Load the checkpoint
    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # The saved state contains target_encoder, encoder, predictor
    # 'target_encoder' is usually the EMA teacher which produces stable features.
    if 'target_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['target_encoder'])
        print("[*] Successfully loaded 'target_encoder' weights into the model.")
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        print("[*] Successfully loaded 'encoder' weights into the model.")
    else:
        print("[!] Warning: Could not find 'target_encoder' or 'encoder' in checkpoint keys.")
        print(f"Available keys: {checkpoint.keys()}")

    encoder.eval()

    # Generate dummy data: (batch_size, channels, frames, height, width)
    batch_size = 1
    channels = 3
    print(f"\n[*] Creating dummy video input of shape: ({batch_size}, {channels}, {num_frames}, {crop_size}, {crop_size})")
    dummy_input = torch.randn(batch_size, channels, num_frames, crop_size, crop_size).to(device)

    # Run inference
    print("[*] Running inference forward pass...")
    with torch.no_grad():
        # V-JEPA encoder expects input without masks for inference 
        # (or passing empty mask list if signature requires it)
        try:
            features = encoder(dummy_input)
            print(f"[*] Inference successful! Output features shape: {features.shape}")
        except Exception as e:
            print(f"[!] Inference failed. Trying with empty mask list: {e}")
            features = encoder(dummy_input, masks=[])
            print(f"[*] Inference successful! Output features shape: {features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth.tar file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config .yaml file')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        exit(1)
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        exit(1)
        
    run_inference(args.checkpoint, args.config)
