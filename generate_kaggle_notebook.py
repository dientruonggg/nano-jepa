import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Train nano-JEPA on Kaggle (P100 / T4x2 GPU) with UCF-101
This notebook downloads the UCF-101 dataset, configures nano-JEPA for Kaggle GPU environments (16GB VRAM, 4 CPU cores), and trains the model. 

**IMPORTANT**: Make sure your Kaggle Session options has **Persistence** set to **Files only**. This notebook is designed to survive disconnects and will automatically resume training from where it left off!
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 1. Setup Working Directory and Clone Repository
Kaggle's writable directory is `/kaggle/working`. If the repo already exists (from a previous session), we just pull the latest changes.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
%cd /kaggle/working
if not os.path.exists('nano-jepa'):
    !git clone https://github.com/dientribution/nano-jepa.git
else:
    print("nano-jepa already exists. Pulling latest changes...")
    %cd nano-jepa
    !git pull origin main
    %cd ..

%cd nano-jepa
!pip install -e .
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 2. Download and Extract UCF-101 Dataset
This will download the 6.5GB UCF-101 dataset from the official source and extract it. Skips if already downloaded.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
if not os.path.exists('/tmp/UCF-101'):
    print("Downloading UCF-101 to /tmp (avoids Kaggle output limits)...")
    !apt-get update > /dev/null
    !apt-get install -y unrar > /dev/null
    %cd /tmp
    !wget -c --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    print("Extracting UCF-101...")
    !unrar x -idq UCF101.rar
    !rm UCF101.rar
    %cd /kaggle/working/nano-jepa
else:
    print("UCF-101 already exists in /tmp. Skipping download.")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 3. Prepare Dataset CSV
Create a CSV file mapping video paths for the DataLoader.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
import glob

# Find all .avi videos in the /tmp extracted folder
videos = glob.glob('/tmp/UCF-101/**/*.avi', recursive=True)

if len(videos) < 13000:
    print(f"WARNING: Expected ~13320 videos, but found {len(videos)}. The download or extraction might have failed!")

# Write absolute paths to ucf101.csv
with open('ucf101.csv', 'w') as f:
    for v in videos:
        f.write(f"{os.path.abspath(v)} 0\\n")
        
print(f"Generated ucf101.csv with {len(videos)} videos.")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 4. Update Configuration for Kaggle GPU
We modify `vitt16.yaml` to utilize the Kaggle GPU effectively:
- `batch_size: 8` (prevents CUDA Out of Memory on 16GB GPUs like P100 or T4)
- `num_workers: 4` (Kaggle standard instance has 4 vCPUs)
- `epochs: 300`, `warmup: 30` (proper training schedule)
"""))

cells.append(nbf.v4.new_code_cell("""\
import yaml

config_path = 'configs/pretrain/vitt16.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update paths
config['data']['datasets'] = ['./ucf101.csv']
config['logging']['folder'] = './logs'

# Adjust for Kaggle GPU Environment
config['data']['batch_size'] = 32 # T4 16GB can handle much more than 8 for vit_tiny
config['data']['num_workers'] = 4 # Kaggle has 4 CPU cores
config['meta']['dtype'] = 'float16' # T4 does NOT support bfloat16 hardware acceleration!

# Hyperparameter fixes for UCF-101 self-supervised pre-training
config['optimization']['epochs'] = 300
config['optimization']['warmup'] = 30
config['optimization']['ipe'] = None  # Dynamic steps per epoch

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("Configuration updated for Kaggle GPU!")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 5. Auto-Resume Checkpoint Setup
Automatically scans the `logs/` directory for the latest checkpoint and updates `vitt16.yaml` to resume from it.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
import glob
import yaml

logs_dir = './logs'
config_path = 'configs/pretrain/vitt16.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Find the latest checkpoint
checkpoints = glob.glob(f'{logs_dir}/*-latest.pth.tar')
if checkpoints:
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"Found existing checkpoint: {latest_ckpt}")
    config['meta']['load_checkpoint'] = True
    config['meta']['read_checkpoint'] = latest_ckpt
else:
    print("No existing checkpoints found. Starting from scratch.")
    config['meta']['load_checkpoint'] = False
    config['meta']['read_checkpoint'] = None

with open(config_path, 'w') as f:
    yaml.dump(config, f)
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 6. Start Pretraining
Run the training script!
"""))

cells.append(nbf.v4.new_code_cell("""\
!mkdir -p logs
!torchrun --nproc_per_node=2 -m app.train_nano_jepa --fname configs/pretrain/vitt16.yaml
"""))

nb['cells'] = cells

with open('kaggle_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at kaggle_training.ipynb")
