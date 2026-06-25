import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Train nano-JEPA on Google Colab (T4 GPU) with UCF-101
This notebook downloads the UCF-101 dataset, configures nano-JEPA for a T4 GPU (16GB VRAM, 2 CPU cores), and trains the model.
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 1. Clone Repository and Install Dependencies
"""))

cells.append(nbf.v4.new_code_cell("""\
!git clone https://github.com/dientribution/nano-jepa.git
%cd nano-jepa
!pip install -e .
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 2. Download and Extract UCF-101 Dataset
This will download the 6.5GB UCF-101 dataset and extract it.
"""))

cells.append(nbf.v4.new_code_cell("""\
!apt-get install unrar
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
!unrar x UCF101.rar > /dev/null
!rm UCF101.rar
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 3. Prepare Dataset CSV
Create a CSV file mapping video paths for the DataLoader.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
import glob

# Find all .avi videos in the UCF-101 folder
videos = glob.glob('UCF-101/**/*.avi', recursive=True)

# Write absolute paths to ucf101.csv
with open('ucf101.csv', 'w') as f:
    for v in videos:
        f.write(f"{os.path.abspath(v)} 0\\n")
        
print(f"Generated ucf101.csv with {len(videos)} videos.")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 4. Update Configuration for T4 GPU
We modify `vitt16.yaml` to utilize the T4 effectively:
- `batch_size: 32` (fits comfortably in 16GB VRAM for vit_tiny 16 frames)
- `num_workers: 2` (Colab standard instance has 2 vCPUs)
- `reg_coeff: 0.3` (prevent representation collapse)
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

# Adjust for T4 GPU and Colab Environment
config['data']['batch_size'] = 32
config['data']['num_workers'] = 2

# Hyperparameter fixes for UCF-101 self-supervised pre-training
config['loss']['reg_coeff'] = 0.3
config['optimization']['epochs'] = 300
config['optimization']['warmup'] = 30
config['optimization']['ipe'] = None  # Dynamic steps per epoch

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("Configuration updated for T4 GPU!")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 5. Start Pretraining
Run the training script!
"""))

cells.append(nbf.v4.new_code_cell("""\
!python -m app.train_nano_jepa --fname configs/pretrain/vitt16.yaml
"""))

nb['cells'] = cells
with open('colab_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at colab_training.ipynb")
