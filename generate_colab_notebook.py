import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Train nano-JEPA on Google Colab (T4 GPU)
This notebook contains all the steps to clone the repository, set up the environment, configure the model correctly, and start training.
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
## 2. Update Configuration for Colab
We need to update the configuration to use relative paths instead of local absolute paths, and apply the fixed parameters (reg_coeff=0.3, epochs=100, ipe=None).
"""))

cells.append(nbf.v4.new_code_cell("""\
import yaml

config_path = 'configs/pretrain/vitt16.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update data paths to be relative inside the Colab environment
config['data']['datasets'] = ['./nano-train.csv']
config['logging']['folder'] = './logs'

# Update hyperparameters to fix representation collapse and warmup issues
config['loss']['reg_coeff'] = 0.3
config['optimization']['epochs'] = 100
config['optimization']['ipe'] = None  # None allows dynamic calculation based on dataset size

# Save back to yaml
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("Configuration updated successfully!")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## 3. Run Pretraining
Run the training script using the updated config.
"""))

cells.append(nbf.v4.new_code_cell("""\
!python -m app.train_nano_jepa --fname configs/pretrain/vitt16.yaml
"""))

nb['cells'] = cells
with open('colab_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at colab_training.ipynb")
