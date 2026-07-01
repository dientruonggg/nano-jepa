import os
import math
import csv
import shutil

class MockOptimizer:
    def __init__(self):
        self.param_groups = [{'lr': 0}]

class WarmupCosineSchedule(object):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr

# Configuration from params yaml
start_lr = 0.0002
ref_lr = 0.000625
final_lr = 1.0e-06
num_epochs = 300
ipe = 416
ipe_scale = 1.25
warmup = 30

warmup_steps = int(warmup * ipe)
T_max = int(ipe_scale * num_epochs * ipe)

optimizer = MockOptimizer()
scheduler = WarmupCosineSchedule(
    optimizer=optimizer,
    warmup_steps=warmup_steps,
    start_lr=start_lr,
    ref_lr=ref_lr,
    T_max=T_max,
    final_lr=final_lr
)

# The LR depends on total steps elapsed.
# If the log starts at epoch 80, we need to fast-forward the scheduler to the correct global step.
# For epoch 80 (0-indexed if it started at 0), step_count = 80 * 416 = 33280 steps have been completed before this log starts.
# Wait, the log file has rows like '80,0'. Is epoch 1-indexed in the log?
# Let's read the first data row to determine the global step.
# Wait! In train_nano_jepa.py, the scheduler is stepped once per iteration:
# _new_lr = scheduler.step() inside train_step.
# But when resuming from checkpoint, does the scheduler state resume properly?
# Yes, because scheduler relies on `self._step`. But wait, in train_nano_jepa.py, `scheduler` is created from scratch using `init_opt`!
# The `init_opt` does not take `last_epoch` or restore `scheduler` state.
# Is that true?
# Wait! In `train_nano_jepa.py`, `optimizer, scaler, scheduler, wd_scheduler = init_opt(...)`.
# Then `load_checkpoint` restores `opt.load_state_dict(checkpoint['opt'])`, but it DOES NOT restore `scheduler` state!
# Wait! `WarmupCosineSchedule` doesn't implement `load_state_dict`. So its `_step` is ALWAYS 0 when `train_nano_jepa.py` starts!
# Wait, if `load_ckpt` is true, does it start the scheduler from 0?
# Let's check `train_nano_jepa.py`.
# Oh! In `train_nano_jepa.py`:
# pbar = tqdm(range(ipe), desc=f'Epoch {epoch + 1}/{num_epochs}', ...)
# But before the epoch loop, there's NO code to fast-forward the scheduler!
# If it resumed from epoch 80, and scheduler _step is 0, then the LR started from `start_lr` (0.0002) again!!
# Let's check if the scheduler step is fast-forwarded. No, there's no fast-forward!
# That means the LR for epoch 80 iteration 0 was literally the first step of warmup!
# Let's write the script to simulate exactly what the script did: creating the scheduler fresh, and calling `.step()` for every logged iteration!

csv_path = "logs/2026-06-29_13-26-29-log_file.csv"
temp_path = "logs/2026-06-29_13-26-29-log_file_temp.csv"

with open(csv_path, 'r') as infile, open(temp_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header_written = False
    
    for row in reader:
        if not row:
            continue
        
        # It's a header row
        if row[0] == 'epoch':
            if not header_written:
                new_row = row[:5] + ['lr'] + row[5:]
                writer.writerow(new_row)
                header_written = True
            continue
        
        # It's a data row
        # Since the scheduler _step was initialized to 0 and steps every iteration logged,
        # we just call step() here!
        lr = scheduler.step()
        
        new_row = row[:5] + [f"{lr:.2e}"] + row[5:]
        writer.writerow(new_row)

# Replace original with the recovered file
shutil.move(temp_path, csv_path)

print("Recovered LR and saved to CSV.")
