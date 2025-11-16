# RoNIN Project Structure

This document describes the organized directory structure for RoNIN research and development.

## Directory Layout

```
ronin/
├── source/                      # Original RoNIN source code (DO NOT MODIFY)
├── lists/                       # Original RoNIN data splits
├── config/                      # Original RoNIN configurations
│
├── Data/                        # All datasets
│   ├── ronin/                   # Original RoNIN dataset (152 sequences)
│   │   ├── a001_1/
│   │   ├── a001_2/
│   │   └── ...
│   └── clr_data/                 # Your custom datasets
│       └── README.md
│
├── models/                      # All model checkpoints
│   ├── pretrained/              # Original pretrained RoNIN models
│   │   ├── ronin_resnet/
│   │   ├── ronin_lstm/
│   │   ├── ronin_tcn/
│   │   └── ronin_body_heading/
│   ├── transfer_learning/       # Transfer learning experiments
│   └── from_scratch/            # Models trained from scratch
│
├── clr_lists/                    # Your custom data split lists
│   └── README.md
│
├── clr_source/                   # Your custom code
│   └── README.md
│
├── experiments/                 # Experiment tracking
│   └── README.md
│
└── notebooks/                   # Jupyter notebooks for analysis
```

## Quick Start

### 1. Train with Original RoNIN Data
```bash
python source/ronin_resnet.py --mode train \
  --root_dir Data/ronin \
  --train_list lists/list_train.txt \
  --out_dir models/from_scratch/exp_001_baseline
```

### 2. Test with Pretrained Model
```bash
python source/ronin_resnet.py --mode test \
  --root_dir Data/ronin \
  --test_list lists/list_test_seen.txt \
  --model_path models/pretrained/ronin_resnet/checkpoint_gsn_latest.pt \
  --out_dir experiments/exp_001_baseline
```

### 3. Transfer Learning with Custom Data
```bash
# First, prepare your data in Data/clr_data/
# Create train/val/test lists in clr_lists/

python source/ronin_lstm_tcn.py train \
  --data_dir Data/clr_data \
  --train_list clr_lists/clr_train.txt \
  --val_list clr_lists/clr_val.txt \
  --continue_from models/pretrained/ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt \
  --out_dir models/transfer_learning/exp_001_finetune_lstm
```

## Key Paths

### Data
- **RoNIN dataset**: `Data/ronin/`
- **Custom data**: `Data/clr_data/`

### Models
- **Pretrained ResNet**: `models/pretrained/ronin_resnet/checkpoint_gsn_latest.pt`
- **Pretrained LSTM**: `models/pretrained/ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt`
- **Pretrained TCN**: `models/pretrained/ronin_tcn/checkpoints/checkpoint_tcn.pt`
- **Pretrained Heading**: `models/pretrained/ronin_body_heading/checkpoints/checkpoint_body_heading.pt`

### Lists
- **Train**: `lists/list_train.txt` (73 sequences)
- **Val**: `lists/list_val.txt`
- **Test (seen)**: `lists/list_test_seen.txt` (32 sequences)
- **Test (unseen)**: `lists/list_test_unseen.txt` (32 sequences)

## Best Practices

1. **Never modify original source code** - Create custom versions in `clr_source/`
2. **Track experiments** - Use descriptive folder names with experiment numbers
3. **Document everything** - Add README files and comments
4. **Version control** - Commit code changes, not data/models
5. **Reproducibility** - Save configs with each experiment

## Git Ignore Recommendations

Add to `.gitignore`:
```
Data/
models/
experiments/*/plots/
experiments/*/checkpoints/
*.hdf5
*.pt
*.pth
```

Keep in git:
```
source/
clr_source/
lists/
clr_lists/
config/
experiments/*/config.yaml
experiments/*/results.json
```
