# Custom Data Lists

Place your custom train/val/test split files here.

## Format

Each list file should contain one sequence name per line:
```
sequence_001
sequence_002
sequence_003
```

## Example Files

- `clr_train.txt` - Training sequences
- `clr_val.txt` - Validation sequences  
- `clr_test.txt` - Test sequences

## Usage

```bash
python source/ronin_resnet.py --mode train \
  --root_dir Data/clr_data \
  --train_list clr_lists/clr_train.txt \
  --val_list clr_lists/clr_val.txt
```
