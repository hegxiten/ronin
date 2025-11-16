# Experiments

Track your experiments here. Each experiment should have its own folder.

## Suggested Structure

```
experiments/
├── exp_001_baseline_ronin/
│   ├── config.yaml
│   ├── results.json
│   ├── training_log.txt
│   └── plots/
├── exp_002_transfer_learning/
└── exp_003_custom_architecture/
```

## What to Track

- **config.yaml**: Hyperparameters, data paths, model settings
- **results.json**: Final metrics (ATE, RTE, loss)
- **training_log.txt**: Epoch-by-epoch training progress
- **plots/**: Visualization of results
- **notes.md**: Observations, ideas, next steps

## Naming Convention

Use descriptive names with experiment number:
- `exp_001_baseline_resnet_lr0.001`
- `exp_002_finetune_lstm_frozen5layers`
- `exp_003_custom_data_tcn`
