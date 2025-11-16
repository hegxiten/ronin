# Custom Source Code

Place your custom code and modifications here.

## Suggested Structure

- `custom_dataloader.py` - Custom data loading logic
- `modified_models.py` - Modified network architectures
- `train_transfer_learning.py` - Transfer learning scripts
- `train_custom.py` - Custom training loops
- `utils.py` - Helper functions

## Best Practices

1. **Import from original source**: 
   ```python
   import sys
   sys.path.append('../source')
   from data_glob_speed import GlobSpeedSequence
   ```

2. **Keep original code intact**: Don't modify files in `source/`

3. **Document changes**: Add comments explaining modifications

4. **Version experiments**: Use descriptive names for output directories
