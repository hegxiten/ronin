#!/usr/bin/env python3
"""
CUDA Test and Proof-of-Concept for RoNIN
Tests hardware setup and runs a quick training test
"""

import sys
import os

print("=" * 80)
print("RoNIN CUDA Hardware Test & Proof-of-Concept")
print("=" * 80)

# Test 1: Check Python packages
print("\n[1/5] Checking Python packages...")
try:
    import numpy as np
    print(f"  ✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy not found: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch not found: {e}")
    print("\n  Install with: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

try:
    import h5py
    print(f"  ✓ h5py: {h5py.__version__}")
except ImportError as e:
    print(f"  ✗ h5py not found: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"  ✓ CUDA is available")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ Device count: {torch.cuda.device_count()}")
    print(f"  ✓ Device name: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print(f"  ✗ CUDA is NOT available - will use CPU")
    print(f"  Note: Training will be much slower on CPU")

# Test 3: Test CUDA operations
print("\n[3/5] Testing CUDA operations...")
try:
    device = torch.device('cuda:0' if cuda_available else 'cpu')
    print(f"  Using device: {device}")
    
    # Create test tensors
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # Test matrix multiplication
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize() if cuda_available else None
    elapsed = time.time() - start
    
    print(f"  ✓ Matrix multiplication (1000x1000): {elapsed*1000:.2f} ms")
    print(f"  ✓ Result shape: {z.shape}")
except Exception as e:
    print(f"  ✗ CUDA operation failed: {e}")
    sys.exit(1)

# Test 4: Check RoNIN data
print("\n[4/5] Checking RoNIN dataset...")
data_dir = "datasets/ronin"
if not os.path.exists(data_dir):
    print(f"  ✗ Dataset not found at {data_dir}")
    sys.exit(1)

sequences = [d for d in os.listdir(data_dir) if d.startswith('a') and os.path.isdir(os.path.join(data_dir, d))]
print(f"  ✓ Found {len(sequences)} sequences")

# Check a sample sequence
sample_seq = sequences[0] if sequences else None
if sample_seq:
    sample_path = os.path.join(data_dir, sample_seq)
    data_file = os.path.join(sample_path, 'data.hdf5')
    info_file = os.path.join(sample_path, 'info.json')
    
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / 1024**2
        print(f"  ✓ Sample sequence: {sample_seq}")
        print(f"  ✓ data.hdf5 size: {file_size:.2f} MB")
        
        # Try to read the file
        try:
            with h5py.File(data_file, 'r') as f:
                if 'synced' in f:
                    print(f"  ✓ HDF5 structure valid")
                    if 'synced/gyro' in f:
                        gyro_shape = f['synced/gyro'].shape
                        print(f"  ✓ Gyro data shape: {gyro_shape}")
        except Exception as e:
            print(f"  ✗ Error reading HDF5: {e}")
    
    if os.path.exists(info_file):
        print(f"  ✓ info.json exists")

# Test 5: Quick model test
print("\n[5/5] Testing model initialization...")
try:
    # Add source to path
    sys.path.insert(0, 'source')
    
    # Try to import model
    from model_resnet1d import ResNet1D
    
    # Create a small model
    model = ResNet1D(in_dim=6, out_dim=2, channel_list=[16, 32], kernel_size=3)
    model = model.to(device)
    
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Model on device: {next(model.parameters()).device}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 200
    test_input = torch.randn(batch_size, 6, seq_len).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Input shape: {test_input.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Device: {device}")
print(f"CUDA Available: {cuda_available}")
print(f"Dataset: {len(sequences)} sequences in {data_dir}")
print(f"Ready for training: {'YES ✓' if cuda_available and len(sequences) > 0 else 'PARTIAL (CPU only)'}")
print("\nNext steps:")
print("  1. Install missing packages: pip3 install -r requirements.txt")
print("  2. Run quick test: python source/ronin_resnet.py --mode test --test_list lists/list_test_seen.txt --root_dir datasets/ronin --model_path models/pretrained/ronin_resnet/checkpoint_gsn_latest.pt --out_dir experiments/poc_test")
print("  3. Run training: python source/ronin_resnet.py --mode train --train_list lists/list_train.txt --root_dir datasets/ronin --out_dir models/from_scratch/poc_train --epochs 5")
print("=" * 80)
