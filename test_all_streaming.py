#!/usr/bin/env python3
"""
Test all datasets in streaming mode with 1 sample each.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing all datasets in streaming mode...")

# Test each dataset individually to isolate any issues
datasets = ["ProcessedVoiceTH", "MozillaCommonVoice", "GigaSpeech2"]  # All datasets

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Testing {dataset}")
    print('='*60)
    
    sys.argv = [
        "main.py",
        "--fresh",
        dataset,
        "--streaming",
        "--sample",
        "--sample-size", "1",
        "--no-upload"
    ]
    
    try:
        from main import main
        exit_code = main()
        if exit_code == 0:
            print(f"✓ {dataset} streaming test passed")
        else:
            print(f"✗ {dataset} streaming test failed with code {exit_code}")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        break
    except Exception as e:
        print(f"✗ {dataset} streaming test error: {e}")

print("\nDone!")