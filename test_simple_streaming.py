#!/usr/bin/env python3
"""
Simple test to verify streaming mode works.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if we can import everything
try:
    from main import main, process_streaming_mode
    from utils.streaming import StreamingUploader, StreamingBatchProcessor
    from processors.gigaspeech2 import GigaSpeech2Processor
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test streaming with ProcessedVoiceTH (usually the most accessible)
print("\nTesting streaming mode with ProcessedVoiceTH (1 sample)...")
sys.argv = [
    "main.py",
    "--fresh",
    "ProcessedVoiceTH",
    "--streaming",
    "--sample",
    "--sample-size", "1",
    "--no-upload",
    "--verbose"
]

try:
    exit_code = main()
    if exit_code == 0:
        print("✓ Streaming test completed successfully!")
    else:
        print(f"✗ Streaming test failed with exit code: {exit_code}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()