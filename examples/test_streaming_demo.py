#!/usr/bin/env python3
"""
Demo script to test streaming mode with small samples from each dataset.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_streaming_sample():
    """Test streaming mode with small samples."""
    print("=" * 80)
    print("Testing Streaming Mode with Small Samples")
    print("=" * 80)
    
    # Test with each dataset individually
    datasets = ["GigaSpeech2", "MozillaCV", "ProcessedVoiceTH"]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {dataset} in streaming mode (5 samples)")
        print(f"{'='*60}")
        
        # Simulate command line arguments
        sys.argv = [
            "main.py",
            "--fresh",
            dataset,
            "--streaming",
            "--sample",
            "--sample-size", "5",
            "--no-upload",  # Don't upload for testing
            "--verbose"
        ]
        
        try:
            exit_code = main()
            if exit_code == 0:
                print(f"✓ {dataset} streaming test completed successfully")
            else:
                print(f"✗ {dataset} streaming test failed with exit code: {exit_code}")
        except Exception as e:
            print(f"✗ {dataset} streaming test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test all datasets together
    print(f"\n{'='*60}")
    print("Testing ALL datasets in streaming mode (5 samples each)")
    print(f"{'='*60}")
    
    sys.argv = [
        "main.py",
        "--fresh",
        "--all",
        "--streaming",
        "--sample",
        "--sample-size", "5",
        "--no-upload",
        "--verbose"
    ]
    
    try:
        exit_code = main()
        if exit_code == 0:
            print("✓ All datasets streaming test completed successfully")
        else:
            print(f"✗ All datasets streaming test failed with exit code: {exit_code}")
    except Exception as e:
        print(f"✗ All datasets streaming test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_streaming_resume():
    """Test streaming mode resume capability."""
    print(f"\n{'='*80}")
    print("Testing Streaming Mode Resume Capability")
    print(f"{'='*80}")
    
    # First run - process some samples
    print("\n1. Starting initial streaming run...")
    sys.argv = [
        "main.py",
        "--fresh",
        "ProcessedVoiceTH",
        "--streaming",
        "--sample",
        "--sample-size", "10",
        "--no-upload",
        "--streaming-batch-size", "5"  # Force checkpoint after 5 samples
    ]
    
    try:
        # This should process 10 samples and create checkpoints
        exit_code = main()
        print(f"Initial run completed with exit code: {exit_code}")
    except KeyboardInterrupt:
        print("Simulating interruption...")
    except Exception as e:
        print(f"Error during initial run: {str(e)}")
    
    # Second run - resume from checkpoint
    print("\n2. Resuming from checkpoint...")
    sys.argv = [
        "main.py",
        "--fresh",
        "ProcessedVoiceTH",
        "--streaming",
        "--sample",
        "--sample-size", "10",
        "--no-upload",
        "--resume"
    ]
    
    try:
        exit_code = main()
        if exit_code == 0:
            print("✓ Resume test completed successfully")
        else:
            print(f"✗ Resume test failed with exit code: {exit_code}")
    except Exception as e:
        print(f"✗ Resume test failed with error: {str(e)}")

def test_cached_vs_streaming():
    """Compare cached mode vs streaming mode."""
    print(f"\n{'='*80}")
    print("Comparing Cached Mode vs Streaming Mode")
    print(f"{'='*80}")
    
    dataset = "ProcessedVoiceTH"
    sample_size = 5
    
    # Test cached mode
    print(f"\n1. Testing {dataset} in CACHED mode ({sample_size} samples)...")
    sys.argv = [
        "main.py",
        "--fresh",
        dataset,
        "--sample",
        "--sample-size", str(sample_size),
        "--no-upload"
    ]
    
    try:
        exit_code = main()
        print(f"Cached mode completed with exit code: {exit_code}")
    except Exception as e:
        print(f"Cached mode error: {str(e)}")
    
    # Test streaming mode
    print(f"\n2. Testing {dataset} in STREAMING mode ({sample_size} samples)...")
    sys.argv = [
        "main.py",
        "--fresh",
        dataset,
        "--streaming",
        "--sample",
        "--sample-size", str(sample_size),
        "--no-upload"
    ]
    
    try:
        exit_code = main()
        print(f"Streaming mode completed with exit code: {exit_code}")
    except Exception as e:
        print(f"Streaming mode error: {str(e)}")

if __name__ == "__main__":
    print("Thai Audio Dataset Collection - Streaming Mode Test")
    print("=" * 80)
    
    # Check if HF token exists
    if not os.path.exists("hf_token.txt"):
        print("WARNING: hf_token.txt not found. Some tests may fail.")
        print("Create hf_token.txt with your Hugging Face token to enable all features.")
        print()
    
    # Run tests
    try:
        # Test 1: Basic streaming with samples
        test_streaming_sample()
        
        # Test 2: Resume capability
        test_streaming_resume()
        
        # Test 3: Compare modes
        test_cached_vs_streaming()
        
        print(f"\n{'='*80}")
        print("All tests completed!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()