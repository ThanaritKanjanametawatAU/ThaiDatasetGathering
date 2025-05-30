#!/usr/bin/env python3
"""
Quick timing test for dataset processing
"""
import time
import subprocess
import sys

def run_test(samples, features):
    """Run a test with given samples and features."""
    start_time = time.time()
    
    cmd = [
        "python", "main.py",
        "--fresh",
        "GigaSpeech2",
        "--sample",
        "--sample-size", str(samples),
        "--hf-repo", f"Thanarit/Thai-Voice-Timing-Test-{samples}",
        "--streaming"
    ]
    
    # Add features
    if "speaker_id" in features:
        cmd.append("--enable-speaker-id")
    if "stt" in features:
        cmd.append("--enable-stt")
    if "enhancement" in features:
        cmd.extend(["--enable-audio-enhancement", "--enhancement-level", "moderate"])
    
    print(f"\nRunning test with {samples} samples, features: {features}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Success in {elapsed:.1f} seconds")
            print(f"  Rate: {samples/elapsed:.1f} samples/second")
            print(f"  Estimated for 100k samples: {100000*elapsed/samples/3600:.1f} hours")
        else:
            print(f"✗ Failed after {elapsed:.1f} seconds")
            error_msg = result.stderr.split('\n')[-2] if result.stderr else 'Unknown'
            print(f"  Error: {error_msg}")
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ Timeout after {elapsed:.1f} seconds")
        
    return elapsed

# Test configurations
tests = [
    (100, ["speaker_id"]),  # Just speaker ID
    (100, ["speaker_id", "enhancement"]),  # Speaker ID + enhancement
    (100, ["speaker_id", "stt", "enhancement"]),  # All features
]

print("Starting timing tests...")
print("=" * 60)

results = []
for samples, features in tests:
    elapsed = run_test(samples, features)
    results.append((samples, features, elapsed))
    time.sleep(5)  # Brief pause between tests

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

for samples, features, elapsed in results:
    rate = samples/elapsed if elapsed > 0 else 0
    hours_100k = 100000*elapsed/samples/3600 if samples > 0 and elapsed > 0 else 0
    print(f"{samples} samples with {', '.join(features)}:")
    print(f"  Time: {elapsed:.1f}s, Rate: {rate:.1f} samples/s")
    print(f"  Estimated for 100k samples: {hours_100k:.1f} hours")