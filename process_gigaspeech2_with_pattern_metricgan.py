#!/usr/bin/env python3
"""
Process GigaSpeech2 Samples with Pattern→MetricGAN+ Method
Uses the loaded GigaSpeech2 samples with the winning enhancement approach
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import soundfile as sf
import json
from pathlib import Path

# Import the final processor
sys.path.append(str(Path(__file__).parent))
from final_pattern_metricgan_processor_fixed import FinalPatternMetricGANProcessor

print("Processing GigaSpeech2 Samples with Pattern→MetricGAN+ Method")
print("=" * 80)

def process_gigaspeech2_samples():
    """Process the downloaded GigaSpeech2 samples"""
    
    # Check if samples exist
    samples_dir = Path("gigaspeech2_samples")
    if not samples_dir.exists():
        print("✗ GigaSpeech2 samples not found. Please run load_gigaspeech2_samples.py first.")
        return
    
    # Load metadata
    metadata_path = samples_dir / "metadata.json"
    if not metadata_path.exists():
        print("✗ Metadata file not found.")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} GigaSpeech2 samples to process")
    
    # Initialize processor
    processor = FinalPatternMetricGANProcessor()
    
    # Create output directory
    output_dir = Path("intelligent_enhancement_results")
    output_dir.mkdir(exist_ok=True)
    
    # Clear existing files
    print("\nClearing existing files...")
    for file in output_dir.glob("*.wav"):
        file.unlink()
    
    print("\nProcessing samples...")
    
    # Process each sample
    processing_times = []
    total_interruptions = 0
    
    for i, sample_info in enumerate(metadata[:50]):  # Process up to 50 samples
        sample_id = i + 1
        
        try:
            # Load original audio
            original_path = samples_dir / sample_info['filename']
            audio, sr = sf.read(original_path)
            
            print(f"\nSample {sample_id:02d}: {sample_info['filename']} ({sample_info['duration']:.2f}s)")
            if sample_info.get('text'):
                print(f"  Text: {sample_info['text'][:60]}{'...' if len(sample_info['text']) > 60 else ''}")
            
            # Save original to results directory
            new_original_path = output_dir / f"sample_{sample_id:02d}_original.wav"
            sf.write(new_original_path, audio, sr, subtype='PCM_16')
            
            # Process with Pattern→MetricGAN+
            start_time = time.time()
            enhanced_audio, num_interruptions = processor.process_audio(audio, sr, sample_id)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            total_interruptions += num_interruptions
            
            # Save enhanced result
            enhanced_path = output_dir / f"sample_{sample_id:02d}_final_pattern_then_metricgan.wav"
            sf.write(enhanced_path, enhanced_audio, sr, subtype='PCM_16')
            
            if sample_id % 10 == 0:
                print(f"\n  ✓ Completed {sample_id}/{len(metadata)} samples")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_id}: {e}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    
    avg_time = np.mean(processing_times) if processing_times else 0
    total_time = sum(processing_times)
    
    report_path = output_dir / "GIGASPEECH2_PATTERN_METRICGAN_RESULTS.md"
    
    with open(report_path, 'w') as f:
        f.write("# GigaSpeech2 Pattern→MetricGAN+ Processing Results\n\n")
        
        f.write("## Dataset Information\n\n")
        f.write("- **Source**: GigaSpeech2 Dataset\n")
        f.write("- **Total samples processed**: 50\n")
        f.write("- **Sample rate**: 16kHz\n")
        f.write("- **Audio format**: 16-bit PCM WAV\n\n")
        
        f.write("## Processing Results\n\n")
        f.write(f"- **Samples processed**: {len(processing_times)}\n")
        f.write(f"- **Total interruptions detected**: {total_interruptions}\n")
        f.write(f"- **Average interruptions per sample**: {total_interruptions/len(processing_times):.2f}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Average processing time**: {avg_time:.3f}s per sample\n")
        f.write(f"- **Total processing time**: {total_time:.2f}s\n")
        f.write(f"- **Processing rate**: {len(processing_times)/total_time:.1f} samples/sec\n\n")
        
        f.write("## Method Summary\n\n")
        f.write("The Pattern→MetricGAN+ method consists of:\n")
        f.write("1. **Ultra-conservative pattern detection** - Detects interruptions with >0.8 confidence\n")
        f.write("2. **Gentle suppression** - 50ms padding with 85% suppression\n")
        f.write("3. **MetricGAN+ enhancement** - Overall audio quality improvement\n\n")
        
        f.write("## Output Files\n\n")
        f.write("All processed samples are saved in `intelligent_enhancement_results/` with:\n")
        f.write("- `sample_XX_original.wav` - Original GigaSpeech2 audio\n")
        f.write("- `sample_XX_final_pattern_then_metricgan.wav` - Enhanced audio\n")
    
    print(f"✓ Report saved: {report_path}")
    
    print("\n" + "="*80)
    print("GIGASPEECH2 PROCESSING COMPLETE!")
    print("="*80)
    print(f"\n✓ {len(processing_times)} GigaSpeech2 samples processed")
    print(f"✓ {total_interruptions} interruptions detected and processed")
    print(f"✓ Average processing time: {avg_time:.3f}s per sample")
    print("\n🎧 Results available in intelligent_enhancement_results/")

if __name__ == "__main__":
    process_gigaspeech2_samples()