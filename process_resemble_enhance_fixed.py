#!/usr/bin/env python3
"""
Fixed version: Process with Resemble-Enhance using correct API
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import soundfile as sf
import librosa
from pathlib import Path

print("Resemble-Enhance Comparison Pipeline (Fixed)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


class ResembleEnhanceProcessor:
    """Process audio with Resemble-Enhance"""
    
    def __init__(self):
        self.device = device
        self.enhancer = None
        self.load_model()
        
    def load_model(self):
        """Load Resemble-Enhance model"""
        print("Loading Resemble-Enhance model...")
        try:
            from resemble_enhance.enhancer.inference import enhance
            self.enhance_fn = enhance
            print("✓ Resemble-Enhance loaded successfully")
            self.available = True
        except ImportError:
            print("✗ Resemble-Enhance not available")
            self.available = False
                
    def process(self, input_path, output_path):
        """Process audio file with Resemble-Enhance"""
        if not self.available:
            print("  Resemble-Enhance not available, copying input to output")
            import shutil
            shutil.copy2(input_path, output_path)
            return False
            
        try:
            # Load audio first
            audio, sr = librosa.load(input_path, sr=None)
            
            print(f"  Enhancing with Resemble-Enhance (sr={sr})...")
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            # Call enhance with required parameters
            enhanced = self.enhance_fn(audio_tensor, sr, self.device)
            
            # Convert back to numpy
            if isinstance(enhanced, torch.Tensor):
                enhanced = enhanced.cpu().numpy().squeeze()
            
            # Ensure same length
            if len(enhanced) != len(audio):
                enhanced = librosa.util.fix_length(enhanced, size=len(audio))
            
            # Save enhanced audio
            sf.write(output_path, enhanced, sr)
            print(f"  ✓ Enhanced successfully")
            return True
            
        except Exception as e:
            print(f"  ✗ Enhancement failed: {e}")
            # Copy original if enhancement fails
            import shutil
            shutil.copy2(input_path, output_path)
            return False


def main():
    """Main processing pipeline"""
    
    # Initialize processor
    processor = ResembleEnhanceProcessor()
    
    if not processor.available:
        print("\n❌ Resemble-Enhance is not installed.")
        print("Please install it with:")
        print("pip install git+https://github.com/resemble-ai/resemble-enhance.git")
        return
    
    # Set up directories
    results_dir = Path("intelligent_enhancement_results")
    
    # Process each sample
    print(f"\n{'='*80}")
    print("Processing samples with Resemble-Enhance")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for i in range(1, 51):
        sample_id = f"{i:02d}"
        
        # File paths
        metricgan_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
        resemble_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan_then_resemble_enhance.wav"
        
        if not metricgan_path.exists():
            print(f"Sample {sample_id}: Missing MetricGAN+ file, skipping")
            continue
            
        print(f"Sample {sample_id}:")
        
        try:
            # Process with Resemble-Enhance
            start_time = time.time()
            success = processor.process(str(metricgan_path), str(resemble_path))
            process_time = time.time() - start_time
            
            print(f"  Processing time: {process_time:.2f}s")
            
            if success:
                successful += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
            
        if i % 10 == 0:
            print(f"\nProgress: {i}/50 samples processed\n")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  Successfully enhanced: {successful}/50 samples")
    print(f"  Failed/Copied: {failed}/50 samples")
    
    if successful > 0:
        print(f"\n✓ Enhanced audio files saved with suffix: _final_pattern_then_metricgan_then_resemble_enhance.wav")
    
    print(f"\n🎯 All files in: {results_dir}/")


if __name__ == "__main__":
    main()