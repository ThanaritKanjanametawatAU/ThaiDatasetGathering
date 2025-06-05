#!/usr/bin/env python3
"""
Test script specifically for S3 and S5 samples that have secondary speaker issues.

This script loads and processes the problematic samples to verify that
the SpeechBrain implementation successfully removes secondary speakers.
"""

import os
import sys
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speechbrain_separator import (
    SpeechBrainSeparator,
    SeparationConfig
)
from processors.audio_enhancement.core import AudioEnhancer
from utils.huggingface import authenticate_hf, read_hf_token
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_samples():
    """Download S3 and S5 samples from the dataset"""
    logger.info("Authenticating with HuggingFace...")
    
    # Read token
    token = read_hf_token()
    if not token:
        logger.error("No HuggingFace token found. Please add your token to hf_token.txt")
        sys.exit(1)
    
    # Authenticate
    authenticate_hf(token)
    
    logger.info("Loading dataset samples...")
    
    # Load just a few samples
    dataset = load_dataset(
        "Thanarit/Thai-Voice",
        split="train",
        streaming=True,
        token=token
    )
    
    # Get samples S3 and S5 (and some others for comparison)
    samples_to_test = ['S1', 'S3', 'S5', 'S9', 'S10']
    found_samples = {}
    
    logger.info(f"Looking for samples: {samples_to_test}")
    
    count = 0
    for sample in dataset:
        if sample['ID'] in samples_to_test:
            found_samples[sample['ID']] = sample
            logger.info(f"Found sample {sample['ID']}")
            
            if len(found_samples) == len(samples_to_test):
                break
        
        count += 1
        if count > 100:  # Don't search too long
            logger.warning("Searched 100 samples, stopping...")
            break
    
    return found_samples


def test_speechbrain_separation(samples):
    """Test SpeechBrain separation on the samples"""
    
    # Create output directory
    output_dir = Path("test_separation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize separator with optimal settings
    config = SeparationConfig(
        device="cuda",
        confidence_threshold=0.7,
        batch_size=16,
        speaker_selection="energy",
        use_mixed_precision=True,
        quality_thresholds={
            "min_pesq": 3.5,
            "min_stoi": 0.85,
            "max_spectral_distortion": 0.15
        }
    )
    
    logger.info("Initializing SpeechBrain separator...")
    separator = SpeechBrainSeparator(config)
    
    # Also test with AudioEnhancer
    logger.info("Initializing AudioEnhancer...")
    enhancer = AudioEnhancer(
        use_gpu=True,
        enhancement_level="ultra_aggressive"
    )
    
    results = {}
    
    for sample_id, sample in samples.items():
        logger.info(f"\nProcessing {sample_id}...")
        
        # Extract audio
        audio_data = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        
        # Ensure 16kHz
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
            sample_rate = 16000
        
        # Save original
        original_path = output_dir / f"{sample_id}_original.wav"
        sf.write(original_path, audio_data, sample_rate)
        
        # Test 1: Direct SpeechBrain separation
        logger.info(f"  - Running SpeechBrain separation...")
        sep_result = separator.separate_speakers(audio_data, sample_rate)
        
        if not sep_result.rejected:
            separated_path = output_dir / f"{sample_id}_speechbrain.wav"
            sf.write(separated_path, sep_result.audio, sample_rate)
            
            logger.info(f"  - Confidence: {sep_result.confidence:.3f}")
            logger.info(f"  - Speakers detected: {sep_result.num_speakers_detected}")
            logger.info(f"  - Processing time: {sep_result.processing_time_ms:.1f}ms")
            
            # Print metrics
            for metric, value in sep_result.metrics.items():
                logger.info(f"  - {metric}: {value:.3f}")
        else:
            logger.warning(f"  - REJECTED: {sep_result.rejection_reason}")
        
        # Test 2: Full enhancement pipeline
        logger.info(f"  - Running full enhancement pipeline...")
        enhanced_audio, metadata = enhancer.enhance(audio_data, sample_rate)
        
        enhanced_path = output_dir / f"{sample_id}_enhanced.wav"
        sf.write(enhanced_path, enhanced_audio, sample_rate)
        
        # Store results
        results[sample_id] = {
            'speechbrain_result': sep_result,
            'enhancement_metadata': metadata.get('enhancement_metadata', {})
        }
        
        # Compare energies
        orig_energy = np.sqrt(np.mean(audio_data ** 2))
        sep_energy = np.sqrt(np.mean(sep_result.audio ** 2)) if not sep_result.rejected else 0
        enh_energy = np.sqrt(np.mean(enhanced_audio ** 2))
        
        logger.info(f"  - Energy comparison:")
        logger.info(f"    - Original: {orig_energy:.4f}")
        logger.info(f"    - SpeechBrain: {sep_energy:.4f}")
        logger.info(f"    - Enhanced: {enh_energy:.4f}")
    
    return results


def analyze_results(results):
    """Analyze and report on the results"""
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    
    # Check S3 and S5 specifically
    for sample_id in ['S3', 'S5']:
        if sample_id in results:
            result = results[sample_id]
            sep_result = result['speechbrain_result']
            
            logger.info(f"\n{sample_id} Analysis:")
            
            if not sep_result.rejected:
                logger.info("  ✓ Successfully processed")
                logger.info(f"  - Confidence: {sep_result.confidence:.3f}")
                logger.info(f"  - STOI: {sep_result.metrics.get('stoi', 0):.3f}")
                logger.info(f"  - Speakers: {sep_result.num_speakers_detected}")
                
                # Check if secondary speakers were detected and handled
                if sep_result.num_speakers_detected > 1:
                    logger.info("  ✓ Multiple speakers detected and separated")
                else:
                    logger.info("  ! Only one speaker detected")
            else:
                logger.error(f"  ✗ FAILED: {sep_result.rejection_reason}")
    
    # Overall statistics
    total = len(results)
    successful = sum(1 for r in results.values() 
                    if not r['speechbrain_result'].rejected)
    
    logger.info(f"\nOverall Results:")
    logger.info(f"  - Total processed: {total}")
    logger.info(f"  - Successful: {successful} ({successful/total*100:.1f}%)")
    
    # Speaker detection stats
    multi_speaker = sum(1 for r in results.values() 
                       if r['speechbrain_result'].num_speakers_detected > 1)
    logger.info(f"  - Multi-speaker detected: {multi_speaker}")


def main():
    """Main test function"""
    
    logger.info("Starting S3/S5 Secondary Speaker Removal Test")
    logger.info("="*60)
    
    # Option 1: Download from HuggingFace
    try:
        samples = download_samples()
    except Exception as e:
        logger.error(f"Failed to download samples: {e}")
        logger.info("Please ensure you have access to Thanarit/Thai-Voice dataset")
        sys.exit(1)
    
    if not samples:
        logger.error("No samples found!")
        sys.exit(1)
    
    # Test separation
    results = test_speechbrain_separation(samples)
    
    # Analyze results
    analyze_results(results)
    
    logger.info("\n" + "="*60)
    logger.info("Test complete! Check 'test_separation_results' directory for output files.")
    logger.info("Compare original vs separated files to verify secondary speaker removal.")
    
    # Provide listening instructions
    logger.info("\nListening Guide:")
    logger.info("  - S3_original.wav vs S3_speechbrain.wav")
    logger.info("  - S5_original.wav vs S5_speechbrain.wav")
    logger.info("  - Listen for secondary speakers in the background")
    logger.info("  - Verify they are removed in the processed versions")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Test with local files
        logger.info("Testing with local files...")
        # Add local file testing code here
    else:
        # Test with downloaded samples
        main()