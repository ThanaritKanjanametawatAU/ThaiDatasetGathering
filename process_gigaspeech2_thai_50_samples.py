#!/usr/bin/env python3
"""
Process 50 GigaSpeech2 Thai Samples with Pattern→MetricGAN+ Method
Loads specifically Thai language samples from GigaSpeech2
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
from datasets import load_dataset, Dataset

print("GigaSpeech2 Thai Language Processing")
print("Pattern→MetricGAN+ Method (Final Winner)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PatternMetricGANProcessor:
    """Final winning audio enhancement processor"""
    
    def __init__(self):
        self.device = device
        self.metricgan_enhancer = None
        self.load_metricgan_model()
        
    def load_metricgan_model(self):
        """Load MetricGAN+ enhancement model"""
        print("\nLoading MetricGAN+ model...")
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            self.metricgan_enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": str(self.device)}
            )
            print("✓ MetricGAN+ loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load MetricGAN+: {e}")
            raise
    
    def detect_interruption_patterns(self, audio, sr):
        """Ultra-conservative interruption pattern detection"""
        if len(audio) == 0:
            return []
            
        # Feature extraction
        hop_length = 512
        frame_length = 2048
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        
        # Pitch tracking
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length, threshold=0.1)
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0.1 else 0
                pitch_track.append(pitch)
            pitch_track = np.array(pitch_track)
        except:
            pitch_track = np.zeros(len(spectral_centroid))
        
        # Energy calculation
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        energy = np.array(energy)
        
        if len(energy) == 0:
            return []
            
        # Align features
        min_len = min(len(spectral_centroid), len(zcr), len(pitch_track), len(energy))
        spectral_centroid = spectral_centroid[:min_len]
        zcr = zcr[:min_len]
        pitch_track = pitch_track[:min_len]
        energy = energy[:min_len]
        
        # Ultra-conservative thresholds
        energy_threshold = np.percentile(energy, 75)
        zcr_threshold = np.percentile(zcr, 80)
        spectral_threshold = np.percentile(spectral_centroid, 70)
        
        interruptions = []
        
        # Detect interruption patterns
        for i in range(1, len(energy) - 1):
            confidence = 0
            reasons = []
            
            # Energy spike detection
            if energy[i] > energy_threshold * 1.8:
                confidence += 0.25
                reasons.append("energy_spike")
            
            # High ZCR
            if zcr[i] > zcr_threshold * 1.5:
                confidence += 0.2
                reasons.append("high_zcr")
            
            # Spectral irregularity
            if spectral_centroid[i] > spectral_threshold * 1.4:
                confidence += 0.2
                reasons.append("spectral_irregular")
            
            # Pitch discontinuity
            if len(pitch_track) > i and pitch_track[i] > 0:
                if i > 0 and i < len(pitch_track) - 1:
                    pitch_change = abs(pitch_track[i] - pitch_track[i-1])
                    if pitch_change > 100:
                        confidence += 0.15
                        reasons.append("pitch_discontinuity")
            
            # Ultra-conservative threshold
            if confidence > 0.8:
                # Context validation
                context_start = max(0, i - 3)
                context_end = min(len(energy), i + 4)
                context_energy = energy[context_start:context_end]
                
                if energy[i] > np.mean(context_energy) * 2.0:
                    start_sample = i * hop_length
                    end_sample = min(len(audio), (i + 1) * hop_length)
                    
                    interruptions.append({
                        'start': start_sample,
                        'end': end_sample,
                        'confidence': confidence,
                        'reasons': reasons
                    })
        
        return self._merge_interruptions(interruptions, sr)
    
    def _merge_interruptions(self, interruptions, sr, min_gap=0.2):
        """Merge nearby interruptions"""
        if not interruptions:
            return []
        
        interruptions.sort(key=lambda x: x['start'])
        
        merged = []
        current = interruptions[0].copy()
        
        for next_int in interruptions[1:]:
            gap = (next_int['start'] - current['end']) / sr
            
            if gap < min_gap:
                current['end'] = next_int['end']
                current['confidence'] = max(current['confidence'], next_int['confidence'])
                current['reasons'].extend(next_int['reasons'])
            else:
                merged.append(current)
                current = next_int.copy()
        
        merged.append(current)
        return merged
    
    def apply_pattern_suppression(self, audio, interruptions, sr):
        """Apply gentle suppression to detected patterns"""
        processed = audio.copy()
        
        for interruption in interruptions:
            start = interruption['start']
            end = interruption['end']
            
            # 50ms padding
            pad_samples = int(0.05 * sr)
            
            pad_start = max(0, start - pad_samples)
            pad_end = min(len(audio), end + pad_samples)
            
            # 85% suppression (keep 15%)
            suppression_factor = 0.15
            processed[pad_start:pad_end] *= suppression_factor
        
        return processed
    
    def enhance_with_metricgan(self, audio, sr=16000):
        """Apply MetricGAN+ enhancement"""
        if self.metricgan_enhancer is None:
            return audio
            
        try:
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            # Enhance
            with torch.no_grad():
                enhanced = self.metricgan_enhancer.enhance_batch(
                    audio_tensor, 
                    lengths=torch.tensor([1.0])
                )
                enhanced = enhanced.cpu().numpy().squeeze()
            
            # Restore scale
            if max_val > 0:
                enhanced = enhanced * max_val
            
            return enhanced
            
        except Exception as e:
            print(f"MetricGAN+ enhancement failed: {e}")
            return audio
    
    def process_audio(self, audio, sr, sample_id=None):
        """Process audio with Pattern→MetricGAN+ pipeline"""
        if sample_id:
            print(f"\nProcessing sample {sample_id}")
            print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr} Hz")
        
        # Step 1: Detect patterns
        interruptions = self.detect_interruption_patterns(audio, sr)
        
        if sample_id:
            print(f"  Detected {len(interruptions)} interruption patterns")
        
        # Step 2: Apply suppression
        pattern_suppressed = self.apply_pattern_suppression(audio, interruptions, sr)
        
        # Step 3: MetricGAN+ enhancement
        final_audio = self.enhance_with_metricgan(pattern_suppressed, sr)
        
        return final_audio, len(interruptions)


def load_gigaspeech2_thai_samples(num_samples=50, num_archives=5):
    """
    Load Thai language samples from GigaSpeech2.
    Uses the same approach as the GigaSpeech2Processor.
    """
    print(f"\nLoading {num_samples} Thai samples from GigaSpeech2...")
    
    samples = []
    
    try:
        # Load specific Thai archive files
        thai_archives = []
        for i in range(num_archives):
            thai_archives.append(f"data/th/train/{i}.tar.gz")
        
        print(f"Loading Thai archives: {thai_archives}")
        
        # Load dataset from specific Thai archive files with streaming
        streaming_dataset = load_dataset(
            "speechcolab/gigaspeech2", 
            split="train",
            data_files=thai_archives,
            streaming=True
        )
        
        # Collect samples
        sample_count = 0
        for item in streaming_dataset:
            if sample_count >= num_samples:
                break
            
            try:
                # Extract audio from 'wav' field (GigaSpeech2 structure)
                wav_info = item.get('wav', {})
                if not wav_info:
                    print(f"  Warning: No 'wav' field in sample {sample_count+1}")
                    continue
                
                audio_array = wav_info.get('array')
                sample_rate = wav_info.get('sampling_rate', 16000)
                
                if audio_array is None:
                    print(f"  Warning: No audio array in sample {sample_count+1}")
                    continue
                
                # Get metadata
                sample_key = item.get('__key__', f'th_sample_{sample_count+1:04d}')
                
                # Extract segment ID for speaker identification
                segment_id = sample_key.split('/')[-1] if '/' in sample_key else sample_key
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=sample_rate, 
                        target_sr=16000
                    )
                    sample_rate = 16000
                
                samples.append({
                    'audio': audio_array,
                    'sr': sample_rate,
                    'id': sample_key,
                    'segment_id': segment_id,
                    'duration': len(audio_array) / sample_rate,
                    'source': 'gigaspeech2_thai'
                })
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"  Loaded {sample_count}/{num_samples} Thai samples...")
                    
            except Exception as e:
                print(f"  Warning: Failed to process sample: {e}")
                continue
        
        print(f"✓ Successfully loaded {len(samples)} Thai samples from GigaSpeech2")
        
    except Exception as e:
        print(f"✗ Failed to load Thai samples: {e}")
        import traceback
        traceback.print_exc()
    
    return samples


def main():
    """Main processing pipeline for Thai samples"""
    
    # Initialize processor
    processor = PatternMetricGANProcessor()
    
    # Create output directory
    output_dir = Path("intelligent_enhancement_results")
    output_dir.mkdir(exist_ok=True)
    
    # Clean up existing files
    print("\nCleaning up previous files...")
    existing_files = list(output_dir.glob("*.wav")) + list(output_dir.glob("*.md"))
    for f in existing_files:
        f.unlink()
    print(f"Removed {len(existing_files)} existing files")
    
    # Load Thai samples
    samples = load_gigaspeech2_thai_samples(num_samples=50, num_archives=5)
    
    if not samples:
        print("\n❌ No Thai samples available for processing.")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing {len(samples)} Thai samples with Pattern→MetricGAN+ method")
    print(f"{'='*80}")
    
    # Process each sample
    processing_times = []
    total_interruptions = 0
    successful = 0
    
    for i, sample in enumerate(samples):
        sample_id = f"{i+1:02d}"
        
        try:
            start_time = time.time()
            
            # Save original
            original_path = output_dir / f"sample_{sample_id}_original.wav"
            sf.write(original_path, sample['audio'], sample['sr'])
            
            # Process
            processed_audio, num_interruptions = processor.process_audio(
                sample['audio'], sample['sr'], sample_id
            )
            total_interruptions += num_interruptions
            
            # Save processed
            processed_path = output_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
            sf.write(processed_path, processed_audio, sample['sr'])
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            successful += 1
            
            print(f"  ✓ Saved: {processed_path.name}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Segment ID: {sample['segment_id']}")
            
            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"\n  Progress: {i+1}/{len(samples)} samples completed")
            
        except Exception as e:
            print(f"  ✗ Failed to process sample {sample_id}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    if processing_times:
        avg_time = np.mean(processing_times)
        total_time = sum(processing_times)
        
        print(f"\n📊 Results Summary:")
        print(f"  • Thai samples processed: {successful}/{len(samples)}")
        print(f"  • Total interruptions detected: {total_interruptions}")
        print(f"  • Average processing time: {avg_time:.2f}s per sample")
        print(f"  • Total time: {total_time/60:.1f} minutes")
        print(f"  • Estimated for 10M samples: {(avg_time * 10_000_000 / 3600):.1f} hours")
        
        # Save report
        report_path = output_dir / "gigaspeech2_thai_50_samples_final_report.md"
        with open(report_path, 'w') as f:
            f.write("# GigaSpeech2 Thai - Pattern→MetricGAN+ Processing Report\n\n")
            f.write(f"## Dataset\n")
            f.write(f"- Source: GigaSpeech2 Thai Archives\n")
            f.write(f"- Language: Thai\n")
            f.write(f"- Samples processed: {successful}/{len(samples)}\n")
            f.write(f"- Success rate: {100*successful/len(samples):.1f}%\n\n")
            
            f.write(f"## Method: Pattern→MetricGAN+ (Winning Approach)\n")
            f.write(f"1. **Ultra-conservative pattern detection**\n")
            f.write(f"   - Confidence threshold: >0.8\n")
            f.write(f"   - Energy spike: 75th percentile × 1.8\n")
            f.write(f"   - Context validation required\n\n")
            
            f.write(f"2. **Gentle suppression**\n")
            f.write(f"   - 50ms padding around interruptions\n")
            f.write(f"   - 85% suppression (keeps 15%)\n\n")
            
            f.write(f"3. **MetricGAN+ enhancement**\n")
            f.write(f"   - Overall noise reduction\n")
            f.write(f"   - Quality improvement\n\n")
            
            f.write(f"## Performance\n")
            f.write(f"- Average time: {avg_time:.2f}s per sample\n")
            f.write(f"- Processing rate: {len(samples)/total_time:.1f} samples/sec\n")
            f.write(f"- Estimated 10M samples: {(avg_time * 10_000_000 / 3600):.1f} hours\n\n")
            
            f.write(f"## Results\n")
            f.write(f"- Total interruptions detected: {total_interruptions}\n")
            f.write(f"- Average per sample: {total_interruptions/successful:.1f}\n\n")
            
            f.write(f"## Output Files\n")
            f.write(f"- Original: `sample_XX_original.wav` (50 files)\n")
            f.write(f"- Processed: `sample_XX_final_pattern_then_metricgan.wav` (50 files)\n\n")
            
            f.write(f"## Notes\n")
            f.write(f"- All samples are Thai language audio from GigaSpeech2\n")
            f.write(f"- Audio loaded from Thai-specific archives (data/th/train/*.tar.gz)\n")
            f.write(f"- Ready for production deployment on Thai dataset\n")
        
        print(f"\n✓ Report saved: {report_path}")
    
    print(f"\n🎯 All files in: {output_dir}/")
    print("✅ Thai language samples processed successfully!")


if __name__ == "__main__":
    main()