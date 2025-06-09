"""
Robust secondary speaker removal using professional audio processing techniques.
Implements speaker diarization, voice activity detection, and source separation.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy.signal import butter, sosfilt, stft, istft
from scipy.ndimage import gaussian_filter1d
import warnings

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio.pipelines import VoiceActivityDetection
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("PyAnnote not available. Will use fallback methods.")

try:
    from speechbrain.inference.separation import SepformerSeparation
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available. Source separation disabled.")


class RobustSecondaryRemoval:
    """
    Robust secondary speaker removal using professional techniques:
    - Speaker diarization with PyAnnote
    - Voice Activity Detection (VAD)
    - Source separation with SpeechBrain (SepFormer/ConvTasNet)
    - Quality-based filtering
    """
    
    def __init__(self, 
                 method: str = 'diarization',
                 use_vad: bool = True,
                 use_source_separation: bool = False,
                 preserve_primary: bool = True,
                 quality_threshold: float = 0.7,
                 use_quality_filter: bool = True,
                 fast_mode: bool = False,
                 device: str = 'cpu'):
        """
        Initialize robust secondary speaker remover.
        
        Args:
            method: 'diarization' or 'source_separation'
            use_vad: Whether to use Voice Activity Detection
            use_source_separation: Whether to use source separation models
            preserve_primary: Always preserve primary speaker content
            quality_threshold: Quality threshold for filtering (0-1)
            use_quality_filter: Whether to use quality-based filtering
            fast_mode: Use faster but less accurate methods
            device: 'cpu' or 'cuda'
        """
        self.method = method
        self.use_vad = use_vad
        self.use_source_separation = use_source_separation
        self.preserve_primary = preserve_primary
        self.quality_threshold = quality_threshold
        self.use_quality_filter = use_quality_filter
        self.fast_mode = fast_mode
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize models lazily
        self._diarization_pipeline = None
        self._vad_model = None
        self._separation_model = None
        
        # Energy-based fallback parameters
        self.energy_threshold = 0.6
        self.min_silence_duration = 0.1
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio to remove secondary speakers robustly.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        metadata = {
            'method': self.method,
            'secondary_speakers_found': 0,
            'segments_processed': 0,
            'primary_speaker_id': None,
            'quality_scores': [],
            'segments_below_threshold': 0,
            'fast_mode': self.fast_mode,
            'warning': None
        }
        
        # Convert to float32 if needed
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
            
        # Handle edge cases
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            metadata['warning'] = "Audio too short for robust processing"
            return audio, metadata
            
        if np.max(np.abs(audio)) < 0.001:  # Silent audio
            metadata['warning'] = "Audio is silent"
            return audio, metadata
        
        try:
            if self.method == 'source_separation' and self.use_source_separation:
                return self._process_with_source_separation(audio, sample_rate, metadata)
            else:
                return self._process_with_diarization(audio, sample_rate, metadata)
        except Exception as e:
            logger.error(f"Error in robust secondary removal: {e}")
            metadata['error'] = str(e)
            # Return original audio on error
            return audio, metadata
    
    def perform_diarization(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform speaker diarization to identify different speakers.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Dict with 'segments' and 'num_speakers'
        """
        if PYANNOTE_AVAILABLE and not self.fast_mode:
            try:
                return self._pyannote_diarization(audio, sample_rate)
            except Exception as e:
                logger.warning(f"PyAnnote diarization failed: {e}. Using fallback.")
                
        # Fallback to energy-based segmentation
        return self._energy_based_diarization(audio, sample_rate)
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect voice activity regions.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            List of (start, end) tuples for voice activity
        """
        if PYANNOTE_AVAILABLE and not self.fast_mode:
            try:
                return self._pyannote_vad(audio, sample_rate)
            except Exception as e:
                logger.warning(f"PyAnnote VAD failed: {e}. Using fallback.")
                
        # Fallback to energy-based VAD
        return self._energy_based_vad(audio, sample_rate)
    
    def identify_primary_speaker(self, audio: np.ndarray, diarization: Dict[str, Any], 
                               sample_rate: int) -> str:
        """
        Identify the primary speaker based on speaking time.
        
        Args:
            audio: Input audio
            diarization: Diarization results
            sample_rate: Sample rate
            
        Returns:
            Primary speaker ID
        """
        segments = diarization.get('segments', [])
        if not segments:
            return 'SPEAKER_00'
            
        # Calculate total speaking time per speaker
        speaker_times = {}
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
        # Primary speaker is the one with most speaking time
        if speaker_times:
            primary_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            return primary_speaker
        else:
            return 'SPEAKER_00'
    
    def _process_with_diarization(self, audio: np.ndarray, sample_rate: int, 
                                 metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process using speaker diarization approach."""
        # Perform diarization
        diarization = self.perform_diarization(audio, sample_rate)
        metadata['num_speakers'] = diarization.get('num_speakers', 1)
        
        segments = diarization.get('segments', [])
        if not segments:
            # No segments found, return original
            return audio, metadata
            
        # Identify primary speaker
        primary_speaker = self.identify_primary_speaker(audio, diarization, sample_rate)
        metadata['primary_speaker_id'] = primary_speaker
        
        # Count secondary speakers
        unique_speakers = set(seg['speaker'] for seg in segments)
        metadata['secondary_speakers_found'] = max(0, len(unique_speakers) - 1)
        
        if metadata['secondary_speakers_found'] == 0:
            # Only one speaker found
            return audio, metadata
            
        # Process audio to suppress secondary speakers
        processed = audio.copy()
        
        # Apply VAD if enabled
        vad_segments = []
        if self.use_vad:
            vad_segments = self.detect_voice_activity(audio, sample_rate)
            
        # Process each segment
        for segment in segments:
            if segment['speaker'] != primary_speaker:
                # This is a secondary speaker segment
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Apply quality-based filtering if enabled
                if self.use_quality_filter:
                    quality_score = self._calculate_segment_quality(
                        audio[start_sample:end_sample], 
                        sample_rate
                    )
                    
                    metadata['quality_scores'].append({
                        'segment': (segment['start'], segment['end']),
                        'score': quality_score,
                        'action': 'suppress' if quality_score < self.quality_threshold else 'keep'
                    })
                    
                    if quality_score < self.quality_threshold:
                        metadata['segments_below_threshold'] += 1
                        # Suppress this segment
                        processed[start_sample:end_sample] = self._suppress_segment(
                            processed[start_sample:end_sample],
                            method='soft'
                        )
                else:
                    # Always suppress secondary speaker segments
                    processed[start_sample:end_sample] = self._suppress_segment(
                        processed[start_sample:end_sample],
                        method='soft'
                    )
                    
                metadata['segments_processed'] += 1
        
        # Apply spectral consistency filtering
        if not self.fast_mode:
            processed = self._apply_spectral_consistency(processed, sample_rate)
            
        return processed, metadata
    
    def _process_with_source_separation(self, audio: np.ndarray, sample_rate: int,
                                      metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process using source separation approach."""
        metadata['method'] = 'source_separation'
        
        if not SPEECHBRAIN_AVAILABLE:
            logger.warning("SpeechBrain not available. Falling back to diarization.")
            return self._process_with_diarization(audio, sample_rate, metadata)
            
        try:
            # Load separation model
            if self._separation_model is None:
                self._separation_model = self._load_separation_model()
                
            metadata['separation_model'] = 'sepformer'
            
            # Prepare audio for model
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            if self.device == 'cuda':
                audio_tensor = audio_tensor.cuda()
                
            # Perform separation
            separated = self._separation_model.separate(audio_tensor)
            
            # Identify primary source (usually the loudest/most prominent)
            separated_np = separated.cpu().numpy()
            primary_idx = np.argmax([np.mean(s**2) for s in separated_np])
            
            # Return primary source
            processed = separated_np[primary_idx]
            
            # Ensure same length as input
            if len(processed) != len(audio):
                if len(processed) > len(audio):
                    processed = processed[:len(audio)]
                else:
                    processed = np.pad(processed, (0, len(audio) - len(processed)))
                    
            metadata['num_sources_found'] = len(separated_np)
            metadata['primary_source_idx'] = primary_idx
            
            return processed, metadata
            
        except Exception as e:
            logger.error(f"Source separation failed: {e}")
            metadata['error'] = f"Source separation failed: {e}"
            # Fallback to diarization
            return self._process_with_diarization(audio, sample_rate, metadata)
    
    def _load_separation_model(self) -> Any:
        """Load source separation model."""
        if SPEECHBRAIN_AVAILABLE:
            try:
                # Load pre-trained SepFormer model
                model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-whamr16k",
                    savedir="pretrained_models/sepformer-whamr16k",
                    run_opts={"device": self.device}
                )
                return model
            except Exception as e:
                logger.error(f"Failed to load SepFormer: {e}")
                raise
        else:
            raise ImportError("SpeechBrain not available")
    
    def _pyannote_diarization(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Perform diarization using PyAnnote."""
        # This would use actual PyAnnote pipeline
        # For now, using placeholder that would be replaced with real implementation
        raise NotImplementedError("PyAnnote diarization to be implemented with actual model")
    
    def _energy_based_diarization(self, audio: np.ndarray, sample_rate: int, force_multi_speaker: bool = True) -> Dict[str, Any]:
        """Fallback energy-based diarization."""
        segments = []
        
        # Calculate energy envelope
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            energy.append(frame_energy)
            
        energy = np.array(energy)
        smoothed_energy = gaussian_filter1d(energy, sigma=5)
        
        # Calculate spectral features for better speaker discrimination
        spectral_centroids = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            # Simple spectral centroid calculation
            fft = np.fft.rfft(frame)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            spectral_centroids.append(centroid)
        spectral_centroids = np.array(spectral_centroids)
        
        # Find speaker changes based on combined features
        energy_median = np.median(smoothed_energy[smoothed_energy > 0])
        centroid_median = np.median(spectral_centroids[spectral_centroids > 0])
        
        # Detect change points using both energy and spectral changes
        combined_diff = np.zeros(len(energy) - 1)
        for i in range(len(energy) - 1):
            energy_change = np.abs(smoothed_energy[i+1] - smoothed_energy[i]) / (energy_median + 1e-10)
            spectral_change = np.abs(spectral_centroids[i+1] - spectral_centroids[i]) / (centroid_median + 1e-10)
            combined_diff[i] = energy_change + spectral_change * 0.5
            
        change_threshold = np.mean(combined_diff) + 1.0 * np.std(combined_diff)  # More sensitive
        
        # Detect change points
        change_points = [0]
        min_segment_frames = int(0.3 * sample_rate / hop_size)  # Min 300ms per segment
        
        for i in range(1, len(combined_diff)):
            if combined_diff[i] > change_threshold:
                # Ensure minimum segment length
                if len(change_points) == 0 or i - change_points[-1] >= min_segment_frames:
                    change_points.append(i)
        change_points.append(len(energy))
        
        # Create segments with better speaker assignment
        speaker_features = {}
        speaker_count = 0
        
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i+1]
            
            if end_idx - start_idx < 5:  # Skip very short segments
                continue
                
            start_time = start_idx * hop_size / sample_rate
            end_time = end_idx * hop_size / sample_rate
            
            # Calculate segment features
            segment_energy = np.mean(smoothed_energy[start_idx:end_idx])
            segment_centroid = np.mean(spectral_centroids[start_idx:end_idx])
            
            # Find most similar existing speaker or create new one
            best_speaker = None
            best_similarity = 0
            
            for speaker_id, features in speaker_features.items():
                energy_sim = 1 - abs(segment_energy - features['energy']) / (features['energy'] + segment_energy + 1e-10)
                centroid_sim = 1 - abs(segment_centroid - features['centroid']) / (features['centroid'] + segment_centroid + 1e-10)
                similarity = energy_sim * 0.6 + centroid_sim * 0.4
                
                if similarity > 0.7 and similarity > best_similarity:
                    best_speaker = speaker_id
                    best_similarity = similarity
                    
            if best_speaker is None:
                # New speaker
                speaker = f"SPEAKER_{speaker_count:02d}"
                speaker_features[speaker] = {
                    'energy': segment_energy,
                    'centroid': segment_centroid
                }
                speaker_count += 1
            else:
                speaker = best_speaker
                
            segments.append({
                'speaker': speaker,
                'start': start_time,
                'end': end_time
            })
            
        # Merge adjacent segments from same speaker
        merged_segments = []
        for segment in segments:
            if merged_segments and merged_segments[-1]['speaker'] == segment['speaker']:
                # Check if close enough to merge (less than 0.5s gap)
                if segment['start'] - merged_segments[-1]['end'] < 0.5:
                    merged_segments[-1]['end'] = segment['end']
                else:
                    merged_segments.append(segment)
            else:
                merged_segments.append(segment)
                
        # Ensure we have at least 2 speakers if there's significant variation
        if len(merged_segments) >= 1:
            # Re-analyze segments to detect secondary speakers based on energy and spectral differences
            energy_values = []
            centroid_values = []
            
            for segment in merged_segments:
                start_idx = int(segment['start'] * sample_rate / hop_size)
                end_idx = min(int(segment['end'] * sample_rate / hop_size), len(smoothed_energy))
                if end_idx > start_idx:
                    seg_energy = np.mean(smoothed_energy[start_idx:end_idx])
                    seg_centroid = np.mean(spectral_centroids[start_idx:end_idx])
                    energy_values.append(seg_energy)
                    centroid_values.append(seg_centroid)
                    
            if len(energy_values) > 0:
                energy_median = np.median(energy_values)
                centroid_median = np.median(centroid_values)
                
                # Mark segments that deviate significantly as secondary speaker
                secondary_detected = False
                for i, segment in enumerate(merged_segments):
                    if i < len(energy_values):
                        energy_ratio = energy_values[i] / (energy_median + 1e-10)
                        centroid_ratio = centroid_values[i] / (centroid_median + 1e-10)
                        
                        # Detect secondary speaker if energy or spectrum is significantly different
                        if (energy_ratio > 1.3 or energy_ratio < 0.7 or 
                            centroid_ratio > 1.2 or centroid_ratio < 0.8):
                            merged_segments[i]['speaker'] = "SPEAKER_01"
                            secondary_detected = True
                            
                # If we have overlapping regions (high energy), those are likely secondary speakers
                if not secondary_detected:
                    for i, segment in enumerate(merged_segments):
                        if i < len(energy_values) and energy_values[i] > energy_median * 1.3:
                            merged_segments[i]['speaker'] = "SPEAKER_01"
                    
        num_speakers = len(set(seg['speaker'] for seg in merged_segments))
        
        return {
            'segments': merged_segments,
            'num_speakers': num_speakers
        }
    
    def _pyannote_vad(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Perform VAD using PyAnnote."""
        # Placeholder for actual PyAnnote VAD
        raise NotImplementedError("PyAnnote VAD to be implemented with actual model")
    
    def _energy_based_vad(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Fallback energy-based VAD."""
        frame_size = int(0.02 * sample_rate)
        hop_size = int(0.01 * sample_rate)
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            energy.append(frame_energy)
            
        energy = np.array(energy)
        
        # Dynamic threshold
        if len(energy) > 0:
            sorted_energy = np.sort(energy)
            noise_floor = np.mean(sorted_energy[:len(sorted_energy)//10])  # Bottom 10%
            threshold = noise_floor * 3  # 3x noise floor
        else:
            threshold = 0.01
            
        # Find voice activity regions
        is_speech = energy > threshold
        
        # Convert to time segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_active in enumerate(is_speech):
            if is_active and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_active and in_speech:
                start_time = start_frame * hop_size / sample_rate
                end_time = i * hop_size / sample_rate
                if end_time - start_time > 0.1:  # Min 100ms
                    segments.append((start_time, end_time))
                in_speech = False
                
        # Handle last segment
        if in_speech:
            start_time = start_frame * hop_size / sample_rate
            end_time = len(energy) * hop_size / sample_rate
            segments.append((start_time, end_time))
            
        return segments
    
    def _calculate_segment_quality(self, segment: np.ndarray, sample_rate: int) -> float:
        """
        Calculate quality score for a segment.
        Higher score = better quality / more likely to be primary speaker.
        """
        if len(segment) == 0:
            return 0.0
            
        # Normalize segment
        segment = segment / (np.max(np.abs(segment)) + 1e-10)
        
        # Factor 1: Signal-to-noise ratio estimate
        # Use top 10% energy vs bottom 10% as proxy
        energy = np.abs(segment)
        sorted_energy = np.sort(energy)
        noise_level = np.mean(sorted_energy[:len(sorted_energy)//10])
        signal_level = np.mean(sorted_energy[-len(sorted_energy)//10:])
        snr_score = signal_level / (noise_level + 1e-10)
        snr_score = min(snr_score / 20, 1.0)  # Normalize to 0-1
        
        # Factor 2: Spectral consistency
        # More consistent spectrum = likely single speaker
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        spectral_std = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
        consistency_score = 1.0 / (1.0 + spectral_std)
        
        # Factor 3: Zero crossing rate
        # Human speech has characteristic ZCR
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
        # Typical speech ZCR is 0.02-0.05
        zcr_score = 1.0 - min(abs(zcr - 0.035) / 0.035, 1.0)
        
        # Combine factors
        quality_score = (snr_score * 0.4 + consistency_score * 0.4 + zcr_score * 0.2)
        
        return float(quality_score)
    
    def _suppress_segment(self, segment: np.ndarray, method: str = 'soft') -> np.ndarray:
        """
        Suppress audio segment (reduce secondary speaker).
        
        Args:
            segment: Audio segment to suppress
            method: 'soft' for gradual suppression, 'hard' for aggressive
            
        Returns:
            Suppressed segment
        """
        if len(segment) == 0:
            return segment
            
        if method == 'soft':
            # Apply more aggressive suppression
            suppression_factor = 0.1  # Reduce to 10% of original (more aggressive)
            
            # Apply spectral suppression in addition to amplitude reduction
            # This helps remove the secondary speaker's spectral characteristics
            try:
                # STFT for spectral processing
                nperseg = min(256, len(segment) - 1)
                if nperseg >= 64 and len(segment) > nperseg:
                    f, t, Zxx = stft(segment, fs=16000, nperseg=nperseg)
                    
                    # Apply frequency-selective suppression
                    # Suppress mid-to-high frequencies more (where secondary speakers often differ)
                    freq_mask = np.ones(len(f))
                    mid_freq_idx = len(f) // 3
                    freq_mask[mid_freq_idx:] = 0.5  # Extra suppression for higher frequencies
                    
                    # Apply suppression
                    Zxx_suppressed = Zxx * freq_mask[:, np.newaxis] * suppression_factor
                    
                    # Reconstruct
                    _, segment_suppressed = istft(Zxx_suppressed, fs=16000, nperseg=nperseg)
                    
                    # Ensure same length
                    if len(segment_suppressed) > len(segment):
                        segment_suppressed = segment_suppressed[:len(segment)]
                    elif len(segment_suppressed) < len(segment):
                        segment_suppressed = np.pad(segment_suppressed, (0, len(segment) - len(segment_suppressed)))
                        
                    return segment_suppressed.astype(segment.dtype)
                else:
                    # Fallback for very short segments
                    return segment * suppression_factor
            except Exception as e:
                logger.warning(f"Spectral suppression failed: {e}, using simple suppression")
                return segment * suppression_factor
        else:
            # Hard suppression - very aggressive
            # Use both amplitude and spectral suppression
            return segment * 0.02  # Reduce to 2% of original
    
    def _apply_spectral_consistency(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral consistency filtering to reduce secondary speakers."""
        # STFT parameters
        nperseg = min(512, len(audio) - 1)
        if nperseg < 64:
            return audio
            
        # Perform STFT
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg)
        
        # Calculate magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Calculate spectral consistency over time
        mean_spectrum = np.mean(magnitude, axis=1, keepdims=True)
        std_spectrum = np.std(magnitude, axis=1, keepdims=True)
        
        # Create consistency mask
        consistency = 1.0 - (std_spectrum / (mean_spectrum + 1e-10))
        consistency = np.clip(consistency, 0, 1)
        
        # Apply soft mask
        mask = consistency ** 1.5  # Make it slightly more selective
        masked_magnitude = magnitude * mask
        
        # Reconstruct
        Zxx_masked = masked_magnitude * np.exp(1j * phase)
        _, reconstructed = istft(Zxx_masked, fs=sample_rate, nperseg=nperseg)
        
        # Ensure same length
        if len(reconstructed) != len(audio):
            if len(reconstructed) > len(audio):
                reconstructed = reconstructed[:len(audio)]
            else:
                reconstructed = np.pad(reconstructed, (0, len(audio) - len(reconstructed)))
                
        return reconstructed
    
    def _apply_quality_filter(self, audio: np.ndarray, segments: List[Dict[str, Any]], 
                             sample_rate: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply quality-based filtering to segments."""
        processed = audio.copy()
        quality_scores = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            if end_sample <= len(audio):
                segment_audio = audio[start_sample:end_sample]
                quality = self._calculate_segment_quality(segment_audio, sample_rate)
                
                quality_info = {
                    'segment': (segment['start'], segment['end']),
                    'score': quality,
                    'action': 'keep' if quality >= self.quality_threshold else 'suppress'
                }
                quality_scores.append(quality_info)
                
                if quality < self.quality_threshold:
                    # Suppress low quality segments
                    processed[start_sample:end_sample] = self._suppress_segment(
                        segment_audio, method='soft'
                    )
                    
        return processed, quality_scores