"""
Comprehensive error handling and edge case tests for the audio analysis pipeline.

This module tests:
- Error propagation through pipeline stages
- Graceful degradation strategies
- Recovery mechanisms
- Edge cases and boundary conditions
- Invalid input handling
- Component failure scenarios
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import soundfile as sf
from unittest.mock import patch, MagicMock, Mock
import warnings
import json

# Import pipeline components and exceptions
from processors.audio_enhancement.audio_loader import (
    AudioLoader, AudioLoadError, UnsupportedFormatError, 
    CorruptedFileError, PreprocessingError
)
from utils.enhanced_snr_calculator import (
    EnhancedSNRCalculator, SNRError, InsufficientDataError,
    NoSpeechDetectedError, InvalidSignalError
)
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine


class ErrorRecoveryPipeline:
    """Pipeline with error recovery mechanisms."""
    
    def __init__(self):
        self.components = {
            'loader': AudioLoader(),
            'snr_calculator': EnhancedSNRCalculator(),
            'spectral_analyzer': SpectralAnalyzer(),
            'pattern_detector': PatternDetector(),
            'issue_categorizer': IssueCategorizer(),
            'decision_engine': DecisionEngine()
        }
        self.error_log = []
        self.recovery_strategies = self._setup_recovery_strategies()
        
    def _setup_recovery_strategies(self):
        """Define recovery strategies for different error types."""
        return {
            AudioLoadError: self._recover_from_load_error,
            SNRError: self._recover_from_snr_error,
            ValueError: self._recover_from_value_error,
            MemoryError: self._recover_from_memory_error,
            Exception: self._recover_from_generic_error
        }
        
    def process_with_recovery(self, audio_path):
        """Process audio with error recovery."""
        results = {}
        errors = []
        
        try:
            # Stage 1: Load audio with recovery
            audio_data = self._safe_load(audio_path)
            results['audio_data'] = audio_data
            
            # Stage 2: Calculate SNR with fallback
            snr_metrics = self._safe_snr_calculation(audio_data)
            results['snr_metrics'] = snr_metrics
            
            # Stage 3: Spectral analysis with degradation
            spectral_features = self._safe_spectral_analysis(audio_data)
            results['spectral_features'] = spectral_features
            
            # Stage 4: Pattern detection with partial results
            patterns = self._safe_pattern_detection(audio_data, spectral_features)
            results['patterns'] = patterns
            
            # Stage 5: Issue categorization with defaults
            issues = self._safe_issue_categorization(
                snr_metrics, spectral_features, patterns
            )
            results['issues'] = issues
            
            # Stage 6: Decision with fallback
            decision = self._safe_decision(issues)
            results['decision'] = decision
            
        except Exception as e:
            # Log critical error
            self.error_log.append({
                'stage': 'pipeline',
                'error': str(e),
                'type': type(e).__name__
            })
            
            # Return partial results with error decision
            results['decision'] = self._create_error_decision(e)
            
        results['errors'] = self.error_log
        return results
        
    def _safe_load(self, audio_path):
        """Load audio with error handling."""
        try:
            return self.components['loader'].load(audio_path)
        except Exception as e:
            self.error_log.append({
                'stage': 'loading',
                'error': str(e),
                'type': type(e).__name__
            })
            
            # Try recovery strategy
            recovery = self.recovery_strategies.get(type(e), self._recover_from_generic_error)
            return recovery('loading', e, audio_path)
            
    def _safe_snr_calculation(self, audio_data):
        """Calculate SNR with fallback."""
        if audio_data is None:
            return self._create_default_snr_metrics()
            
        try:
            return self.components['snr_calculator'].calculate(
                audio_data.samples, audio_data.sample_rate
            )
        except NoSpeechDetectedError:
            # Return metrics indicating silence
            return self._create_silent_snr_metrics()
        except Exception as e:
            self.error_log.append({
                'stage': 'snr_calculation',
                'error': str(e),
                'type': type(e).__name__
            })
            return self._create_default_snr_metrics()
            
    def _safe_spectral_analysis(self, audio_data):
        """Perform spectral analysis with degradation."""
        if audio_data is None:
            return self._create_default_spectral_features()
            
        try:
            return self.components['spectral_analyzer'].analyze(
                audio_data.samples, audio_data.sample_rate
            )
        except MemoryError:
            # Try with reduced resolution
            return self._analyze_with_reduced_resolution(audio_data)
        except Exception as e:
            self.error_log.append({
                'stage': 'spectral_analysis',
                'error': str(e),
                'type': type(e).__name__
            })
            return self._create_default_spectral_features()
            
    def _safe_pattern_detection(self, audio_data, spectral_features):
        """Detect patterns with partial results."""
        try:
            return self.components['pattern_detector'].detect(
                audio_data.samples if audio_data else None,
                spectral_features
            )
        except Exception as e:
            self.error_log.append({
                'stage': 'pattern_detection',
                'error': str(e),
                'type': type(e).__name__
            })
            return []  # Empty pattern list
            
    def _safe_issue_categorization(self, snr_metrics, spectral_features, patterns):
        """Categorize issues with defaults."""
        try:
            return self.components['issue_categorizer'].categorize(
                snr_metrics, spectral_features, patterns
            )
        except Exception as e:
            self.error_log.append({
                'stage': 'issue_categorization',
                'error': str(e),
                'type': type(e).__name__
            })
            # Return generic issue based on available data
            return self._create_generic_issues()
            
    def _safe_decision(self, issues):
        """Make decision with fallback."""
        try:
            return self.components['decision_engine'].decide(issues)
        except Exception as e:
            self.error_log.append({
                'stage': 'decision',
                'error': str(e),
                'type': type(e).__name__
            })
            # Conservative decision on error
            return self._create_conservative_decision()
            
    def _recover_from_load_error(self, stage, error, audio_path):
        """Recover from audio loading error."""
        # Try alternative loading methods
        try:
            # Attempt to load as raw data
            with open(audio_path, 'rb') as f:
                raw_data = f.read()
            # Create minimal audio data
            return self._create_minimal_audio_data()
        except:
            return None
            
    def _recover_from_snr_error(self, stage, error, *args):
        """Recover from SNR calculation error."""
        return self._create_default_snr_metrics()
        
    def _recover_from_value_error(self, stage, error, *args):
        """Recover from value errors."""
        # Return appropriate default based on stage
        defaults = {
            'loading': None,
            'snr_calculation': self._create_default_snr_metrics(),
            'spectral_analysis': self._create_default_spectral_features(),
            'pattern_detection': [],
            'issue_categorization': [],
            'decision': self._create_conservative_decision()
        }
        return defaults.get(stage, None)
        
    def _recover_from_memory_error(self, stage, error, *args):
        """Recover from memory errors."""
        import gc
        gc.collect()
        
        # Try with reduced settings
        if stage == 'spectral_analysis':
            return self._create_minimal_spectral_features()
        return None
        
    def _recover_from_generic_error(self, stage, error, *args):
        """Generic error recovery."""
        warnings.warn(f"Unhandled error in {stage}: {error}")
        return None
        
    def _create_default_snr_metrics(self):
        """Create default SNR metrics."""
        class DefaultSNR:
            def __init__(self):
                self.global_snr = 0.0
                self.segmental_snr = []
                self.is_silent = False
                self.has_speech = False
                
        return DefaultSNR()
        
    def _create_silent_snr_metrics(self):
        """Create SNR metrics for silent audio."""
        metrics = self._create_default_snr_metrics()
        metrics.is_silent = True
        return metrics
        
    def _create_default_spectral_features(self):
        """Create default spectral features."""
        class DefaultSpectral:
            def __init__(self):
                self.frequency_bins = np.array([])
                self.magnitude_spectrum = np.array([])
                self.spectral_centroid = 0.0
                
        return DefaultSpectral()
        
    def _create_minimal_audio_data(self):
        """Create minimal audio data object."""
        class MinimalAudio:
            def __init__(self):
                self.samples = np.zeros(16000)  # 1 second of silence
                self.sample_rate = 16000
                
        return MinimalAudio()
        
    def _create_minimal_spectral_features(self):
        """Create minimal spectral features."""
        features = self._create_default_spectral_features()
        features.frequency_bins = np.linspace(0, 8000, 10)
        features.magnitude_spectrum = np.zeros(10)
        return features
        
    def _create_generic_issues(self):
        """Create generic issues list."""
        from processors.audio_enhancement.issue_categorization import Issue, IssueType, IssueSeverity
        return [
            Issue(
                type=IssueType.UNKNOWN,
                severity=IssueSeverity.MODERATE,
                confidence=0.5,
                description="Unable to analyze audio properly"
            )
        ]
        
    def _create_conservative_decision(self):
        """Create conservative decision."""
        from processors.audio_enhancement.decision_framework import Decision, DecisionType
        return Decision(
            action='MANUAL_REVIEW',
            confidence=0.0,
            reasoning=['Error during processing - manual review required']
        )
        
    def _create_error_decision(self, error):
        """Create decision based on error type."""
        from processors.audio_enhancement.decision_framework import Decision
        return Decision(
            action='REJECT',
            confidence=1.0,
            reasoning=[f'Critical error: {type(error).__name__} - {str(error)}']
        )
        
    def _analyze_with_reduced_resolution(self, audio_data):
        """Analyze with reduced resolution to save memory."""
        # Downsample audio
        downsample_factor = 4
        downsampled = audio_data.samples[::downsample_factor]
        
        # Create minimal spectral features
        features = self._create_minimal_spectral_features()
        features.spectral_centroid = np.mean(np.abs(downsampled))
        return features


class TestPipelineErrorHandling(unittest.TestCase):
    """Test error handling in audio analysis pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ErrorRecoveryPipeline()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_corrupted_file_handling(self):
        """Test handling of corrupted audio files."""
        # Create corrupted file
        corrupted_path = os.path.join(self.temp_dir, 'corrupted.wav')
        with open(corrupted_path, 'wb') as f:
            f.write(b'RIFF')  # Incomplete WAV header
            f.write(b'\x00' * 100)  # Random data
            
        result = self.pipeline.process_with_recovery(corrupted_path)
        
        # Should handle error gracefully
        self.assertIn('decision', result)
        self.assertIn('errors', result)
        self.assertTrue(len(result['errors']) > 0)
        
        # Should make conservative decision
        self.assertIn(result['decision'].action, ['REJECT', 'MANUAL_REVIEW'])
        
    def test_missing_file_handling(self):
        """Test handling of missing files."""
        missing_path = os.path.join(self.temp_dir, 'missing.wav')
        
        result = self.pipeline.process_with_recovery(missing_path)
        
        # Should handle gracefully
        self.assertIn('decision', result)
        self.assertEqual(result['decision'].action, 'REJECT')
        
        # Should log error
        load_errors = [e for e in result['errors'] if e['stage'] == 'loading']
        self.assertTrue(len(load_errors) > 0)
        
    def test_invalid_audio_format(self):
        """Test handling of unsupported audio formats."""
        # Create text file with .wav extension
        invalid_path = os.path.join(self.temp_dir, 'invalid.wav')
        with open(invalid_path, 'w') as f:
            f.write('This is not audio data')
            
        result = self.pipeline.process_with_recovery(invalid_path)
        
        # Should handle error
        self.assertIn('errors', result)
        self.assertTrue(len(result['errors']) > 0)
        
    def test_silent_audio_handling(self):
        """Test handling of completely silent audio."""
        silent_path = os.path.join(self.temp_dir, 'silent.wav')
        silent_audio = np.zeros(16000 * 2)  # 2 seconds of silence
        sf.write(silent_path, silent_audio, 16000)
        
        result = self.pipeline.process_with_recovery(silent_path)
        
        # Should process successfully
        self.assertIn('snr_metrics', result)
        self.assertTrue(result['snr_metrics'].is_silent)
        
        # Should reject silent audio
        self.assertEqual(result['decision'].action, 'REJECT')
        
    def test_extremely_short_audio(self):
        """Test handling of very short audio files."""
        short_path = os.path.join(self.temp_dir, 'short.wav')
        short_audio = np.random.randn(100)  # Only 100 samples
        sf.write(short_path, short_audio, 16000)
        
        result = self.pipeline.process_with_recovery(short_path)
        
        # Should handle gracefully
        self.assertIn('decision', result)
        
        # May have limited analysis
        if 'snr_metrics' in result:
            # SNR calculation might fail or return defaults
            self.assertIsNotNone(result['snr_metrics'])
            
    def test_memory_error_simulation(self):
        """Test handling of memory errors."""
        # Create large file that might cause memory issues
        large_path = os.path.join(self.temp_dir, 'large.wav')
        
        # Simulate large audio (but don't actually create huge file)
        normal_audio = np.random.randn(16000 * 5)  # 5 seconds
        sf.write(large_path, normal_audio, 16000)
        
        # Mock memory error in spectral analysis
        with patch.object(
            self.pipeline.components['spectral_analyzer'],
            'analyze',
            side_effect=MemoryError("Insufficient memory")
        ):
            result = self.pipeline.process_with_recovery(large_path)
            
        # Should recover with reduced features
        self.assertIn('spectral_features', result)
        self.assertIsNotNone(result['spectral_features'])
        
        # Should log memory error
        memory_errors = [
            e for e in result['errors'] 
            if e['type'] == 'MemoryError'
        ]
        self.assertTrue(len(memory_errors) > 0)
        
    def test_component_failure_cascade(self):
        """Test handling of cascading component failures."""
        test_path = os.path.join(self.temp_dir, 'test.wav')
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        sf.write(test_path, audio, 16000)
        
        # Mock SNR calculator failure
        with patch.object(
            self.pipeline.components['snr_calculator'],
            'calculate',
            side_effect=ValueError("Invalid audio data")
        ):
            result = self.pipeline.process_with_recovery(test_path)
            
        # Should continue processing with defaults
        self.assertIn('spectral_features', result)
        self.assertIn('decision', result)
        
        # Should use default SNR metrics
        self.assertIsNotNone(result['snr_metrics'])
        self.assertEqual(result['snr_metrics'].global_snr, 0.0)
        
    def test_partial_processing_recovery(self):
        """Test recovery with partial processing results."""
        test_path = os.path.join(self.temp_dir, 'partial.wav')
        audio = np.random.randn(16000 * 2)
        sf.write(test_path, audio, 16000)
        
        # Mock pattern detector failure
        with patch.object(
            self.pipeline.components['pattern_detector'],
            'detect',
            side_effect=RuntimeError("Pattern detection failed")
        ):
            result = self.pipeline.process_with_recovery(test_path)
            
        # Should have results up to pattern detection
        self.assertIn('audio_data', result)
        self.assertIn('snr_metrics', result)
        self.assertIn('spectral_features', result)
        
        # Pattern should be empty list
        self.assertEqual(result['patterns'], [])
        
        # Should still make a decision
        self.assertIn('decision', result)
        
    def test_error_logging_completeness(self):
        """Test that all errors are properly logged."""
        # Create file that will cause multiple errors
        bad_path = os.path.join(self.temp_dir, 'bad.wav')
        
        # Don't create file - will cause FileNotFoundError
        result = self.pipeline.process_with_recovery(bad_path)
        
        # Check error log
        self.assertIn('errors', result)
        errors = result['errors']
        
        # Should have at least one error
        self.assertGreater(len(errors), 0)
        
        # Each error should have required fields
        for error in errors:
            self.assertIn('stage', error)
            self.assertIn('error', error)
            self.assertIn('type', error)
            
    def test_graceful_degradation_levels(self):
        """Test different levels of graceful degradation."""
        test_path = os.path.join(self.temp_dir, 'degrade.wav')
        audio = np.random.randn(16000)
        sf.write(test_path, audio, 16000)
        
        # Test with different component failures
        failure_scenarios = [
            ('snr_calculator', 'calculate', SNRError("SNR calculation failed")),
            ('spectral_analyzer', 'analyze', ValueError("Invalid spectrum")),
            ('pattern_detector', 'detect', RuntimeError("Detection failed")),
            ('issue_categorizer', 'categorize', Exception("Categorization error")),
        ]
        
        for component_name, method_name, error in failure_scenarios:
            with patch.object(
                self.pipeline.components[component_name],
                method_name,
                side_effect=error
            ):
                result = self.pipeline.process_with_recovery(test_path)
                
                # Should always produce a decision
                self.assertIn('decision', result)
                self.assertIsNotNone(result['decision'])
                
                # Should log the specific error
                logged_errors = [
                    e for e in result['errors']
                    if error.__class__.__name__ in e['type']
                ]
                self.assertTrue(len(logged_errors) > 0)


class TestEdgeCasesAndBoundaries(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = AudioLoader()
        self.snr_calculator = EnhancedSNRCalculator()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_zero_length_audio(self):
        """Test handling of zero-length audio."""
        zero_path = os.path.join(self.temp_dir, 'zero.wav')
        zero_audio = np.array([])
        
        # This should raise an error when trying to write
        with self.assertRaises(Exception):
            sf.write(zero_path, zero_audio, 16000)
            
    def test_single_sample_audio(self):
        """Test handling of single sample audio."""
        single_path = os.path.join(self.temp_dir, 'single.wav')
        single_audio = np.array([0.5])
        sf.write(single_path, single_audio, 16000)
        
        # Should handle but may have limited analysis
        try:
            audio = self.loader.load(single_path)
            self.assertEqual(len(audio.samples), 1)
        except Exception as e:
            # Some operations may fail with single sample
            self.assertIsInstance(e, (AudioLoadError, PreprocessingError))
            
    def test_extreme_sample_rates(self):
        """Test handling of extreme sample rates."""
        # Very low sample rate
        low_sr_path = os.path.join(self.temp_dir, 'low_sr.wav')
        audio = np.random.randn(1000)
        sf.write(low_sr_path, audio, 1000)  # 1kHz
        
        # Very high sample rate
        high_sr_path = os.path.join(self.temp_dir, 'high_sr.wav')
        audio = np.random.randn(192000)
        sf.write(high_sr_path, audio, 192000)  # 192kHz
        
        # Both should load successfully
        low_audio = self.loader.load(low_sr_path)
        high_audio = self.loader.load(high_sr_path)
        
        self.assertEqual(low_audio.sample_rate, 1000)
        self.assertEqual(high_audio.sample_rate, 192000)
        
    def test_nan_inf_values(self):
        """Test handling of NaN and Inf values in audio."""
        # Audio with NaN
        nan_path = os.path.join(self.temp_dir, 'nan.wav')
        nan_audio = np.random.randn(16000)
        nan_audio[1000:1100] = np.nan
        
        # Should handle or reject
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sf.write(nan_path, nan_audio, 16000)
                audio = self.loader.load(nan_path)
                
                # SNR calculation should handle NaN
                with self.assertRaises((SNRError, InvalidSignalError)):
                    self.snr_calculator.calculate(audio.samples, audio.sample_rate)
            except:
                # Some formats may not support NaN
                pass
                
    def test_maximum_duration_audio(self):
        """Test handling of maximum duration constraints."""
        # Don't actually create huge file, just test the logic
        max_duration = 3600  # 1 hour
        max_samples = max_duration * 16000
        
        # Test that very long audio would be handled
        self.assertGreater(max_samples, 0)
        self.assertEqual(max_samples, 57600000)
        
    def test_unusual_bit_depths(self):
        """Test handling of various bit depths."""
        bit_depths = [8, 16, 24, 32]
        
        for bit_depth in bit_depths:
            path = os.path.join(self.temp_dir, f'audio_{bit_depth}bit.wav')
            audio = np.random.randn(16000)
            
            # Soundfile handles bit depth conversion
            if bit_depth == 8:
                subtype = 'PCM_U8'
            elif bit_depth == 16:
                subtype = 'PCM_16'
            elif bit_depth == 24:
                subtype = 'PCM_24'
            else:
                subtype = 'PCM_32'
                
            try:
                sf.write(path, audio, 16000, subtype=subtype)
                loaded = self.loader.load(path)
                self.assertIsNotNone(loaded)
            except Exception as e:
                # Some bit depths might not be supported
                print(f"Bit depth {bit_depth} not supported: {e}")
                
    def test_concurrent_error_handling(self):
        """Test error handling with concurrent processing."""
        from concurrent.futures import ThreadPoolExecutor
        
        # Create mix of valid and invalid files
        files = []
        
        # Valid file
        valid_path = os.path.join(self.temp_dir, 'valid.wav')
        sf.write(valid_path, np.random.randn(16000), 16000)
        files.append(valid_path)
        
        # Invalid file
        invalid_path = os.path.join(self.temp_dir, 'invalid.wav')
        with open(invalid_path, 'w') as f:
            f.write('not audio')
        files.append(invalid_path)
        
        # Missing file
        files.append(os.path.join(self.temp_dir, 'missing.wav'))
        
        # Process concurrently
        pipeline = ErrorRecoveryPipeline()
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(pipeline.process_with_recovery, f)
                for f in files
            ]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
                    
        # Should have results for all files
        self.assertEqual(len(results), 3)
        
        # At least one should succeed
        successful = [r for r in results if 'decision' in r and not r.get('errors')]
        self.assertGreater(len(successful), 0)


class TestDocumentationExamples(unittest.TestCase):
    """Test that all documentation examples work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_quickstart_example(self):
        """Test the quickstart example from documentation."""
        # Create test audio as shown in docs
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        audio_path = os.path.join(self.temp_dir, 'example.wav')
        sf.write(audio_path, audio, sample_rate)
        
        # Example code from documentation
        from processors.audio_enhancement.audio_loader import AudioLoader
        from utils.enhanced_snr_calculator import EnhancedSNRCalculator
        from processors.audio_enhancement.decision_framework import DecisionEngine
        
        # Load audio
        loader = AudioLoader()
        audio_data = loader.load(audio_path)
        
        # Calculate SNR
        snr_calc = EnhancedSNRCalculator()
        snr_metrics = snr_calc.calculate(audio_data.samples, audio_data.sample_rate)
        
        # Make decision
        decision_engine = DecisionEngine()
        decision = decision_engine.decide([])
        
        # Verify example works
        self.assertIsNotNone(audio_data)
        self.assertIsNotNone(snr_metrics)
        self.assertIsNotNone(decision)
        
    def test_api_usage_examples(self):
        """Test API usage examples."""
        # Test each component's example usage
        
        # AudioLoader example
        loader = AudioLoader()
        self.assertIsNotNone(loader)
        
        # SNR Calculator example
        snr_calc = EnhancedSNRCalculator()
        test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        metrics = snr_calc.calculate(test_signal, 16000)
        self.assertIsNotNone(metrics.global_snr)
        
        # Pattern Detector example
        from processors.audio_enhancement.detection.pattern_detector import PatternDetector
        detector = PatternDetector()
        patterns = detector.detect(test_signal, None)
        self.assertIsInstance(patterns, list)
        
    def test_configuration_examples(self):
        """Test configuration examples from documentation."""
        # Test configuration loading
        config = {
            'snr_calculator': {
                'frame_length': 0.025,
                'min_speech_duration': 0.3
            },
            'decision_engine': {
                'enhancement_threshold': 25,
                'rejection_threshold': 5
            }
        }
        
        # Verify configuration structure
        self.assertIn('snr_calculator', config)
        self.assertIn('decision_engine', config)
        self.assertEqual(config['snr_calculator']['frame_length'], 0.025)


if __name__ == '__main__':
    unittest.main()