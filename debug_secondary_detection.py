"""Debug secondary speaker detection"""

import numpy as np
import matplotlib.pyplot as plt
from tests.test_secondary_speaker_removal_tdd import TestSecondaryRemoval
from processors.audio_enhancement.secondary_removal import SecondaryRemover

# Create test instance
test = TestSecondaryRemoval()
test.setUpClass()

# Create test audio
audio = test.create_test_audio(3.3, [
    (0.0, 3.0, 1, 200),    # Primary speaker
    (3.0, 3.3, 2, 400)     # Secondary speaker at end
])

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {test.sample_rate}")

# Test detection
remover = SecondaryRemover()
result = remover.detect_secondary_speakers(audio, test.sample_rate)

print(f"\nDetection result:")
print(f"  has_secondary_at_end: {result.has_secondary_at_end}")
print(f"  secondary_start_time: {result.secondary_start_time}")
print(f"  num_speakers: {result.num_speakers}")
print(f"  confidence: {result.confidence}")

# Visualize
plt.figure(figsize=(12, 6))
time = np.arange(len(audio)) / test.sample_rate
plt.plot(time, audio)
plt.axvline(x=3.0, color='r', linestyle='--', label='Expected secondary start')
if result.secondary_start_time > 0:
    plt.axvline(x=result.secondary_start_time, color='g', linestyle='--', label='Detected secondary start')
plt.title('Test Audio with Speaker Changes')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('test_audio_output/debug_secondary_detection.png')
plt.close()

# Test VAD
from processors.audio_enhancement.secondary_removal import VoiceActivityDetector
vad = VoiceActivityDetector()
segments = vad.detect_speech_segments(audio, test.sample_rate)
print(f"\nVAD segments: {len(segments)}")
for i, seg in enumerate(segments):
    print(f"  Segment {i}: {seg.start_time:.2f}s - {seg.end_time:.2f}s (energy: {seg.energy:.4f})")

# Test energy analyzer
from processors.audio_enhancement.secondary_removal import EnergyAnalyzer
analyzer = EnergyAnalyzer()
changes = analyzer.detect_energy_changes(audio, test.sample_rate)
print(f"\nEnergy changes: {len(changes)}")
for i, change in enumerate(changes):
    print(f"  Change {i}: {change.time:.2f}s (ratio: {change.energy_ratio:.2f})")

# Test end detector
from processors.audio_enhancement.secondary_removal import SmartEndDetector
detector = SmartEndDetector()
end_analysis = detector.analyze_end(audio, test.sample_rate)
print(f"\nEnd analysis:")
print(f"  has_secondary_speaker: {end_analysis.has_secondary_speaker}")
print(f"  primary_end_time: {end_analysis.primary_end_time}")
print(f"  secondary_start_time: {end_analysis.secondary_start_time}")
print(f"  confidence: {end_analysis.confidence}")

# Check energy profile of last second
last_second = audio[-test.sample_rate:]
print(f"\nLast second analysis:")
print(f"  Max amplitude: {np.max(np.abs(last_second)):.4f}")
print(f"  RMS energy: {np.sqrt(np.mean(last_second**2)):.4f}")