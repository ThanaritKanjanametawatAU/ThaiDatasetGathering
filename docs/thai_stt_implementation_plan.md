# Thai Speech-to-Text Implementation Plan for Thai Dataset Gathering

## Executive Summary

This document outlines a comprehensive plan for implementing Thai Speech-to-Text (STT) transcription capabilities in the Thai Dataset Gathering project. Based on current research (January 2025), we recommend a multi-model approach with primary focus on the airesearch/wav2vec2-large-xlsr-53-th model for its excellent Thai-specific performance.

## Current Best Thai STT Models (January 2025)

### 1. **airesearch/wav2vec2-large-xlsr-53-th** (RECOMMENDED - Primary)
- **Developer**: VISTEC/AIResearch.in.th & PyThaiNLP
- **Performance**: 
  - WER: 8.15% (deepcut tokenizer), 13.63% (PyThaiNLP tokenizer)
  - CER: 2.81%
  - SER: 1.235%
- **Advantages**:
  - Specifically fine-tuned for Thai language
  - Open source (CC-BY-SA 4.0)
  - Excellent performance on Thai Common Voice dataset
  - Free to use
  - Easy HuggingFace integration
- **Limitations**:
  - Requires proper Thai tokenization
  - May need GPU for optimal performance

### 2. **OpenAI Whisper** (RECOMMENDED - Secondary)
- **Latest Version**: gpt-4o-transcribe (March 2025)
- **Performance**: WER ~8-10% for general multilingual
- **Advantages**:
  - Strong multilingual support including Thai
  - Multiple model sizes (tiny to large)
  - Can run locally or via API
  - Robust to noise and accents
- **Limitations**:
  - Not specifically optimized for Thai
  - API usage has costs
  - Larger models require significant compute

### 3. **Meta's Massively Multilingual Speech (MMS)**
- **Coverage**: 1,100+ languages including Thai
- **Performance**: Claims 50% lower WER than Whisper
- **Advantages**:
  - Open source
  - Extensive language coverage
  - Good for low-resource scenarios
- **Limitations**:
  - Less Thai-specific optimization
  - Newer, less community support

### 4. **Google Cloud Speech-to-Text V2**
- **Performance**: Commercial-grade accuracy
- **Advantages**:
  - Reliable cloud infrastructure
  - Good Thai language support
  - Real-time streaming capabilities
- **Limitations**:
  - Paid service
  - Requires internet connectivity
  - Data privacy considerations

## Implementation Architecture

### Phase 1: Core STT Integration

```python
# Proposed directory structure
processors/
├── __init__.py
├── base_processor.py
├── stt/
│   ├── __init__.py
│   ├── base_stt.py
│   ├── wav2vec2_thai.py
│   ├── whisper_stt.py
│   └── multi_model_stt.py
└── ...existing processors...

utils/
├── __init__.py
├── audio.py
├── thai_text.py  # New: Thai text processing utilities
└── ...existing utils...
```

### Phase 2: Implementation Steps

#### Step 1: Create Base STT Interface
```python
# processors/stt/base_stt.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

class BaseSTT(ABC):
    """Base class for Speech-to-Text processors"""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Transcribe audio to text"""
        pass
    
    @abstractmethod
    def batch_transcribe(self, audio_batch: List[np.ndarray], sample_rates: List[int]) -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio files"""
        pass
    
    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Preprocess audio for transcription"""
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)
            sample_rate = 16000
        return audio, sample_rate
```

#### Step 2: Implement Thai Wav2Vec2 STT
```python
# processors/stt/wav2vec2_thai.py
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pythainlp.tokenize import word_tokenize
import deepcut

class ThaiWav2Vec2STT(BaseSTT):
    def __init__(self, model_name="airesearch/wav2vec2-large-xlsr-53-th", 
                 tokenizer="deepcut", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        # Preprocess audio
        audio, sample_rate = self.preprocess_audio(audio, sample_rate)
        
        # Process with model
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Post-process with Thai tokenizer
        if self.tokenizer == "deepcut":
            tokens = deepcut.tokenize(transcription)
        else:
            tokens = word_tokenize(transcription, engine=self.tokenizer)
            
        return {
            "text": transcription,
            "tokens": tokens,
            "confidence": float(torch.max(torch.softmax(logits, dim=-1))),
            "model": "wav2vec2-thai"
        }
```

#### Step 3: Implement Whisper STT
```python
# processors/stt/whisper_stt.py
import whisper
import torch

class WhisperSTT(BaseSTT):
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        # Whisper expects float32 audio
        audio = audio.astype(np.float32)
        
        # Transcribe
        result = self.model.transcribe(audio, language="th", task="transcribe")
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "th"),
            "confidence": None,  # Whisper doesn't provide confidence scores
            "model": f"whisper-{self.model.dims.n_audio_state}"
        }
```

#### Step 4: Multi-Model Ensemble
```python
# processors/stt/multi_model_stt.py
class MultiModelSTT(BaseSTT):
    def __init__(self, models: List[BaseSTT], voting_strategy="confidence"):
        self.models = models
        self.voting_strategy = voting_strategy
        
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        results = []
        for model in self.models:
            try:
                result = model.transcribe(audio, sample_rate)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model.__class__.__name__} failed: {e}")
                
        # Ensemble voting
        if self.voting_strategy == "confidence":
            # Choose result with highest confidence
            best_result = max(results, key=lambda x: x.get("confidence", 0))
        elif self.voting_strategy == "majority":
            # Implement majority voting
            texts = [r["text"] for r in results]
            best_result = self._majority_vote(texts, results)
            
        best_result["models_used"] = [r["model"] for r in results]
        return best_result
```

### Phase 3: Integration with Dataset Processors

#### Update Base Processor
```python
# processors/base_processor.py
class BaseProcessor(ABC):
    def __init__(self, config: Dict[str, Any], enable_stt: bool = False, stt_model: str = "wav2vec2-thai"):
        self.config = config
        self.enable_stt = enable_stt
        self.stt_processor = None
        
        if enable_stt:
            self._initialize_stt(stt_model)
            
    def _initialize_stt(self, model_name: str):
        if model_name == "wav2vec2-thai":
            from .stt.wav2vec2_thai import ThaiWav2Vec2STT
            self.stt_processor = ThaiWav2Vec2STT()
        elif model_name == "whisper":
            from .stt.whisper_stt import WhisperSTT
            self.stt_processor = WhisperSTT()
        elif model_name == "multi":
            from .stt.multi_model_stt import MultiModelSTT
            from .stt.wav2vec2_thai import ThaiWav2Vec2STT
            from .stt.whisper_stt import WhisperSTT
            self.stt_processor = MultiModelSTT([
                ThaiWav2Vec2STT(),
                WhisperSTT(model_size="base")
            ])
            
    def process_audio_with_stt(self, audio_bytes: bytes, sample_rate: int, existing_transcript: Optional[str] = None):
        """Process audio and optionally generate/verify transcript"""
        if not self.enable_stt:
            return existing_transcript
            
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        stt_result = self.stt_processor.transcribe(audio_array, sample_rate)
        
        # Compare with existing transcript if available
        if existing_transcript:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, existing_transcript, stt_result["text"]).ratio()
            stt_result["similarity_to_original"] = similarity
            stt_result["original_transcript"] = existing_transcript
            
        return stt_result
```

### Phase 4: CLI Integration

```python
# main.py updates
def parse_arguments():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...
    
    # STT arguments
    parser.add_argument("--enable-stt", action="store_true", 
                       help="Enable Speech-to-Text transcription")
    parser.add_argument("--stt-model", choices=["wav2vec2-thai", "whisper", "multi"], 
                       default="wav2vec2-thai",
                       help="STT model to use")
    parser.add_argument("--verify-transcripts", action="store_true",
                       help="Verify existing transcripts with STT")
    parser.add_argument("--stt-batch-size", type=int, default=8,
                       help="Batch size for STT processing")
    return parser.parse_args()
```

## Performance Optimization

### 1. GPU Acceleration
```python
# config.py additions
STT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "num_workers": 4,
    "cache_dir": "./cache/stt_models",
    "use_amp": True,  # Automatic Mixed Precision
}
```

### 2. Batch Processing
```python
def batch_process_stt(audio_files: List[Dict], batch_size: int = 8):
    """Process multiple audio files in batches for efficiency"""
    dataloader = DataLoader(audio_files, batch_size=batch_size, num_workers=4)
    
    results = []
    for batch in tqdm(dataloader, desc="Transcribing audio"):
        batch_results = stt_processor.batch_transcribe(
            batch["audio"], 
            batch["sample_rate"]
        )
        results.extend(batch_results)
    return results
```

### 3. Caching Strategy
```python
# utils/cache.py additions
class STTCache:
    def __init__(self, cache_dir: str = "./cache/stt_results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, audio_bytes: bytes, model_name: str) -> str:
        """Generate cache key from audio content and model"""
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        return f"{model_name}_{audio_hash}"
        
    def get(self, audio_bytes: bytes, model_name: str) -> Optional[Dict]:
        cache_key = self.get_cache_key(audio_bytes, model_name)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
        
    def set(self, audio_bytes: bytes, model_name: str, result: Dict):
        cache_key = self.get_cache_key(audio_bytes, model_name)
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(result, ensure_ascii=False))
```

## Testing Strategy

### 1. Unit Tests
```python
# tests/test_stt.py
import unittest
import numpy as np

class TestSTT(unittest.TestCase):
    def setUp(self):
        self.wav2vec2_stt = ThaiWav2Vec2STT()
        self.test_audio = self._generate_test_audio()
        
    def test_wav2vec2_transcription(self):
        result = self.wav2vec2_stt.transcribe(self.test_audio, 16000)
        self.assertIn("text", result)
        self.assertIn("tokens", result)
        self.assertIsInstance(result["text"], str)
        
    def test_whisper_transcription(self):
        whisper_stt = WhisperSTT(model_size="tiny")
        result = whisper_stt.transcribe(self.test_audio, 16000)
        self.assertIn("text", result)
        self.assertIn("segments", result)
```

### 2. Integration Tests
```python
# tests/test_stt_integration.py
class TestSTTIntegration(unittest.TestCase):
    def test_processor_with_stt(self):
        processor = GigaSpeech2Processor(config, enable_stt=True)
        dataset = processor.process(sample_mode=True, sample_size=5)
        
        for item in dataset:
            if "transcript" in item:
                self.assertIn("stt_result", item)
                self.assertIn("similarity_to_original", item["stt_result"])
```

## Deployment Considerations

### 1. Resource Requirements
- **CPU-only**: Wav2Vec2 (base) ~2GB RAM, Whisper (base) ~1GB RAM
- **GPU**: NVIDIA GPU with 4GB+ VRAM recommended
- **Storage**: ~5GB for models + cache

### 2. Docker Support
```dockerfile
# Dockerfile additions
RUN pip install torch transformers pythainlp deepcut openai-whisper

# Download models during build
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; \
               Wav2Vec2Processor.from_pretrained('airesearch/wav2vec2-large-xlsr-53-th'); \
               Wav2Vec2ForCTC.from_pretrained('airesearch/wav2vec2-large-xlsr-53-th')"
```

### 3. API Endpoint (Optional)
```python
# api/stt_service.py
from fastapi import FastAPI, UploadFile
import soundfile as sf
import io

app = FastAPI()
stt_processor = ThaiWav2Vec2STT()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    contents = await file.read()
    audio, sample_rate = sf.read(io.BytesIO(contents))
    
    result = stt_processor.transcribe(audio, sample_rate)
    return result
```

## Timeline and Milestones

### Phase 1: Foundation (Week 1-2)
- [ ] Implement base STT interface
- [ ] Integrate airesearch/wav2vec2-large-xlsr-53-th
- [ ] Basic testing framework

### Phase 2: Enhancement (Week 3-4)
- [ ] Add Whisper support
- [ ] Implement multi-model ensemble
- [ ] Performance optimization

### Phase 3: Integration (Week 5-6)
- [ ] Update all dataset processors
- [ ] Add CLI commands
- [ ] Comprehensive testing

### Phase 4: Production (Week 7-8)
- [ ] Documentation
- [ ] Docker integration
- [ ] Performance benchmarking
- [ ] Deployment guide

## Conclusion

This implementation plan provides a robust framework for adding Thai STT capabilities to the Thai Dataset Gathering project. The recommended approach using airesearch/wav2vec2-large-xlsr-53-th as the primary model offers excellent Thai-specific performance while maintaining flexibility to use other models like Whisper for comparison or ensemble approaches.

The modular architecture allows for easy extension and testing, while the caching and batch processing optimizations ensure efficient processing of large datasets.