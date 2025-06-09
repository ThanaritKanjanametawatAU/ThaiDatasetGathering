# Thai Audio Dataset Collection - Project Overview

## Project Mission

To create a high-quality, standardized Thai language audio dataset by collecting, processing, and enhancing audio data from multiple sources, making it readily available for speech recognition, voice cloning, and other audio AI applications.

## Business Value

- **Enables Thai Language AI Development**: Provides researchers and developers with a comprehensive, cleaned, and standardized Thai audio dataset
- **Improves Voice Technology**: High-quality audio (35dB SNR) suitable for training voice cloning and TTS systems
- **Reduces Development Time**: Pre-processed and standardized data eliminates repetitive preprocessing work
- **Open Source Contribution**: Freely available on HuggingFace for the global AI community

## Key Stakeholders

- **Primary Users**: AI researchers, speech recognition developers, voice cloning engineers
- **Data Sources**: GigaSpeech2, ProcessedVoiceTH, Mozilla Common Voice, VISTEC Common Voice
- **Platform**: HuggingFace (hosting as Thanarit/Thai-Voice dataset)
- **Development Team**: Led by Thanarit Kanjanametawat

## Technical Stack

### Core Technologies
- **Language**: Python 3.10+
- **ML Framework**: PyTorch, Transformers
- **Audio Processing**: librosa, soundfile, torchaudio
- **Speaker Analysis**: pyannote.audio, SpeechBrain
- **Data Management**: HuggingFace datasets, pandas
- **Enhancement**: denoiser, noisereduce, audio-separator

### Infrastructure
- **Compute**: GPU-accelerated processing (CUDA support)
- **Storage**: Streaming mode to handle large datasets without full download
- **Distribution**: HuggingFace Hub for dataset hosting

## Current Status (June 2025)

### Completed Features
- ✅ Multi-source data collection pipeline
- ✅ Streaming mode for efficient processing
- ✅ Speaker identification and clustering
- ✅ Audio enhancement with 35dB SNR target
- ✅ Robust secondary speaker removal
- ✅ Speech-to-text transcription
- ✅ Comprehensive test suite (100+ tests)

### Recent Achievements
- Implemented Test-Driven Development (TDD) for secondary speaker removal
- Fixed authentication issues with PyAnnote models
- Enhanced code quality following Power of 10 rules
- Reduced code duplication by ~600 lines

### Active Development
- Processing pipeline optimization
- Real-time streaming support improvements
- Enhanced speaker diarization capabilities

## Success Metrics

1. **Audio Quality**: 90% of samples achieve 35dB SNR or better
2. **Dataset Size**: Target 10M samples (currently configured for 100K)
3. **Processing Speed**: ~100 samples/minute with all features enabled
4. **Speaker Accuracy**: Correct clustering of known speaker patterns (S1-S8,S10 same; S9 different)
5. **Test Coverage**: Maintaining comprehensive test suite with all tests passing

## Development Workflow

1. **Local Development**: Use `main.sh` for testing with sample data
2. **Feature Development**: Follow TDD methodology
3. **Quality Assurance**: Run full test suite before commits
4. **Documentation**: Update relevant docs for significant changes
5. **Deployment**: Push to HuggingFace dataset repository

## Key Commands

```bash
# Run with all features (sample mode)
./main.sh

# Run specific dataset
python main.py GigaSpeech2 --sample --sample-size 100

# Run tests
pytest tests/

# Check audio enhancement
python processors/audio_enhancement/core.py
```

## Important Guidelines

### From CLAUDE.md
- Always preserve existing features when implementing new ones
- Use descriptive variable names
- Prefer editing existing files over creating new ones
- Run tests before pushing to GitHub
- Use `huggingface-cli` for HuggingFace operations
- Speaker clustering verification: S1-S8,S10 should have same ID, S9 different

### Security & Best Practices
- HF token stored in `.env` file (not `.hf_token`)
- No system-breaking commands (sudo, apt)
- Follow Power of 10 coding rules
- Maintain ML pipeline best practices

## Contact & Resources

- **GitHub**: ThanaritKanjanametawatAU
- **Dataset**: https://huggingface.co/datasets/Thanarit/Thai-Voice
- **Documentation**: See `/docs` directory for detailed technical docs