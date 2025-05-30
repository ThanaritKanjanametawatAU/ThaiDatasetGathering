#!/bin/bash
# Main test script for Thai Audio Dataset Collection with all features enabled

# Script configuration
set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
CONDA_ENV="thaidataset"
SAMPLES_PER_DATASET=100
HF_REPO="Thanarit/Thai-Voice-Test2"  # Test repository for verification

# Feature flags (all enabled)
ENABLE_SPEAKER_ID="--enable-speaker-id"  # Re-enabled after fixing clustering
ENABLE_STT="--enable-stt"
ENABLE_AUDIO_ENHANCEMENT="--enable-audio-enhancement"
ENABLE_STREAMING="--streaming"
ENABLE_DASHBOARD=""  # Disabled dashboard for now

# Enhancement settings
ENHANCEMENT_LEVEL="ultra_aggressive"  # Changed to ultra_aggressive for MAXIMUM noise removal
ENHANCEMENT_GPU="--enhancement-gpu"  # Using GPU for faster processing (remove if no GPU)

# Datasets to test (2 datasets)
DATASETS="GigaSpeech2 ProcessedVoiceTH"

# Additional audio preprocessing settings for aggressive enhancement
ENABLE_VOLUME_NORM=""  # Keep volume normalization enabled (default)
TARGET_DB="-20.0"  # Target volume level in dB
SAMPLE_RATE="16000"  # Target sample rate (16kHz is standard for speech)

log "Starting Thai Audio Dataset Collection test with all features enabled"
log "Configuration:"
log "  - Samples per dataset: $SAMPLES_PER_DATASET"
log "  - Datasets: $DATASETS"
log "  - Target repository: $HF_REPO"
log "  - Features: Speaker ID, STT, Audio Enhancement, Streaming"
log "  - Enhancement level: $ENHANCEMENT_LEVEL (maximum noise removal)"
log "  - Audio preprocessing:"
log "    - Sample rate: $SAMPLE_RATE Hz"
log "    - Target volume: $TARGET_DB dB"
log "    - Volume normalization: Enabled"
log "    - GPU acceleration: $([ -n \"$ENHANCEMENT_GPU\" ] && echo \"Enabled\" || echo \"Disabled\")"

# Activate conda environment
log "Activating conda environment: $CONDA_ENV"
eval "$(/home/money/anaconda3/bin/conda shell.bash hook)"
conda activate $CONDA_ENV || {
    error "Failed to activate conda environment: $CONDA_ENV"
    exit 1
}

# Check Python version
log "Python version: $(python --version)"

# Clean up any previous test data
log "Cleaning up previous test data..."
rm -rf checkpoints/*test* || true
rm -rf cache/*test* || true
rm -rf enhancement_metrics || true

# Run the main script with all features enabled
log "Running main.py with all features enabled..."

CMD="python main.py \
    --append \
    MozillaCommonVoice \
    --sample \
    --sample-size $SAMPLES_PER_DATASET \
    --hf-repo $HF_REPO \
    $ENABLE_STREAMING \
    $ENABLE_SPEAKER_ID \
    --speaker-batch-size 50 \
    $ENABLE_STT \
    --stt-batch-size 16 \
    $ENABLE_AUDIO_ENHANCEMENT \
    --enhancement-level $ENHANCEMENT_LEVEL \
    --enhancement-batch-size 32 \
    $ENHANCEMENT_GPU \
    --sample-rate $SAMPLE_RATE \
    --target-db $TARGET_DB \
    --verbose"

log "Executing command:"
echo "$CMD"
echo ""

# Run the command and capture output
if $CMD; then
    log "Successfully completed dataset processing!"
    
    # Show enhancement metrics if available
    if [ -d "enhancement_metrics" ]; then
        log "Enhancement metrics saved in: enhancement_metrics/"
        if [ -f "enhancement_metrics/summary.json" ]; then
            log "Enhancement summary:"
            python -c "import json; print(json.dumps(json.load(open('enhancement_metrics/summary.json')), indent=2))" || true
        fi
    fi
    
    # Show checkpoint info
    log "Checkpoints created:"
    ls -la checkpoints/*checkpoint*.json 2>/dev/null || warning "No checkpoints found"
    
    # Show speaker model info
    if [ -f "checkpoints/speaker_model.json" ]; then
        log "Speaker model saved successfully"
        python -c "import json; data = json.load(open('checkpoints/speaker_model.json')); print(f'Total speakers identified: {data.get(\"speaker_counter\", 0)}')" || true
    fi
    
    log "Test completed successfully!"
    log "Please check the HuggingFace repository at: https://huggingface.co/datasets/$HF_REPO"
    
    # Print dataset statistics
    log "Dataset statistics:"
    python -c "
from utils.huggingface import read_hf_token, get_last_id
from huggingface_hub import HfApi
import os

token = read_hf_token('.hf_token')
if token:
    api = HfApi()
    try:
        dataset_info = api.dataset_info('$HF_REPO', token=token)
        print(f'  - Dataset size: {dataset_info.size_on_disk_str if hasattr(dataset_info, \"size_on_disk_str\") else \"Unknown\"}')
        last_id = get_last_id('$HF_REPO')
        if last_id:
            print(f'  - Total samples: {last_id}')
    except Exception as e:
        print(f'  - Could not fetch dataset info: {str(e)}')
" || true
    
else
    error "Dataset processing failed!"
    exit 1
fi

log "Script execution completed"