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

# Parse command line arguments
MODE="--fresh"  # Default to fresh mode
RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            MODE="--fresh"  # Still use fresh mode but add resume flag
            RESUME_FLAG="--resume"
            shift
            ;;
        --append)
            MODE="--append"
            shift
            ;;
        *)
            error "Unknown option: $1"
            echo "Usage: $0 [--resume|--append]"
            echo "  --resume  Resume from checkpoint (continue interrupted processing)"
            echo "  --append  Append new samples to existing dataset"
            exit 1
            ;;
    esac
done

# Configuration
export HF_HUB_ENABLE_HF_TRANSFER=1
CONDA_ENV="thaidataset"
SAMPLES_PER_DATASET=100
HF_REPO="Thanarit/Thai-Voice-10000000"  # Repository for 1M samples

# Feature flags (all enabled)
ENABLE_SPEAKER_ID="--enable-speaker-id"  # Re-enabled after fixing clustering
ENABLE_STT="--enable-stt"
ENABLE_AUDIO_ENHANCEMENT="--enable-audio-enhancement"
ENABLE_STREAMING="--streaming"
ENABLE_DASHBOARD=""  # Disabled dashboard for now

# Enhancement settings
ENHANCEMENT_LEVEL="ultra_aggressive"  # Balanced noise removal without secondary speaker removal
ENHANCEMENT_GPU="--enhancement-gpu"  # Using GPU for faster processing (remove if no GPU)

# Speaker identification settings - ADJUST THESE TO CONTROL SPEAKER SEGMENTATION
# Lower threshold = more grouping, Higher threshold = more separation
SPEAKER_THRESHOLD="0.9"         # Similarity threshold (0.5-0.95, default: 0.7)
                                # 0.5-0.6: Very lenient (groups many speakers)
                                # 0.7-0.8: Moderate (balanced)
                                # 0.85-0.95: Strict (more separation)

SPEAKER_BATCH_SIZE="10000000"         # Batch size (<50 uses Agglomerative, >=50 uses HDBSCAN)
SPEAKER_MIN_CLUSTER_SIZE="15"    # Min cluster size for HDBSCAN (only used if batch>=50)
SPEAKER_MIN_SAMPLES="5"         # Min samples for HDBSCAN core points
SPEAKER_EPSILON="0.2"           # HDBSCAN epsilon (smaller = tighter clusters)

# Datasets to test (fresh mode with GigaSpeech2 and ProcessedVoiceTH)
# DATASETS="GigaSpeech2 MozillaCommonVoice"
DATASETS="GigaSpeech2"
UPLOAD_BATCH_SIZE=100000  # Increased for better performance with 1M samples


# Additional audio preprocessing settings for aggressive enhancement
ENABLE_VOLUME_NORM=""  # Keep volume normalization enabled (default)
TARGET_DB="-20.0"  # Target volume level in dB
SAMPLE_RATE="16000"  # Target sample rate (16kHz is standard for speech)

# Determine mode description
if [ -n "$RESUME_FLAG" ]; then
    MODE_DESC="RESUME (continuing from checkpoint)"
elif [ "$MODE" = "--append" ]; then
    MODE_DESC="APPEND (adding to existing dataset)"
else
    MODE_DESC="FRESH (creating new dataset)"
fi

log "Starting Thai Audio Dataset Collection"
log "Configuration:"
log "  - Mode: $MODE_DESC"
log "  - Samples per dataset: $SAMPLES_PER_DATASET"
log "  - Datasets: $DATASETS"
log "  - Target repository: $HF_REPO"
log "  - Features: Speaker ID, STT, Audio Enhancement, Streaming"
log "  - Enhancement level: $ENHANCEMENT_LEVEL (maximum noise removal)"
log "  - Speaker identification:"
log "    - Threshold: $SPEAKER_THRESHOLD (higher = more speaker separation)"
log "    - Batch size: $SPEAKER_BATCH_SIZE (algorithm: $([ $SPEAKER_BATCH_SIZE -lt 50 ] && echo \"Agglomerative\" || echo \"HDBSCAN\"))"
log "    - Min cluster size: $SPEAKER_MIN_CLUSTER_SIZE"
log "    - Epsilon: $SPEAKER_EPSILON"
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

# Check if dataset exists
log "Checking if dataset exists..."
python -c "
from utils.huggingface import read_hf_token
from huggingface_hub import HfApi

token = read_hf_token('.hf_token')
if token:
    api = HfApi()
    try:
        dataset_info = api.dataset_info('$HF_REPO', token=token)
        print(f'  - Dataset exists: {dataset_info.id}')
    except Exception:
        print(f'  - Dataset does not exist yet (expected for fresh mode)')
" || warning "Could not check dataset existence"

# Clean up any previous data (only in fresh mode without resume)
if [ "$MODE" = "--fresh" ] && [ -z "$RESUME_FLAG" ]; then
    log "Fresh mode: Cleaning up ALL previous checkpoints and cache..."
    # Remove ALL checkpoint files to ensure truly fresh start
    rm -rf checkpoints/*_checkpoint.json || true
    rm -rf checkpoints/*_unified_checkpoint.json || true
    rm -rf checkpoints/speaker_model.json || true
    rm -rf cache/* || true
    rm -rf enhancement_metrics || true
    log "Removed all checkpoints for fresh start"
elif [ -n "$RESUME_FLAG" ]; then
    log "Resume mode: Preserving existing checkpoints and cache"
    # Show existing checkpoints
    if ls checkpoints/*checkpoint*.json 1> /dev/null 2>&1; then
        log "Found existing checkpoints:"
        ls -la checkpoints/*checkpoint*.json | while read line; do
            echo "  - $line"
        done
    fi
fi

# Run the main script with all features enabled
log "Running main.py with all features enabled..."

CMD="python main.py \
    $MODE \
    $DATASETS \
    $RESUME_FLAG \
    --sample \
    --sample-size $SAMPLES_PER_DATASET \
    --hf-repo $HF_REPO \
    $ENABLE_STREAMING \
    $ENABLE_SPEAKER_ID \
    --speaker-batch-size $SPEAKER_BATCH_SIZE \
    --speaker-threshold $SPEAKER_THRESHOLD \
    --speaker-min-cluster-size $SPEAKER_MIN_CLUSTER_SIZE \
    --speaker-min-samples $SPEAKER_MIN_SAMPLES \
    --speaker-epsilon $SPEAKER_EPSILON \
    $ENABLE_STT \
    --stt-batch-size 16 \
    $ENABLE_AUDIO_ENHANCEMENT \
    --enhancement-level $ENHANCEMENT_LEVEL \
    --enhancement-batch-size 10 \
    $ENHANCEMENT_GPU \
    --sample-rate $SAMPLE_RATE \
    --target-db $TARGET_DB \
    --upload-batch-size $UPLOAD_BATCH_SIZE \
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