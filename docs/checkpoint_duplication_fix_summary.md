# Checkpoint Duplication Fix Summary

## Issue
The HuggingFace dataset at https://huggingface.co/datasets/Thanarit/Thai-Voice-Test-1000000 had duplicate IDs. Instead of the expected 20 samples, there were 240 rows (12x duplication).

## Root Cause
The `StreamingUploader` class wasn't tracking which sample IDs had already been uploaded, causing the same samples to be uploaded multiple times during retries or resumes.

## Solution Implemented

### 1. Added ID Tracking to StreamingUploader
- Added `uploaded_ids` set to track which sample IDs have been uploaded
- Added `checkpoint_path` parameter to persist tracking across runs
- Implemented checkpoint save/load functionality with atomic writes

### 2. Modified Upload Logic
- `add_sample()` method checks for duplicates before adding to batch
- `upload_batch()` method filters out already uploaded samples
- Checkpoint is saved after each successful upload

### 3. Updated main.py
- Added checkpoint path generation for the uploader
- Path is based on the repository name to avoid conflicts

## Key Changes

### utils/streaming.py
```python
class StreamingUploader:
    def __init__(self, ..., checkpoint_path: Optional[str] = None):
        # Added checkpoint tracking
        self.uploaded_ids = set()
        self.checkpoint_path = checkpoint_path
        
    def upload_batch(self, samples):
        # Filter out duplicates
        filtered_samples = []
        for sample in samples:
            sample_id = sample.get('ID', str(self.total_samples))
            if sample_id not in self.uploaded_ids:
                filtered_samples.append(sample)
                self.uploaded_ids.add(sample_id)
```

### main.py
```python
# Create checkpoint path for uploader
uploader_checkpoint_path = os.path.join("checkpoints", 
    f"{target_repo.replace('/', '_')}_upload_checkpoint.json")

uploader = StreamingUploader(
    repo_id=target_repo,
    token=token,
    private=args.private,
    append_mode=append_mode,
    checkpoint_path=uploader_checkpoint_path
)
```

## Verification
Created comprehensive test suite (test_checkpoint_duplication_fix.py) that verifies:
1. Each sample is uploaded exactly once
2. Checkpoints properly track uploaded IDs
3. Resume functionality doesn't re-upload existing samples
4. Different splits maintain separate checkpoints
5. Atomic checkpoint writes prevent corruption

All tests pass successfully.

## Impact
- Prevents duplicate uploads in all scenarios
- Supports proper resume functionality
- Maintains backward compatibility
- No performance impact (ID lookup is O(1) with set)

## Usage
The fix is automatic - no changes required to existing code. The uploader will:
1. Create a checkpoint file in the checkpoints directory
2. Track all uploaded sample IDs
3. Skip any samples that have already been uploaded
4. Persist state across runs for reliable resume functionality