# Checkpoint System Documentation

## Overview

The Thai Audio Dataset Collection project includes a robust checkpoint and resume system that allows interrupted processing to be resumed without re-processing already completed samples. This is especially important for large datasets that can take hours to process.

## Features

### Unified Checkpoint Format (v2.0)

The checkpoint system now uses a unified format that works for both regular and streaming modes:

```json
{
  "mode": "unified",
  "samples_processed": 1000,
  "current_split": "train",
  "split_index": 500,
  "shard_num": 0,
  "last_sample_id": "S1000",
  "processed_ids": ["S998", "S999", "S1000"],
  "dataset_specific": {},
  "timestamp": 1234567890,
  "processor": "GigaSpeech2Processor"
}
```

### Key Improvements

1. **Automatic Periodic Checkpointing**: Checkpoints are automatically saved every 1000 samples (configurable) during processing
2. **Split Tracking**: Proper tracking of which split is being processed for datasets with multiple splits
3. **Backward Compatibility**: Seamlessly loads and converts old checkpoint formats to the new unified format
4. **Error Recovery**: Checkpoints are saved when errors occur, allowing recovery from the last successful state

## Usage

### Basic Resume from Checkpoint

```bash
# Process with automatic checkpoint saving
python main.py --fresh --all

# If interrupted, resume from latest checkpoint
python main.py --fresh --all --resume

# Resume from specific checkpoint file
python main.py --fresh --all --checkpoint ./checkpoints/GigaSpeech2_1234567890.json
```

### Streaming Mode

```bash
# Stream processing with checkpoint support
python main.py --fresh --all --streaming

# Resume streaming from checkpoint
python main.py --fresh --all --streaming --resume
```

### Configuring Checkpoint Interval

```python
# In process_all_splits method
for sample in processor.process_all_splits(
    checkpoint_interval=500  # Save checkpoint every 500 samples
):
    # Process sample
```

## Checkpoint Files

Checkpoints are saved in two locations:

1. **Unified checkpoint**: `{processor_name}_unified_checkpoint.json` - Always contains the latest state
2. **Timestamped checkpoint**: `{processor_name}_{timestamp}.json` - Historical checkpoints for recovery

## Implementation Details

### Checkpoint Saving

The system saves checkpoints in multiple scenarios:

1. **Periodic**: Every N samples (default: 1000)
2. **On Completion**: When processing completes successfully
3. **On Error**: When an error occurs during processing
4. **Manual**: Via `save_unified_checkpoint()` method

### Checkpoint Loading

When resuming:

1. Loads the checkpoint file
2. Converts old formats to unified format if needed
3. Skips to the correct split and position
4. Continues processing from where it left off

### Split Handling

For datasets with multiple splits (train, test, val):

- Tracks current split name
- Tracks position within current split
- Properly skips completed splits when resuming
- Handles split transitions correctly

## Example: Custom Processor Implementation

```python
class MyProcessor(BaseProcessor):
    def process_all_splits(self, checkpoint=None, sample_mode=False, sample_size=5):
        # The base class now handles all checkpoint logic
        # Just call the parent method
        yield from super().process_all_splits(checkpoint, sample_mode, sample_size)
    
    def _process_single_split(self, split, checkpoint=None, sample_mode=False, sample_size=5):
        # Implement your split processing logic
        for i, sample in enumerate(self.load_split_data(split)):
            # Process sample
            yield self.convert_to_standard_format(sample)
```

## Troubleshooting

### Common Issues

1. **"shard_num KeyError"**: This was fixed in v2.0. Update to the latest code.
2. **Duplicate samples after resume**: Ensure you're using the unified checkpoint system
3. **Resume starts from beginning**: Check that checkpoint file exists and is valid

### Debugging

Enable verbose logging to see checkpoint operations:

```bash
python main.py --fresh --all --resume --verbose
```

Check checkpoint contents:

```bash
cat ./checkpoints/*_unified_checkpoint.json | jq .
```

## Best Practices

1. **Don't modify checkpoint files manually** - The system expects specific formats
2. **Keep checkpoint directory clean** - Old checkpoints can be deleted after successful completion
3. **Test resume capability** - Use the included test script: `python test_checkpoint_resume.py`
4. **Monitor progress** - Checkpoints include progress information for monitoring

## Future Improvements

- Checkpoint compression for large processed_ids lists
- Cloud checkpoint storage option
- Checkpoint integrity verification
- Automatic checkpoint cleanup policies