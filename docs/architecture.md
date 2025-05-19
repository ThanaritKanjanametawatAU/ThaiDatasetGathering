# Thai Audio Dataset Collection - Architecture Documentation

## System Overview

The Thai Audio Dataset Collection system is designed to gather Thai audio data from multiple sources and combine them into a single Huggingface dataset with a standardized schema. The system follows a modular architecture that allows for easy addition of new dataset sources.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                             main.py                                  │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐ │
│  │ Command-line│   │  Dataset    │   │  Dataset    │   │ Huggingface│
│  │   Parser    │──▶│ Processing  │──▶│ Combination │──▶│  Upload   │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        processors/                                   │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐ │
│  │ GigaSpeech2 │   │ Processed   │   │   VISTEC    │   │  Mozilla  │ │
│  │  Processor  │   │  Voice TH   │   │ Common Voice│   │Common Voice│
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘ │
│          │                 │                 │                │      │
│          └─────────────────┼─────────────────┼────────────────┘      │
│                            │                 │                       │
│                            ▼                 ▼                       │
│                     ┌─────────────────────────────┐                 │
│                     │      BaseProcessor          │                 │
│                     └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          utils/                                      │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │
│  │    Audio    │   │   Logging   │   │ Huggingface │                │
│  │  Utilities  │   │  Utilities  │   │  Utilities  │                │
│  └─────────────┘   └─────────────┘   └─────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│  Dataset  │     │ Processor │     │ Standardized│    │ Combined  │
│  Source   │────▶│           │────▶│  Samples   │───▶│  Dataset  │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
                        │                                   │
                        ▼                                   ▼
                  ┌───────────┐                      ┌───────────┐
                  │Checkpoints│                      │Huggingface│
                  │           │                      │  Dataset  │
                  └───────────┘                      └───────────┘
```

## Sequence Diagram

### Fresh Creation Mode

```
┌─────┐          ┌─────────┐          ┌───────────┐          ┌───────────┐          ┌───────────┐
│User │          │ main.py │          │ Processor │          │ Huggingface│          │ Filesystem│
└──┬──┘          └────┬────┘          └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
   │                  │                     │                      │                      │
   │ --fresh --all    │                     │                      │                      │
   │─────────────────▶│                     │                      │                      │
   │                  │                     │                      │                      │
   │                  │ process()           │                      │                      │
   │                  │────────────────────▶│                      │                      │
   │                  │                     │                      │                      │
   │                  │                     │ save_checkpoint()    │                      │
   │                  │                     │─────────────────────────────────────────────▶
   │                  │                     │                      │                      │
   │                  │ processed samples   │                      │                      │
   │                  │◀────────────────────│                      │                      │
   │                  │                     │                      │                      │
   │                  │ combine_datasets()  │                      │                      │
   │                  │─────────────────────┘                      │                      │
   │                  │                                            │                      │
   │                  │ upload_dataset()                           │                      │
   │                  │───────────────────────────────────────────▶│                      │
   │                  │                                            │                      │
   │                  │ success                                    │                      │
   │                  │◀───────────────────────────────────────────│                      │
   │                  │                                            │                      │
   │ success          │                                            │                      │
   │◀─────────────────│                                            │                      │
   │                  │                                            │                      │
```

### Append Mode

```
┌─────┐          ┌─────────┐          ┌───────────┐          ┌───────────┐          ┌───────────┐
│User │          │ main.py │          │ Processor │          │ Huggingface│          │ Filesystem│
└──┬──┘          └────┬────┘          └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
   │                  │                     │                      │                      │
   │ --append Dataset │                     │                      │                      │
   │─────────────────▶│                     │                      │                      │
   │                  │                     │                      │                      │
   │                  │ get_last_id()       │                      │                      │
   │                  │─────────────────────────────────────────────▶                     │
   │                  │                     │                      │                      │
   │                  │ last_id             │                      │                      │
   │                  │◀─────────────────────────────────────────────                     │
   │                  │                     │                      │                      │
   │                  │ process()           │                      │                      │
   │                  │────────────────────▶│                      │                      │
   │                  │                     │                      │                      │
   │                  │ processed samples   │                      │                      │
   │                  │◀────────────────────│                      │                      │
   │                  │                     │                      │                      │
   │                  │ combine_datasets(start_id=last_id+1)       │                      │
   │                  │─────────────────────┘                      │                      │
   │                  │                                            │                      │
   │                  │ upload_dataset()                           │                      │
   │                  │───────────────────────────────────────────▶│                      │
   │                  │                                            │                      │
   │                  │ success                                    │                      │
   │                  │◀───────────────────────────────────────────│                      │
   │                  │                                            │                      │
   │ success          │                                            │                      │
   │◀─────────────────│                                            │                      │
   │                  │                                            │                      │
```

## API Specifications

### BaseProcessor

```python
class BaseProcessor(ABC):
    """Base class for all dataset processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base processor."""
        
    @abstractmethod
    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """Process the dataset and yield samples in the standard schema."""
        
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return information about the dataset."""
        
    @abstractmethod
    def estimate_size(self) -> int:
        """Estimate the size of the dataset."""
        
    def validate_sample(self, sample: Dict[str, Any]) -> List[str]:
        """Validate a sample against the standard schema."""
        
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_file: Optional[str] = None) -> str:
        """Save checkpoint data to file."""
        
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file."""
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint file for this processor."""
        
    def generate_id(self, current_index: int) -> str:
        """Generate sequential ID in format S{n}."""
```

## Design Patterns

### Factory Pattern

The system uses a factory pattern to create processor instances based on the dataset name:

```python
def create_processor(dataset_name: str, config: Dict[str, Any]) -> BaseProcessor:
    """Create a processor instance for the specified dataset."""
    dataset_config = DATASET_CONFIG.get(dataset_name)
    processor_class_name = dataset_config.get("processor_class")
    processor_class = get_processor_class(processor_class_name)
    return processor_class(config)
```

### Strategy Pattern

The system uses a strategy pattern for dataset processing, where each processor implements a common interface but has its own strategy for processing its specific dataset:

```python
# GigaSpeech2 strategy
class GigaSpeech2Processor(BaseProcessor):
    def process(self, checkpoint=None, sample_mode=False, sample_size=5):
        # GigaSpeech2-specific processing logic
        
# Processed Voice TH strategy
class ProcessedVoiceTHProcessor(BaseProcessor):
    def process(self, checkpoint=None, sample_mode=False, sample_size=5):
        # ProcessedVoiceTH-specific processing logic
```

### Iterator Pattern

The system uses an iterator pattern to process datasets in a streaming fashion, yielding samples one at a time:

```python
def process(self, checkpoint=None, sample_mode=False, sample_size=5):
    # Process dataset
    for sample in dataset:
        # Process sample
        yield processed_sample
```

## Architecture Decisions

### Modular Design

The system is designed with modularity in mind, making it easy to add new dataset processors without modifying existing code. Each dataset processor is implemented as a separate class that inherits from the BaseProcessor abstract base class.

### Streaming Processing

To handle potentially large datasets, the system processes data in a streaming fashion, yielding samples one at a time rather than loading the entire dataset into memory.

### Checkpointing

The system implements a checkpointing mechanism to allow resuming interrupted processing. This is especially important for large datasets that may take a long time to process.

### Sample Mode

The system includes a sample mode feature that allows processing a small number of samples from each dataset for testing purposes. This makes it easier to validate processor functionality without waiting for complete dataset processing.

### Standardized Schema

All datasets are converted to a standardized schema, making it easier to combine them into a single dataset and ensuring consistency across different data sources.

### Command-Line Interface

The system provides a flexible command-line interface that supports different processing modes (fresh creation or incremental appending) and allows processing all datasets or specific ones.

## Extensibility

To add a new dataset processor:

1. Create a new file in the `processors` directory (e.g., `new_dataset.py`)
2. Implement a class that inherits from `BaseProcessor`
3. Implement the required methods: `process()`, `get_dataset_info()`, and `estimate_size()`
4. Add the dataset configuration to `DATASET_CONFIG` in `config.py`

This modular design makes it easy to extend the system with new dataset sources without modifying existing code.
