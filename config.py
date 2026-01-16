"""
Configuration file for Florence-2 LoRA fine-tuning for Object Detection.
Contains hyperparameters, paths, and training settings.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration parameters."""
    r: int = 16  # Rank (higher rank helps with high-density layouts)
    lora_alpha: int = 32  # LoRA alpha scaling parameter
    lora_dropout: float = 0.1  # LoRA dropout rate
    target_modules: List[str] = None  # Target modules for LoRA adaptation
    
    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Model paths
    model_path: str = "./models/florence-2-large"  # Local path to Florence-2 model
    data_yaml_path: str = "./data/data.yaml"  # Path to YOLO data.yaml file
    
    # Dataset paths (relative to paths in data.yaml)
    images_dir: str = "images"  # Images subdirectory name
    labels_dir: str = "labels"  # Labels subdirectory name
    
    # Output paths
    output_dir: str = "./outputs"  # Directory for checkpoints and logs
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2  # Batch size per device
    gradient_accumulation_steps: int = 4  # Gradient accumulation steps
    learning_rate: float = 5e-5  # Learning rate
    num_train_epochs: int = 3  # Number of training epochs
    max_steps: Optional[int] = None  # Maximum training steps (overrides epochs if set)
    
    # Precision
    fp16: bool = True  # Use FP16 mixed precision
    bf16: bool = False  # Use BF16 mixed precision (if supported)
    
    # Optimization
    warmup_steps: int = 100  # Number of warmup steps
    weight_decay: float = 0.01  # Weight decay
    
    # Logging and checkpointing
    logging_steps: int = 10  # Log every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    eval_steps: Optional[int] = None  # Evaluate every N steps (None = no eval)
    save_total_limit: int = 3  # Keep only last N checkpoints
    
    # Validation
    validation_split: float = 0.1  # Fraction of data to use for validation
    
    # Other settings
    seed: int = 42  # Random seed
    dataloader_num_workers: int = 4  # Number of dataloader workers
    remove_unused_columns: bool = False  # Keep all columns in dataset
    num_preprocessing_workers: int = 0  # Number of parallel workers for data preprocessing (0/1 = no parallelism)
    
    # LoRA configuration
    lora: LoRAConfig = None
    
    # Memory optimization
    freeze_vision_tower: bool = False  # Freeze vision tower to save VRAM
    
    # Multi-GPU settings
    use_model_parallelism: bool = False  # If True, use device_map="auto" for model parallelism
                                        # If False, use data parallelism (DDP) - recommended for training
                                        # Note: Model parallelism splits model across GPUs
                                        #       Data parallelism replicates model on each GPU
    
    # Visualization settings
    save_visualizations: bool = True  # Save visualization samples after dataset creation
    num_visualization_samples: int = 10  # Number of samples to visualize per split
    visualization_output_dir: Optional[str] = None  # Directory for visualizations (defaults to output_dir)
    
    def __post_init__(self):
        """Initialize LoRA config if not provided."""
        if self.lora is None:
            self.lora = LoRAConfig()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

