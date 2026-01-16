"""
Main training script for LoRA fine-tuning Florence-2-large for Object Detection.
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from PIL import Image
from typing import Dict, List, Optional
import logging

from config import TrainingConfig
from data_utils import create_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Florence2DataCollator:
    """
    Data collator for Florence-2 that handles image and text batching.
    Uses the processor to pad and tokenize batches.
    """
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            features: List of dictionaries with 'image' and 'text' keys
            
        Returns:
            Dictionary with processed inputs ready for model
        """
        # Extract images and texts
        images = [feature['image'] for feature in features]
        texts = [feature['text'] for feature in features]
        
        # Process images and texts using the processor
        # The processor handles tokenization, padding, and image preprocessing
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True
        )
        
        # For training, we need to create labels from input_ids
        # Florence-2 uses causal language modeling, so labels are the same as input_ids
        # (the model handles shifting internally during forward pass)
        labels = inputs['input_ids'].clone()
        
        # Replace padding tokens with -100 to ignore them in loss calculation
        pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        else:
            # If no pad_token_id, mask based on attention mask if available
            if 'attention_mask' in inputs:
                labels[inputs['attention_mask'] == 0] = -100
        
        inputs['labels'] = labels
        
        return inputs


def load_model_and_processor(
    model_path: str,
    freeze_vision_tower: bool = False,
    use_model_parallelism: bool = False
):
    """
    Load Florence-2 model and processor from local directory.
    
    Args:
        model_path: Local path to Florence-2 model directory
        freeze_vision_tower: Whether to freeze vision tower parameters
        use_model_parallelism: If True, use device_map="auto" for model parallelism.
                              If False (default), let Trainer handle device placement for data parallelism.
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model and processor from {model_path}")
    
    # Check for multi-GPU setup
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        logger.info(f"Detected {num_gpus} GPUs")
        if use_model_parallelism:
            logger.info("Using model parallelism (device_map='auto')")
        else:
            logger.info("Using data parallelism (Trainer will handle DDP)")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    # For data parallelism (DDP), don't use device_map - Trainer handles it
    # For model parallelism, use device_map="auto" to split model across GPUs
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    
    if use_model_parallelism and torch.cuda.is_available():
        model_kwargs['device_map'] = "auto"
    elif torch.cuda.is_available() and not use_model_parallelism:
        # For data parallelism, place model on first GPU initially
        # Trainer will replicate it across GPUs
        model_kwargs['device_map'] = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # If not using device_map, manually move to GPU for data parallelism
    if not use_model_parallelism and torch.cuda.is_available():
        model = model.cuda()
    
    # Freeze vision tower if requested (for memory optimization)
    if freeze_vision_tower:
        logger.info("Freezing vision tower parameters")
        for name, param in model.named_parameters():
            if 'vision_model' in name or 'vision_tower' in name:
                param.requires_grad = False
    
    return model, processor


def setup_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
):
    """
    Apply LoRA (Low-Rank Adaptation) to the model.
    
    Args:
        model: The base model to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to
        
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    logger.info(f"Setting up LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def print_dataset_samples(dataset: Dataset, num_samples: int = 10, split_name: str = "training"):
    """
    Print sample entries from the dataset for verification.
    
    Args:
        dataset: The dataset to print samples from
        num_samples: Number of samples to print (default: 10)
        split_name: Name of the dataset split (for logging)
    """
    logger.info("="*80)
    logger.info(f"Verification: Printing {num_samples} sample(s) from {split_name} dataset")
    logger.info("="*80)
    
    # Get actual number of samples to print (don't exceed dataset size)
    num_to_print = min(num_samples, len(dataset))
    
    for i in range(num_to_print):
        sample = dataset[i]
        logger.info(f"\n--- Sample {i+1}/{num_to_print} ---")
        
        # Print image info
        if 'image' in sample:
            img = sample['image']
            if isinstance(img, Image.Image):
                logger.info(f"Image: {img.size} (width x height), mode: {img.mode}")
            else:
                logger.info(f"Image: {type(img)}")
        
        # Print text/prompt
        if 'text' in sample:
            text = sample['text']
            # Truncate very long texts for readability
            if len(text) > 500:
                logger.info(f"Text (truncated): {text[:500]}...")
            else:
                logger.info(f"Text: {text}")
        
        # Print any other keys
        other_keys = [k for k in sample.keys() if k not in ['image', 'text']]
        if other_keys:
            logger.info(f"Other keys: {other_keys}")
    
    logger.info("\n" + "="*80)
    logger.info(f"Dataset verification complete. Total {split_name} samples: {len(dataset)}")
    logger.info("="*80 + "\n")


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    # For object detection, we can compute token-level accuracy
    # More sophisticated metrics can be added here
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    # Mask out padding tokens
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {"accuracy": accuracy}


def main(config: TrainingConfig):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    logger.info("Starting Florence-2 LoRA fine-tuning for Object Detection")
    logger.info(f"Configuration: {config}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load model and processor
    model, processor = load_model_and_processor(
        config.model_path,
        freeze_vision_tower=config.freeze_vision_tower,
        use_model_parallelism=config.use_model_parallelism
    )
    
    # Apply LoRA
    model = setup_lora(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules
    )
    
    # Create datasets (with detailed logging)
    train_dataset, val_dataset = create_dataset(
        config.data_yaml_path,
        validation_split=config.validation_split,
        seed=config.seed,
        show_progress=True,
        save_visualizations=config.save_visualizations,
        num_visualization_samples=config.num_visualization_samples,
        visualization_output_dir=config.visualization_output_dir or config.output_dir,
        num_preprocessing_workers=config.num_preprocessing_workers,
    )
    
    # Print sample dataset entries for verification
    print_dataset_samples(train_dataset, num_samples=10, split_name="training")
    if val_dataset is not None:
        print_dataset_samples(val_dataset, num_samples=5, split_name="validation")
    
    # Create data collator
    data_collator = Florence2DataCollator(processor)
    
    # Setup training arguments
    # Note: Hugging Face Trainer automatically detects and uses multiple GPUs
    # when launched with torchrun or accelerate. It uses DistributedDataParallel (DDP)
    # for data parallelism, which replicates the model on each GPU and splits the data.
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1 and not config.use_model_parallelism:
        logger.info(f"Multi-GPU training detected: {num_gpus} GPUs will be used with DDP")
        logger.info(f"Effective batch size: {config.per_device_train_batch_size * num_gpus * config.gradient_accumulation_steps}")
    
    # Setup training arguments
    # Note: Hugging Face Trainer automatically detects and uses multiple GPUs
    # when launched with torchrun or accelerate. It uses DistributedDataParallel (DDP)
    # for data parallelism, which replicates the model on each GPU and splits the data.
    training_args_dict = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "fp16": config.fp16 and torch.cuda.is_available(),
        "bf16": config.bf16 and torch.cuda.is_available(),
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "remove_unused_columns": config.remove_unused_columns,
        "report_to": "tensorboard" if os.path.exists(config.output_dir) else None,
        "greater_is_better": False,
        # DDP settings (automatically handled by Trainer when using torchrun/accelerate)
        "ddp_find_unused_parameters": False,  # Set to True if you encounter DDP errors
    }
    
    # Only add eval_steps if it's not None
    if config.eval_steps is not None:
        training_args_dict["eval_steps"] = config.eval_steps
        training_args_dict["load_best_model_at_end"] = True if val_dataset is not None else False
        training_args_dict["metric_for_best_model"] = "loss" if val_dataset is not None else None
    else:
        training_args_dict["load_best_model_at_end"] = False
        training_args_dict["metric_for_best_model"] = None
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_dataset is not None else None,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    # Load configuration
    config = TrainingConfig()
    
    # Override with command-line arguments or environment variables if needed
    # For example:
    # config.model_path = os.getenv("MODEL_PATH", config.model_path)
    # config.data_yaml_path = os.getenv("DATA_YAML_PATH", config.data_yaml_path)
    
    main(config)

