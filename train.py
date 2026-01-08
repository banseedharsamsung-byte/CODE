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
    freeze_vision_tower: bool = False
):
    """
    Load Florence-2 model and processor from local directory.
    
    Args:
        model_path: Local path to Florence-2 model directory
        freeze_vision_tower: Whether to freeze vision tower parameters
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model and processor from {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
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
        freeze_vision_tower=config.freeze_vision_tower
    )
    
    # Apply LoRA
    model = setup_lora(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_dataset(
        config.data_yaml_path,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data collator
    data_collator = Florence2DataCollator(processor)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        report_to="tensorboard" if os.path.exists(config.output_dir) else None,
        load_best_model_at_end=True if val_dataset is not None and config.eval_steps else False,
        metric_for_best_model="loss" if val_dataset is not None else None,
        greater_is_better=False,
    )
    
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

