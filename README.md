# Florence-2 LoRA Fine-tuning for Object Detection

A modular Python training pipeline to LoRA fine-tune `microsoft/Florence-2-large` for Object Detection using a YOLO-structured dataset.

## Features

- **Local Model Loading**: Loads Florence-2 model and processor from a specified local directory
- **On-the-Fly Data Processing**: Converts YOLO format to Florence-2 format without intermediate files
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using PEFT
- **Modular Design**: Separated configuration, data utilities, and training logic
- **Memory Efficient**: Optional vision tower freezing for limited VRAM scenarios

## Project Structure

```
florence_finetune/
├── config.py          # Hyperparameter management
├── data_utils.py      # YOLO-to-Florence dataset conversion
├── train.py           # Main training script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Florence-2-large model:
```bash
# Using Hugging Face CLI
huggingface-cli download microsoft/Florence-2-large --local-dir ./models/florence-2-large

# Or manually download and extract to ./models/florence-2-large/
```

## Dataset Format

Your dataset should follow the YOLO structure:

```
data/
├── data.yaml          # Dataset configuration file
└── path1/             # Dataset path(s) specified in data.yaml
    ├── images/        # Image files (.jpg, .png, etc.)
    └── labels/        # YOLO label files (.txt)
```

### data.yaml Format

```yaml
# Explicit train/val image directory lists.
# Each directory MUST contain a matching `labels/` directory at the same level.
#
# Example structure for `/dir1/train/images/`:
#   /dir1/train/images/xxx.jpg
#   /dir1/train/labels/xxx.txt   # same stem `xxx`

train:
  - /dir1/train/images/
  - /dir2/train/images/

val:
  - /dir1/val/images/
  - /dir2/val/images/

names:
  0: Subscription
  1: Sign-in
  2: Discount/Coupon
  3: Survey
  4: Cookies
  5: Error
  6: Payment
  7: Install/Download
  8: Notifications
  9: Help
  10: Open-in-app
  11: Confirmation
  12: Advertisement
  13: Unknown
```

### YOLO Label Format

Each label file should contain one line per object:
```
class_id cx cy w h
```

Where:
- `class_id`: Integer class ID (mapped to names in data.yaml)
- `cx, cy`: Normalized center coordinates (0-1)
- `w, h`: Normalized width and height (0-1)

Example:
```
0 0.5 0.5 0.3 0.2
5 0.2 0.8 0.1 0.15
```

## Configuration

Edit `config.py` to customize training parameters:

### Key Parameters

- **Model Path**: `model_path` - Local path to Florence-2 model directory
- **Data Path**: `data_yaml_path` - Path to your data.yaml file
- **LoRA Settings**:
  - `r`: Rank (default: 16)
  - `lora_alpha`: Alpha scaling (default: 32)
  - `lora_dropout`: Dropout rate (default: 0.1)
  - `target_modules`: Modules to apply LoRA (default: ["q_proj", "k_proj", "v_proj", "out_proj"])
- **Training Settings**:
  - `per_device_train_batch_size`: Batch size per device (default: 2)
  - `gradient_accumulation_steps`: Gradient accumulation (default: 4)
  - `learning_rate`: Learning rate (default: 5e-5)
  - `num_train_epochs`: Number of epochs (default: 3)
- **Memory Optimization**:
  - `freeze_vision_tower`: Freeze vision tower to save VRAM (default: False)

## Usage

### Basic Training

```bash
python train.py
```

### Custom Configuration

You can modify `config.py` directly or create a custom configuration:

```python
from config import TrainingConfig, LoRAConfig
from train import main

# Create custom config
config = TrainingConfig(
    model_path="./models/florence-2-large",
    data_yaml_path="./data/data.yaml",
    output_dir="./outputs/custom_run",
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    lora=LoRAConfig(r=32, lora_alpha=64)
)

main(config)
```

## Coordinate Conversion

The pipeline automatically converts YOLO format to Florence-2 format:

**YOLO Format**: Normalized `[cx, cy, w, h]` (0-1 range)
**Florence-2 Format**: Absolute `[y1, x1, y2, x2]` (0-999 range)

The conversion:
1. Converts normalized coordinates to pixel coordinates
2. Calculates absolute corner coordinates
3. Scales to Florence-2's 0-999 coordinate system
4. Formats as: `<OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...`

## Output

Training outputs are saved to the `output_dir` specified in config:
- Model checkpoints (every `save_steps`)
- Final model weights
- Training logs (TensorBoard compatible)

## Multi-GPU Training

The pipeline supports two multi-GPU strategies:

### 1. Data Parallelism (DDP) - Recommended for Training

**Default behavior**: When launched with multiple GPUs, the Trainer automatically uses Distributed Data Parallelism (DDP).

**How to use:**
```bash
# Using torchrun (PyTorch native)
torchrun --nproc_per_node=4 train.py

# Using accelerate (recommended)
accelerate launch --num_processes=4 train.py

# Or configure accelerate first
accelerate config
accelerate launch train.py
```

**Benefits:**
- Each GPU gets a copy of the model
- Data is split across GPUs
- Gradients are synchronized
- Faster training with linear scaling
- Works seamlessly with Hugging Face Trainer

**Configuration:**
- Set `use_model_parallelism=False` in config (default)
- Effective batch size = `per_device_train_batch_size × num_gpus × gradient_accumulation_steps`

### 2. Model Parallelism (Device Map)

**For very large models that don't fit on a single GPU:**

```python
config.use_model_parallelism = True
```

**How it works:**
- Model layers are split across multiple GPUs
- Each GPU holds different parts of the model
- Useful when model is too large for single GPU
- **Note**: Not compatible with Trainer's DDP - use single process

**Limitations:**
- Slower than DDP for training
- More complex communication patterns
- Best for inference or when model doesn't fit on one GPU

### Multi-GPU Best Practices

1. **Use DDP for training** (default): Faster and scales better
2. **Adjust batch size**: With 4 GPUs, `per_device_train_batch_size=2` gives effective batch size of 8
3. **Gradient accumulation**: Use to simulate larger batch sizes without OOM
4. **Learning rate scaling**: Consider scaling LR with number of GPUs (e.g., `lr = base_lr × num_gpus`)

## Memory Optimization

For limited VRAM scenarios:

1. **Freeze Vision Tower**: Set `freeze_vision_tower=True` in config
2. **Reduce Batch Size**: Lower `per_device_train_batch_size`
3. **Increase Gradient Accumulation**: Increase `gradient_accumulation_steps` to maintain effective batch size
4. **Use FP16**: Ensure `fp16=True` (default)
5. **Use Model Parallelism**: If model doesn't fit, set `use_model_parallelism=True`

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Enable `freeze_vision_tower=True`
- Reduce LoRA rank `r`

### Dataset Not Found
- Verify `data_yaml_path` points to correct YOLO data.yaml
- Check that paths in data.yaml exist and contain `images/` and `labels/` directories

### Model Loading Issues
- Ensure model is downloaded to `model_path`
- Verify `trust_remote_code=True` is set (handled automatically)

## License

This code is provided as-is for fine-tuning Florence-2 models. Please refer to Microsoft's Florence-2 license for model usage terms.

