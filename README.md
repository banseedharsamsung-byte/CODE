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

**Requirements:**
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

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
  - `dataloader_num_workers`: PyTorch DataLoader workers for training (default: 4)
  - `num_preprocessing_workers`: Parallel workers for **offline data preprocessing** (YOLO → Florence conversion).
    - `0` or `1`: single-threaded (default, more detailed stats/logging)
    - `>1`: uses a thread pool to load/convert images in parallel (good speedup on IO-bound datasets)
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

## Dataset Format for LoRA Training

The pipeline automatically converts YOLO format to Florence-2 format for LoRA training. The data flows through several stages:

### 1. Raw Dataset Format

Each sample in the raw dataset (before processing) is a dictionary with two keys:

```python
{
    'image': PIL.Image,  # PIL Image object (RGB mode)
    'text': str          # Full target output sequence string
}
```

### 2. Text Format (Target Output Sequence)

The `text` field contains the **target output sequence** that the model should generate:

**Format**: `<OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...`

**Example**:
```
<OD>Subscription<loc_100><loc_200><loc_300><loc_400>Sign-in<loc_150><loc_250><loc_350><loc_450>
```

**Components**:
- `<OD>`: Object Detection task token (required at the start)
- `class_name`: Class name from your data.yaml (e.g., "Subscription", "Sign-in")
- `<loc_y1><loc_x1><loc_y2><loc_x2>`: Bounding box coordinates in Florence-2 format (0-999 range)
  - `y1, x1`: Top-left corner coordinates
  - `y2, x2`: Bottom-right corner coordinates

**Multiple Objects**: For images with multiple objects, the format repeats: `class_name<loc_y1><loc_x1><loc_y2><loc_x2>` for each object.

### 3. Data Collator Processing

The `Florence2DataCollator` processes batches of raw samples and converts them to the format expected by Florence-2:

**Processing Steps:**
1. **Image Processing**: Images are processed with the Florence-2 processor using the input prompt `"<OD>"` only (task token without additional text)
   - This creates `pixel_values` (processed image features)
   - Avoids the AssertionError that would occur if the full target sequence were passed to the processor

2. **Text Tokenization**: Target texts (full sequences) are tokenized separately using the tokenizer directly
   - Bypasses the processor's prompt construction logic
   - Creates `input_ids` and `attention_mask` tensors

3. **Label Creation**: Labels are created from `input_ids` for causal language modeling
   - Padding tokens are masked (set to -100) to ignore them in loss calculation

### 4. Final Format Passed to Florence-2 Model

The data collator returns a dictionary with PyTorch tensors ready for training:

```python
{
    'pixel_values': torch.Tensor,      # Shape: [batch_size, channels, height, width]
                                       # Processed image features from processor
                                       # Created using input prompt: "<OD>" only
    
    'input_ids': torch.Tensor,         # Shape: [batch_size, sequence_length]
                                       # Tokenized target sequences
                                       # Contains full sequence: <OD>class_name<loc_...>
    
    'attention_mask': torch.Tensor,    # Shape: [batch_size, sequence_length]
                                       # Attention mask for the sequences
    
    'labels': torch.Tensor             # Shape: [batch_size, sequence_length]
                                       # Same as input_ids, with padding tokens set to -100
}
```

**Key Points:**
- **Images** are processed with the task token `"<OD>"` only (as required by Florence-2 processor)
- **Target sequences** (full text with class names and locations) are tokenized separately
- The model learns to generate the full target sequence given the image and task token
- This approach avoids the `AssertionError: Task token <OD> should be the only token in the text` error

### Image Format

- **Type**: PIL Image object
- **Mode**: RGB (automatically converted if needed)
- **Size**: Original image dimensions (no resizing during dataset creation)

### Coordinate Conversion

The pipeline automatically converts YOLO format to Florence-2 format:

**YOLO Format**: Normalized `[cx, cy, w, h]` (0-1 range)
**Florence-2 Format**: Absolute `[y1, x1, y2, x2]` (0-999 range)

The conversion process:
1. Converts normalized YOLO coordinates to pixel coordinates
2. Calculates absolute corner coordinates (top-left and bottom-right)
3. Scales to Florence-2's 0-999 coordinate system
4. Formats as: `<OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...`

### Dataset Verification

During training initialization, the pipeline automatically prints 10 sample entries from the training dataset and 5 from the validation dataset (if available) for verification. This helps ensure the data format is correct before training begins.

**Example output**:
```
================================================================================
Verification: Printing 10 sample(s) from training dataset
================================================================================

--- Sample 1/10 ---
Image: (1920, 1080) (width x height), mode: RGB
Text: <OD>Subscription<loc_100><loc_200><loc_300><loc_400>Sign-in<loc_150><loc_250><loc_350><loc_450>
...
```

## Output

Training outputs are saved to the `output_dir` specified in config:
- Model checkpoints (every `save_steps`)
- Final model weights
- Training logs (TensorBoard compatible)
- **Visualization samples** (in `visualizations/` subdirectory):
  - Sample images with bounding boxes drawn
  - Green boxes: Original YOLO coordinates
  - Red boxes: Converted Florence-2 coordinates
  - Both should overlap if conversion is correct
  - Default: 10 samples per split (train/val)

## Saving and Reusing Converted Datasets (Intermediate Florence-Format)

For large datasets, repeatedly running the YOLO → Florence conversion can be slow. The pipeline can **save the converted Hugging Face Datasets to disk** and later **load them directly** to start training without reprocessing.

### Enable Saving of Converted Datasets

In `config.py`, set:

```python
from config import TrainingConfig

config = TrainingConfig(
    # ...
    save_converted_dataset=True,
    converted_dataset_dir="./converted_florence_dataset",  # optional, see below
)
```

- **`save_converted_dataset=True`**:
  - After preprocessing, the pipeline saves:
    - Train dataset: `converted_dataset_dir/train/`
    - Val dataset: `converted_dataset_dir/val/` (if available)
- **`converted_dataset_dir`** (optional):
  - If not set, it defaults to:
    - `<directory_of_data_yaml>/converted_florence_dataset`

The first run will:
- Convert YOLO → Florence
- Save `Dataset` objects to disk
- (Optionally) generate visualizations

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

### AssertionError: Task token <OD> should be the only token in the text

**This error has been fixed** in the current implementation. The data collator now:
- Processes images with the prompt `"<OD>"` only (as required by Florence-2)
- Tokenizes target sequences separately (bypassing the processor's prompt construction)

If you encounter this error, ensure you're using the latest version of `train.py` with the updated `Florence2DataCollator`.

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Enable `freeze_vision_tower=True`
- Reduce LoRA rank `r`
- Reduce `max_length` in the data collator (default: 512) if your sequences are shorter

### Dataset Not Found
- Verify `data_yaml_path` points to correct YOLO data.yaml
- Check that paths in data.yaml exist and contain `images/` and `labels/` directories

### Model Loading Issues
- Ensure model is downloaded to `model_path`
- Verify `trust_remote_code=True` is set (handled automatically)

### Sequence Length Issues
- If you get errors about sequence length, adjust the `max_length` parameter in `Florence2DataCollator.__call__()` in `train.py`
- Default is 512 tokens, which should be sufficient for most object detection tasks

### Converted Dataset Save/Load Issues
- If `load_converted_dataset=True` but the directory does not exist:
  - Ensure `converted_dataset_dir` points to the correct path
  - Check that `train/` and (optionally) `val/` subdirectories exist (created by `Dataset.save_to_disk`)
- If you change `data.yaml` or YOLO labels:
  - Re-run preprocessing with `save_converted_dataset=True` to regenerate the converted datasets

## Inference

After fine-tuning, you can use the `inference.py` script to run object detection on new images.

### Basic Usage

**Single image:**
```bash
python inference.py \
    --base_model_path ./models/florence-2-large \
    --lora_model_path ./outputs/custom_run \
    --image_path path/to/image.jpg \
    --output_dir ./inference_outputs
```

**Directory of images:**
```bash
python inference.py \
    --base_model_path ./models/florence-2-large \
    --lora_model_path ./outputs/custom_run \
    --image_dir path/to/images/ \
    --output_dir ./inference_outputs
```

### Arguments

- `--base_model_path`: Path to base Florence-2 model directory (required)
- `--lora_model_path`: Path to LoRA fine-tuned checkpoint directory (required)
- `--image_path`: Path to single image file (use with `--image_dir` for batch processing)
- `--image_dir`: Path to directory containing images (use with `--image_path` for single image)
- `--output_dir`: Directory to save output visualizations (default: `./inference_outputs`)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `--device`: Device to run inference on - `cuda` or `cpu` (default: auto-detect)

### Output

The script will:
1. **Print detection results** to console:
   - Generated text output from model
   - Parsed detections with class names and coordinates
   - Both Florence-2 coordinates (0-999) and pixel coordinates

2. **Save visualizations** to `--output_dir`:
   - Images with bounding boxes drawn
   - Class labels on each bounding box
   - Filename format: `{original_name}_detections.png`

### Example Output

```
================================================================================
Processing: test_image.jpg
================================================================================

Generated output:
<OD>Subscription<loc_100><loc_200><loc_300><loc_400>Sign-in<loc_150><loc_250><loc_350><loc_450>

Found 2 detection(s):
  1. Subscription: (100, 200, 300, 400) [Florence-2 coords]
  2. Sign-in: (150, 250, 350, 450) [Florence-2 coords]
  Subscription: (123, 245, 369, 491) [pixel coords]
  Sign-in: (185, 306, 432, 553) [pixel coords]
✓ Saved visualization to: ./inference_outputs/test_image_detections.png
```

### Notes

- The script automatically merges LoRA weights into the base model for inference
- Images are processed with the `<OD>` task token (as during training)
- Output format matches the training format: `<OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...`
- Coordinates are automatically converted from Florence-2 format (0-999) to pixel coordinates for visualization

## License

This code is provided as-is for fine-tuning Florence-2 models. Please refer to Microsoft's Florence-2 license for model usage terms.

