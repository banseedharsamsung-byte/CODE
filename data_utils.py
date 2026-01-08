"""
Data utilities for converting YOLO format to Florence-2 format.
Implements on-the-fly dataset generation without intermediate file conversion.
"""
import yaml
from pathlib import Path
from typing import Dict, Iterator, Tuple, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datasets import Dataset
import logging
from tqdm import tqdm
from collections import Counter
import random

# Set up logging
logger = logging.getLogger(__name__)


def parse_data_yaml(yaml_path: str) -> Tuple[Dict[int, str], List[str], List[str]]:
    """
    Parse YOLO data.yaml file to extract class mappings and dataset paths.
    
    Args:
        yaml_path: Path to data.yaml file
        
    Returns:
        Tuple of (class_id_to_name mapping, train_image_dirs, val_image_dirs)
    """
    logger.info(f"Parsing data configuration from: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Extract class names mapping
    names = data.get('names', {})
    class_id_to_name = {}
    
    # Handle both dict and list formats
    if isinstance(names, dict):
        class_id_to_name = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        class_id_to_name = {i: name for i, name in enumerate(names)}
    
    logger.info(f"Found {len(class_id_to_name)} classes: {list(class_id_to_name.values())}")
    
    def _normalize_dirs(value: Optional[Union[str, List[str]]]) -> List[str]:
        """Helper to normalize string or list of strings into a list."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    # New recommended format: explicit train/val lists of image directories
    train_image_dirs = _normalize_dirs(data.get("train"))
    val_image_dirs = _normalize_dirs(data.get("val"))

    # Backwards compatible format: single `path` root(s) that contain images/ and labels/
    # In this case we treat each path as a dataset root and infer images dir as root / "images"
    if not train_image_dirs and not val_image_dirs:
        paths = _normalize_dirs(data.get("path"))
        train_image_dirs = [str(Path(p) / "images") for p in paths]
        logger.info("Using legacy 'path' format - inferring image directories")
        # No explicit val dirs; caller can still randomly split if desired

    logger.info(f"Train image directories: {len(train_image_dirs)}")
    for i, dir_path in enumerate(train_image_dirs, 1):
        logger.info(f"  [{i}] {dir_path}")
    
    if val_image_dirs:
        logger.info(f"Validation image directories: {len(val_image_dirs)}")
        for i, dir_path in enumerate(val_image_dirs, 1):
            logger.info(f"  [{i}] {dir_path}")
    else:
        logger.info("No explicit validation directories found - will use random split if needed")

    return class_id_to_name, train_image_dirs, val_image_dirs


def yolo_to_florence_coords(
    cx: float, cy: float, w: float, h: float,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized [cx, cy, w, h] coordinates to Florence-2 absolute [y1, x1, y2, x2] format.
    
    Florence-2 uses absolute coordinates scaled to 0-999 range.
    Note: Florence-2 uses y1, x1, y2, x2 order (y-coordinate first).
    
    Args:
        cx: Normalized center x coordinate (0-1)
        cy: Normalized center y coordinate (0-1)
        w: Normalized width (0-1)
        h: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (y1, x1, y2, x2) as integers in 0-999 range
    """
    # Convert normalized coordinates to pixel coordinates
    center_x = cx * img_width
    center_y = cy * img_height
    box_width = w * img_width
    box_height = h * img_height
    
    # Calculate absolute corner coordinates
    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2
    
    # Clamp to image boundaries
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    # Scale to Florence-2 coordinate system (0-999)
    # Florence-2 uses 0-999 range for coordinates
    scale_x = 999.0 / img_width
    scale_y = 999.0 / img_height
    
    x1_scaled = int(round(x1 * scale_x))
    y1_scaled = int(round(y1 * scale_y))
    x2_scaled = int(round(x2 * scale_x))
    y2_scaled = int(round(y2 * scale_y))
    
    # Ensure within valid range
    x1_scaled = max(0, min(x1_scaled, 999))
    y1_scaled = max(0, min(y1_scaled, 999))
    x2_scaled = max(0, min(x2_scaled, 999))
    y2_scaled = max(0, min(y2_scaled, 999))
    
    # Return in Florence-2 format: y1, x1, y2, x2
    return y1_scaled, x1_scaled, y2_scaled, x2_scaled


def florence_to_pixel_coords(
    y1: int, x1: int, y2: int, x2: int,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert Florence-2 coordinates (0-999 range) back to pixel coordinates.
    
    Args:
        y1, x1, y2, x2: Florence-2 coordinates (0-999)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    scale_x = img_width / 999.0
    scale_y = img_height / 999.0
    
    x1_pixel = int(round(x1 * scale_x))
    y1_pixel = int(round(y1 * scale_y))
    x2_pixel = int(round(x2 * scale_x))
    y2_pixel = int(round(y2 * scale_y))
    
    return x1_pixel, y1_pixel, x2_pixel, y2_pixel


def yolo_to_pixel_coords(
    cx: float, cy: float, w: float, h: float,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates to pixel coordinates.
    
    Args:
        cx, cy, w, h: Normalized YOLO coordinates (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    center_x = cx * img_width
    center_y = cy * img_height
    box_width = w * img_width
    box_height = h * img_height
    
    x1 = int(round(center_x - box_width / 2))
    y1 = int(round(center_y - box_height / 2))
    x2 = int(round(center_x + box_width / 2))
    y2 = int(round(center_y + box_height / 2))
    
    return x1, y1, x2, y2


def visualize_annotations(
    image: Image.Image,
    yolo_annotations: List[Tuple[int, float, float, float, float]],
    florence_coords: List[Tuple[int, int, int, int]],
    class_id_to_name: Dict[int, str],
    show_yolo: bool = True,
    show_florence: bool = True
) -> Image.Image:
    """
    Draw bounding boxes on an image for visualization.
    
    Args:
        image: PIL Image
        yolo_annotations: List of (class_id, cx, cy, w, h) tuples
        florence_coords: List of (y1, x1, y2, x2) tuples in Florence-2 format
        class_id_to_name: Mapping from class ID to class name
        show_yolo: Whether to draw YOLO boxes (green)
        show_florence: Whether to draw Florence-2 boxes (red)
        
    Returns:
        Image with bounding boxes drawn
    """
    img_width, img_height = image.size
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw YOLO boxes (green) - original coordinates
    if show_yolo:
        for class_id, cx, cy, w, h in yolo_annotations:
            x1, y1, x2, y2 = yolo_to_pixel_coords(cx, cy, w, h, img_width, img_height)
            class_name = class_id_to_name.get(class_id, f"Class_{class_id}")
            
            # Draw rectangle (green for YOLO)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            
            # Draw label background
            if font:
                bbox = draw.textbbox((0, 0), class_name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(class_name) * 6
                text_height = 12
            
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="green")
            draw.text((x1 + 2, y1 - text_height - 2), class_name, fill="white", font=font)
    
    # Draw Florence-2 boxes (red) - converted coordinates
    if show_florence:
        for idx, (y1_flo, x1_flo, y2_flo, x2_flo) in enumerate(florence_coords):
            if idx < len(yolo_annotations):
                class_id = yolo_annotations[idx][0]
            else:
                class_id = 0  # Fallback
            
            x1, y1, x2, y2 = florence_to_pixel_coords(y1_flo, x1_flo, y2_flo, x2_flo, img_width, img_height)
            class_name = class_id_to_name.get(class_id, f"Class_{class_id}")
            
            # Draw rectangle (red for Florence-2)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Draw label background (only if not showing YOLO or if different position)
            if not show_yolo:
                if font:
                    bbox = draw.textbbox((0, 0), class_name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(class_name) * 6
                    text_height = 12
                
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
                draw.text((x1 + 2, y1 - text_height - 2), class_name, fill="white", font=font)
    
    return vis_image


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a YOLO label file.
    
    Format: class_id cx cy w h (normalized coordinates)
    
    Args:
        label_path: Path to YOLO label file
        
    Returns:
        List of tuples: (class_id, cx, cy, w, h)
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                annotations.append((class_id, cx, cy, w, h))
    
    return annotations


def generate_florence_prompt(
    annotations: List[Tuple[int, float, float, float, float]],
    class_id_to_name: Dict[int, str],
    img_width: int,
    img_height: int
) -> str:
    """
    Generate Florence-2 prompt string from YOLO annotations.
    
    Format: <OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...
    
    Args:
        annotations: List of (class_id, cx, cy, w, h) tuples
        class_id_to_name: Mapping from class ID to class name
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Florence-2 formatted prompt string
    """
    prompt_parts = ["<OD>"]
    
    for class_id, cx, cy, w, h in annotations:
        # Get class name
        class_name = class_id_to_name.get(class_id, "Unknown")
        
        # Convert coordinates
        y1, x1, y2, x2 = yolo_to_florence_coords(cx, cy, w, h, img_width, img_height)
        
        # Format: class_name<loc_y1><loc_x1><loc_y2><loc_x2>
        prompt_parts.append(f"{class_name}<loc_{y1}><loc_{x1}><loc_{y2}><loc_{x2}>")
    
    return "".join(prompt_parts)


def yolo_to_florence_generator(
    data_yaml_path: str,
    split: str = "train",
    show_progress: bool = True,
) -> Iterator[Dict]:
    """
    Generator function that yields Florence-2 formatted samples from YOLO dataset.
    
    This generator iterates through images and labels directories without saving
    intermediate converted files, making it memory-efficient.
    
    Args:
        data_yaml_path: Path to data.yaml file
        split: Which split to load: "train", "val", or "all"
        show_progress: Whether to show progress bars
        
    Yields:
        Dictionary with 'image' (PIL Image) and 'text' (prompt string) keys
    """
    # Parse data.yaml
    class_id_to_name, train_image_dirs, val_image_dirs = parse_data_yaml(data_yaml_path)

    # Select which image directory list to iterate over
    if split == "train":
        image_dirs = train_image_dirs
    elif split == "val":
        image_dirs = val_image_dirs
    else:  # "all" or anything else
        image_dirs = train_image_dirs + val_image_dirs

    if not image_dirs:
        raise ValueError(f"No image directories found for split='{split}' in {data_yaml_path}")

    logger.info(f"Processing {split} split: {len(image_dirs)} directory(ies)")
    
    # Statistics tracking
    total_images = 0
    total_annotations = 0
    skipped_no_labels = 0
    skipped_errors = 0
    class_counts = Counter()
    
    # Iterate through all configured image directories
    dir_iterator = tqdm(image_dirs, desc=f"Processing {split} directories", disable=not show_progress) if show_progress else image_dirs
    
    for dir_idx, images_path_str in enumerate(dir_iterator, 1):
        images_path = Path(images_path_str)
        
        if show_progress and hasattr(dir_iterator, 'set_postfix'):
            dir_iterator.set_postfix({"current_dir": images_path.name})

        if not images_path.exists():
            logger.warning(f"Images directory not found: {images_path}")
            continue

        # Infer labels directory:
        # if path ends with 'images', assume sibling 'labels'
        if images_path.name == "images":
            labels_path = images_path.parent / "labels"
        else:
            # Fallback: try replacing 'images' with 'labels' in the path string
            labels_path = Path(str(images_path).replace("images", "labels"))

        if not labels_path.exists():
            logger.warning(f"Labels directory not found for images dir {images_path}: {labels_path}")
            continue

        logger.info(f"[{dir_idx}/{len(image_dirs)}] Processing directory: {images_path}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in images_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"  Found {len(image_files)} image files")
        
        # Process images with progress bar
        file_iterator = tqdm(image_files, desc=f"  Converting images", leave=False, disable=not show_progress) if show_progress else image_files
        
        for image_file in file_iterator:
            # Load image and ensure RGB format
            try:
                image = Image.open(image_file).convert('RGB')
                img_width, img_height = image.size
            except Exception as e:
                logger.warning(f"Failed to load image {image_file}: {e}")
                skipped_errors += 1
                continue
            
            # Find corresponding label file
            label_file = labels_path / f"{image_file.stem}.txt"
            
            # Parse YOLO annotations
            annotations = parse_yolo_label(label_file)
            
            # Skip images without annotations
            if not annotations:
                skipped_no_labels += 1
                continue
            
            # Count classes
            for class_id, _, _, _, _ in annotations:
                class_counts[class_id] += 1
                total_annotations += 1
            
            total_images += 1
            
            # Generate Florence-2 prompt
            prompt = generate_florence_prompt(
                annotations, class_id_to_name, img_width, img_height
            )
            
            yield {
                'image': image,
                'text': prompt
            }
    
    # Log summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Preprocessing Summary for '{split}' split:")
    logger.info(f"{'='*60}")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Total annotations: {total_annotations}")
    logger.info(f"Average annotations per image: {total_annotations / total_images if total_images > 0 else 0:.2f}")
    logger.info(f"Skipped (no labels): {skipped_no_labels}")
    logger.info(f"Skipped (errors): {skipped_errors}")
    logger.info(f"\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_id_to_name.get(class_id, f"Unknown({class_id})")
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        logger.info(f"  {class_name} (ID {class_id}): {count} ({percentage:.1f}%)")
    logger.info(f"{'='*60}\n")


def save_visualization_samples(
    data_yaml_path: str,
    output_dir: str,
    num_samples: int = 10,
    split: str = "train"
) -> None:
    """
    Save visualization samples showing YOLO and Florence-2 bounding boxes.
    
    Args:
        data_yaml_path: Path to data.yaml file
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
        split: Which split to visualize ("train" or "val")
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating visualization samples for {split} split...")
    logger.info(f"{'='*60}")
    
    # Parse data.yaml
    class_id_to_name, train_image_dirs, val_image_dirs = parse_data_yaml(data_yaml_path)
    
    # Select directories
    if split == "train":
        image_dirs = train_image_dirs
    elif split == "val":
        image_dirs = val_image_dirs
    else:
        image_dirs = train_image_dirs + val_image_dirs
    
    if not image_dirs:
        logger.warning(f"No image directories found for split '{split}'")
        return
    
    # Collect samples
    samples_collected = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for images_path_str in image_dirs:
        images_path = Path(images_path_str)
        if not images_path.exists():
            continue
        
        # Infer labels directory
        if images_path.name == "images":
            labels_path = images_path.parent / "labels"
        else:
            labels_path = Path(str(images_path).replace("images", "labels"))
        
        if not labels_path.exists():
            continue
        
        # Get image files
        image_files = [
            f for f in images_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        for image_file in image_files:
            if len(samples_collected) >= num_samples:
                break
            
            try:
                image = Image.open(image_file).convert('RGB')
                img_width, img_height = image.size
                
                label_file = labels_path / f"{image_file.stem}.txt"
                yolo_annotations = parse_yolo_label(label_file)
                
                if not yolo_annotations:
                    continue
                
                # Convert to Florence-2 coordinates
                florence_coords = []
                for class_id, cx, cy, w, h in yolo_annotations:
                    y1, x1, y2, x2 = yolo_to_florence_coords(cx, cy, w, h, img_width, img_height)
                    florence_coords.append((y1, x1, y2, x2))
                
                samples_collected.append({
                    'image': image,
                    'yolo_annotations': yolo_annotations,
                    'florence_coords': florence_coords,
                    'image_file': image_file,
                    'img_width': img_width,
                    'img_height': img_height
                })
            except Exception as e:
                logger.warning(f"Failed to process {image_file} for visualization: {e}")
                continue
        
        if len(samples_collected) >= num_samples:
            break
    
    if not samples_collected:
        logger.warning("No samples collected for visualization")
        return
    
    # Create output directory
    vis_dir = Path(output_dir) / "visualizations" / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualizations
    logger.info(f"Saving {len(samples_collected)} visualization samples to {vis_dir}")
    
    for idx, sample in enumerate(tqdm(samples_collected, desc="Saving visualizations")):
        # Create visualization
        vis_image = visualize_annotations(
            sample['image'],
            sample['yolo_annotations'],
            sample['florence_coords'],
            class_id_to_name,
            show_yolo=True,
            show_florence=True
        )
        
        # Save image
        output_path = vis_dir / f"sample_{idx+1:03d}_{sample['image_file'].stem}.png"
        vis_image.save(output_path, "PNG")
    
    logger.info(f"✓ Visualizations saved to: {vis_dir}")
    logger.info(f"  - Green boxes: YOLO original coordinates")
    logger.info(f"  - Red boxes: Florence-2 converted coordinates")
    logger.info(f"  - Both should overlap if conversion is correct")
    logger.info(f"{'='*60}\n")


def create_dataset(
    data_yaml_path: str,
    validation_split: float = 0.1,
    seed: int = 42,
    show_progress: bool = True,
    save_visualizations: bool = True,
    num_visualization_samples: int = 10,
    visualization_output_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset]:
    """
    Create Hugging Face datasets for training and validation.
    
    Args:
        data_yaml_path: Path to data.yaml file
        validation_split: Fraction of data to use for validation if no explicit val split
        seed: Random seed for splitting
        show_progress: Whether to show progress bars during dataset creation
        save_visualizations: Whether to save visualization samples
        num_visualization_samples: Number of samples to visualize (per split)
        visualization_output_dir: Directory to save visualizations (defaults to output_dir)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("="*60)
    logger.info("Starting dataset creation and preprocessing")
    logger.info("="*60)
    
    # Inspect YAML to see if an explicit val split is provided
    _, train_image_dirs, val_image_dirs = parse_data_yaml(data_yaml_path)

    # Always create train dataset from train image dirs (or inferred paths)
    logger.info("\n[1/2] Creating training dataset...")
    train_dataset = Dataset.from_generator(
        yolo_to_florence_generator,
        gen_kwargs={
            "data_yaml_path": data_yaml_path,
            "split": "train",
            "show_progress": show_progress,
        },
    )
    logger.info(f"✓ Training dataset created: {len(train_dataset)} samples")

    # If explicit val dirs are provided, build a separate val dataset from them
    if val_image_dirs:
        logger.info("\n[2/2] Creating validation dataset from explicit directories...")
        val_dataset = Dataset.from_generator(
            yolo_to_florence_generator,
            gen_kwargs={
                "data_yaml_path": data_yaml_path,
                "split": "val",
                "show_progress": show_progress,
            },
        )
        logger.info(f"✓ Validation dataset created: {len(val_dataset)} samples")
    else:
        # Fallback: random split from the full dataset
        if validation_split > 0:
            logger.info(f"\n[2/2] Creating validation dataset via random split ({validation_split*100:.1f}%)...")
            full_dataset = train_dataset.train_test_split(
                test_size=validation_split,
                seed=seed,
            )
            train_dataset = full_dataset["train"]
            val_dataset = full_dataset["test"]
            logger.info(f"✓ Validation dataset created: {len(val_dataset)} samples (from random split)")
        else:
            logger.info("\n[2/2] No validation split requested")
            val_dataset = None

    logger.info("\n" + "="*60)
    logger.info("Dataset creation completed successfully!")
    logger.info(f"  Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Total samples: {len(train_dataset) + len(val_dataset)}")
    else:
        logger.info(f"  Total samples: {len(train_dataset)}")
    logger.info("="*60 + "\n")
    
    # Save visualization samples if requested
    if save_visualizations:
        output_dir = visualization_output_dir or "./visualizations"
        try:
            save_visualization_samples(
                data_yaml_path,
                output_dir,
                num_samples=num_visualization_samples,
                split="train"
            )
            if val_dataset is not None:
                save_visualization_samples(
                    data_yaml_path,
                    output_dir,
                    num_samples=num_visualization_samples,
                    split="val"
                )
        except Exception as e:
            logger.warning(f"Failed to save visualizations: {e}")

    return train_dataset, val_dataset

