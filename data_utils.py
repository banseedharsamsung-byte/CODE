"""
Data utilities for converting YOLO format to Florence-2 format.
Implements on-the-fly dataset generation without intermediate file conversion.
"""
import yaml
from pathlib import Path
from typing import Dict, Iterator, Tuple, List, Optional
from PIL import Image
import numpy as np
from datasets import Dataset


def parse_data_yaml(yaml_path: str) -> Tuple[Dict[int, str], List[str], List[str]]:
    """
    Parse YOLO data.yaml file to extract class mappings and dataset paths.
    
    Args:
        yaml_path: Path to data.yaml file
        
    Returns:
        Tuple of (class_id_to_name mapping, train_image_dirs, val_image_dirs)
    """
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
    
    def _normalize_dirs(value: Optional[str | List[str]]) -> List[str]:
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
        # No explicit val dirs; caller can still randomly split if desired

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
) -> Iterator[Dict]:
    """
    Generator function that yields Florence-2 formatted samples from YOLO dataset.
    
    This generator iterates through images and labels directories without saving
    intermediate converted files, making it memory-efficient.
    
    Args:
        data_yaml_path: Path to data.yaml file
        split: Which split to load: "train", "val", or "all"
        
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

    # Iterate through all configured image directories
    for images_path_str in image_dirs:
        images_path = Path(images_path_str)

        if not images_path.exists():
            print(f"Warning: Images directory not found: {images_path}")
            continue

        # Infer labels directory:
        # if path ends with 'images', assume sibling 'labels'
        if images_path.name == "images":
            labels_path = images_path.parent / "labels"
        else:
            # Fallback: try replacing 'images' with 'labels' in the path string
            labels_path = Path(str(images_path).replace("images", "labels"))

        if not labels_path.exists():
            print(f"Warning: Labels directory not found for images dir {images_path}: {labels_path}")
            continue

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in images_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        for image_file in image_files:
            # Load image and ensure RGB format
            try:
                image = Image.open(image_file).convert('RGB')
                img_width, img_height = image.size
            except Exception as e:
                print(f"Warning: Failed to load image {image_file}: {e}")
                continue
            
            # Find corresponding label file
            label_file = labels_path / f"{image_file.stem}.txt"
            
            # Parse YOLO annotations
            annotations = parse_yolo_label(label_file)
            
            # Skip images without annotations
            if not annotations:
                continue
            
            # Generate Florence-2 prompt
            prompt = generate_florence_prompt(
                annotations, class_id_to_name, img_width, img_height
            )
            
            yield {
                'image': image,
                'text': prompt
            }


def create_dataset(
    data_yaml_path: str,
    validation_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Create Hugging Face datasets for training and validation.
    
    Args:
        data_yaml_path: Path to data.yaml file
        validation_split: Fraction of data to use for validation if no explicit val split
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Inspect YAML to see if an explicit val split is provided
    _, train_image_dirs, val_image_dirs = parse_data_yaml(data_yaml_path)

    # Always create train dataset from train image dirs (or inferred paths)
    train_dataset = Dataset.from_generator(
        yolo_to_florence_generator,
        gen_kwargs={
            "data_yaml_path": data_yaml_path,
            "split": "train",
        },
    )

    # If explicit val dirs are provided, build a separate val dataset from them
    if val_image_dirs:
        val_dataset = Dataset.from_generator(
            yolo_to_florence_generator,
            gen_kwargs={
                "data_yaml_path": data_yaml_path,
                "split": "val",
            },
        )
    else:
        # Fallback: random split from the full dataset
        if validation_split > 0:
            full_dataset = train_dataset.train_test_split(
                test_size=validation_split,
                seed=seed,
            )
            train_dataset = full_dataset["train"]
            val_dataset = full_dataset["test"]
        else:
            val_dataset = None

    return train_dataset, val_dataset

