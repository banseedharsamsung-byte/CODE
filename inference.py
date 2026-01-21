"""
Inference script for LoRA fine-tuned Florence-2 object detection model.

Loads a fine-tuned model and performs object detection on images.
Prints detection results and saves visualizations with bounding boxes.
"""

import os
import re
import torch
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import logging

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_florence_output(text: str) -> List[Tuple[str, int, int, int, int]]:
    """
    Parse Florence-2 output string to extract class names and bounding boxes.
    
    Format: <OD>class_name<loc_y1><loc_x1><loc_y2><loc_x2>class_name<loc_y1><loc_x1><loc_y2><loc_x2>...
    
    Args:
        text: Generated text string from model
        
    Returns:
        List of tuples: (class_name, y1, x1, y2, x2) in Florence-2 coordinates (0-999)
    """
    detections = []
    
    # Remove <OD> prefix if present
    text = text.strip()
    if text.startswith("<OD>"):
        text = text[4:]
    
    # Pattern to match: class_name<loc_y1><loc_x1><loc_y2><loc_x2>
    # This regex matches class names (any characters except <) followed by location tokens
    pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        class_name = match[0].strip()
        y1 = int(match[1])
        x1 = int(match[2])
        y2 = int(match[3])
        x2 = int(match[4])
        
        detections.append((class_name, y1, x1, y2, x2))
    
    return detections


def florence_to_pixel_coords(
    y1: int, x1: int, y2: int, x2: int,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert Florence-2 coordinates (0-999 range) to pixel coordinates.
    
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
    
    # Ensure coordinates are within image bounds
    x1_pixel = max(0, min(x1_pixel, img_width - 1))
    y1_pixel = max(0, min(y1_pixel, img_height - 1))
    x2_pixel = max(0, min(x2_pixel, img_width - 1))
    y2_pixel = max(0, min(y2_pixel, img_height - 1))
    
    return x1_pixel, y1_pixel, x2_pixel, y2_pixel


def draw_bboxes(
    image: Image.Image,
    detections: List[Tuple[str, int, int, int, int]],
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image
        detections: List of (class_name, x1, y1, x2, y2) in pixel coordinates
        output_path: Optional path to save the visualization
        
    Returns:
        Image with bounding boxes drawn
    """
    img_width, img_height = image.size
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Colors for different classes (cycle through)
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    
    for idx, (class_name, x1, y1, x2, y2) in enumerate(detections):
        color = colors[idx % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        if font:
            bbox = draw.textbbox((0, 0), class_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(class_name) * 8
            text_height = 16
        
        # Draw label background rectangle
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 8, y1],
            fill=color
        )
        draw.text(
            (x1 + 4, label_y),
            class_name,
            fill="white",
            font=font
        )
    
    if output_path:
        vis_image.save(output_path, "PNG")
        logger.info(f"Saved visualization to: {output_path}")
    
    return vis_image


def load_model_and_processor(
    base_model_path: str,
    lora_model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load base model and LoRA weights.
    
    Args:
        base_model_path: Path to base Florence-2 model
        lora_model_path: Path to LoRA fine-tuned checkpoint directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading base model from: {base_model_path}")
    logger.info(f"Loading LoRA weights from: {lora_model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()  # Merge LoRA weights into base model
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    
    return model, processor


def inference_single_image(
    model,
    processor,
    image_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 512
) -> str:
    """
    Run inference on a single image.
    
    Args:
        model: Loaded Florence-2 model
        processor: Florence-2 processor
        image_path: Path to image file
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text string
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Process image with "<OD>" prompt
    inputs = processor(
        images=image,
        text="<OD>",
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to device and align pixel dtype with model dtype (avoid float32 vs float16 mismatch)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (
            v.to(device).to(model_dtype) if k == "pixel_values" else v.to(device)
        )
        for k, v in inputs.items()
    }
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    
    # Decode generated text (keep both raw and cleaned for debugging/parsing)
    generated_text_raw = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]
    generated_text_clean = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    return generated_text_raw, generated_text_clean


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA fine-tuned Florence-2 model"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base Florence-2 model directory"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="Path to LoRA fine-tuned checkpoint directory"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to single image file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save output visualizations"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image_path and not args.image_dir:
        parser.error("Either --image_path or --image_dir must be provided")
    
    if args.image_path and args.image_dir:
        parser.error("Provide either --image_path or --image_dir, not both")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    model, processor = load_model_and_processor(
        args.base_model_path,
        args.lora_model_path,
        device=args.device
    )
    
    # Collect image paths
    image_paths = []
    if args.image_path:
        image_paths = [Path(args.image_path)]
    else:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_dir = Path(args.image_dir)
        image_paths = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        image_paths.sort()
    
    logger.info(f"Processing {len(image_paths)} image(s)")
    
    # Process each image
    for img_path in image_paths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {img_path}")
        logger.info(f"{'='*80}")
        
        try:
            # Run inference
            generated_text_raw, generated_text_clean = inference_single_image(
                model,
                processor,
                str(img_path),
                device=args.device,
                max_new_tokens=args.max_new_tokens
            )
            
            logger.info(f"\nGenerated output (raw, no skipping special tokens):")
            logger.info(f"{generated_text_raw}\n")
            logger.info(f"Generated output (clean, skip_special_tokens=True):")
            logger.info(f"{generated_text_clean}\n")
            
            # Parse detections
            detections_florence = parse_florence_output(generated_text_raw)
            if not detections_florence:
                # Fallback: try parsing the cleaned text
                detections_florence = parse_florence_output(generated_text_clean)
            
            if not detections_florence:
                logger.warning("No detections found in output")
                continue
            
            logger.info(f"Found {len(detections_florence)} detection(s):")
            for idx, (class_name, y1, x1, y2, x2) in enumerate(detections_florence, 1):
                logger.info(f"  {idx}. {class_name}: ({y1}, {x1}, {y2}, {x2}) [Florence-2 coords]")
            
            # Load image and convert coordinates
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            detections_pixel = []
            for class_name, y1, x1, y2, x2 in detections_florence:
                x1_pix, y1_pix, x2_pix, y2_pix = florence_to_pixel_coords(
                    y1, x1, y2, x2, img_width, img_height
                )
                detections_pixel.append((class_name, x1_pix, y1_pix, x2_pix, y2_pix))
                logger.info(f"  {class_name}: ({x1_pix}, {y1_pix}, {x2_pix}, {y2_pix}) [pixel coords]")
            
            # Draw bounding boxes and save
            output_path = output_dir / f"{img_path.stem}_detections.png"
            draw_bboxes(image, detections_pixel, output_path=str(output_path))
            
            logger.info(f"✓ Saved visualization to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info("Inference completed!")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

