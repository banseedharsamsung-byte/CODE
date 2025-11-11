# Setup Guide

## Quick Start

1. **Place Model File**
   - Copy `ui_model_0.7.tflite` to `app/src/main/assets/` directory
   - If the assets folder doesn't exist, create it

2. **Adjust Model Parameters** (if needed)
   - Open `app/src/main/java/com/samsung/uielementtestapp/detection/UIDetectionProcessor.java`
   - Modify these constants based on your model:
     - `INPUT_SIZE`: Your model's input image size (default: 640)
     - `CONFIDENCE_THRESHOLD`: Minimum confidence (default: 0.5)
     - `IOU_THRESHOLD`: NMS IoU threshold (default: 0.5)
   - In `MainActivity.java`, adjust:
     - `BOX_COUNT`: Number of output boxes from your model (default: 8400)

3. **Adjust Output Format** (if needed)
   - The `postProcess` methods in `UIDetectionProcessor.java` assume:
     - Format: `[x_center, y_center, width, height, objectness, ...class_scores]`
     - Coordinates are normalized (0-1)
   - If your model uses a different format, modify the `postProcess` methods accordingly

4. **Build and Run**
   ```bash
   ./gradlew build
   ./gradlew installDebug
   ```

## Model Output Format

The app expects the model output in one of these formats:
- 3D: `[batch, boxes, 5+classes]` where each box is `[x, y, w, h, conf, ...class_scores]`
- 2D: `[boxes, 5+classes]` with the same box format

If your model uses a different format (e.g., corner coordinates instead of center+size), you'll need to modify the post-processing logic in `UIDetectionProcessor.java`.

## Troubleshooting

- **Model not loading**: Ensure `ui_model_0.7.tflite` is in `app/src/main/assets/`
- **No detections**: Check that the output format matches your model
- **Wrong bounding boxes**: Adjust the coordinate conversion in `postProcess` methods
- **App crashes**: Check logcat for detailed error messages

