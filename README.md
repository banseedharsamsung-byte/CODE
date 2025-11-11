# UI Element Detection Test App

A modular Android application for testing UI element detection models using TensorFlow Lite.

## Features

- **Image Upload**: Select images from device storage
- **UI Element Detection**: Run inference on selected images
- **Visualization**: Display bounding boxes with labels and confidence scores
- **Modular Architecture**: Clean separation of concerns

## Project Structure

```
app/src/main/java/com/samsung/uielementtestapp/
├── MainActivity.java                    # Main activity with UI
├── core/
│   └── ModelUtility.java               # TensorFlow Lite model loader
├── detection/
│   ├── Data.java                       # Detection data class
│   ├── UIDetectionProcessor.java       # Main detection processor
│   ├── bbox/
│   │   └── BBoxUtils.java             # Bounding box utilities
│   └── nms/
│       └── NMSOperations.java         # Non-Maximum Suppression
├── view/
│   └── BoundingBoxView.java           # Custom view for visualization
└── config/
    └── UILabels.java                   # UI element labels configuration
```

## Setup Instructions

1. **Place Model File**: 
   - Copy your `ui_model_0.7.tflite` file to `app/src/main/assets/` directory

2. **Build the Project**:
   ```bash
   ./gradlew build
   ```

3. **Install on Device**:
   ```bash
   ./gradlew installDebug
   ```

## Usage

1. Launch the app
2. Click "Select Image" to choose an image from your device
3. Click "Test Detection" to run inference
4. View the detected UI elements with bounding boxes, labels, and confidence scores

## Configuration

### Model Parameters

You may need to adjust these parameters in `UIDetectionProcessor.java` based on your model:

- `INPUT_SIZE`: Model input image size (default: 640)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.5)
- `IOU_THRESHOLD`: IoU threshold for NMS (default: 0.5)
- `BOX_COUNT`: Number of output boxes from model (default: 8400)

### UI Labels

The app supports 19 UI element classes defined in `UILabels.java`:
- Background, Image, Icon, TEXT_INPUT, BUTTON, NAVIGATION_BAR, SLIDER, LIST, DIALOGUE, QRCODE, BARCODE, CONTAINER, PROGRESS_BAR, DATETIME, MAP, KEYBOARD, CALENDAR, WEBVIEW, GRID

## Dependencies

- Android SDK 24+
- TensorFlow Lite 2.14.0
- AndroidX libraries

## Notes

- The detection processor assumes a YOLO-style output format. You may need to adjust the post-processing logic based on your specific model architecture.
- The model input/output shapes may need to be adjusted based on your actual model.

