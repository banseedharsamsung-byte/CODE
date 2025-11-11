package com.samsung.uielementtestapp.detection;

import android.graphics.Bitmap;
import android.graphics.RectF;

import com.samsung.uielementtestapp.config.UILabels;
import com.samsung.uielementtestapp.detection.bbox.BBoxUtils;
import com.samsung.uielementtestapp.detection.nms.NMSOperations;

import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class UIDetectionProcessor {
    private static final String TAG = "UIDetectionProcessor";
    private static final int BATCH_SIZE = 1;
    private static final int INPUT_SIZE = 640; // Common YOLO input size, adjust if needed
    private static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final float IOU_THRESHOLD = 0.5f;

    private Interpreter interpreter;
    private int labelCount;
    private int boxCount;

    public UIDetectionProcessor(Interpreter interpreter, int labelCount, int boxCount) {
        this.interpreter = interpreter;
        this.labelCount = labelCount;
        this.boxCount = boxCount;
    }

    public List<Data> detect(Bitmap inputBitmap) {
        if (interpreter == null || inputBitmap == null) {
            return new ArrayList<>();
        }

        try {
            // Preprocess image
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, INPUT_SIZE, INPUT_SIZE, true);
            ByteBuffer inputBuffer = preprocessBitmap(resizedBitmap);

            // Get output shape from model
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            
            // Allocate output buffer based on actual model output shape
            // Common formats: [1, boxes, 5+classes] or [1, boxes, classes+5]
            // Adjust based on your model's actual output format
            Object output;
            if (outputShape.length == 3) {
                output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
            } else if (outputShape.length == 2) {
                output = new float[outputShape[0]][outputShape[1]];
            } else {
                // Fallback to expected shape
                output = new float[1][boxCount][5 + labelCount];
            }

            // Run inference
            interpreter.run(inputBuffer, output);

            // Post-process results
            List<Data> detections;
            if (output instanceof float[][][]) {
                detections = postProcess((float[][][]) output, inputBitmap.getWidth(), inputBitmap.getHeight());
            } else if (output instanceof float[][]) {
                detections = postProcess2D((float[][]) output, inputBitmap.getWidth(), inputBitmap.getHeight());
            } else {
                detections = new ArrayList<>();
            }

            // Apply NMS
            NMSOperations nms = new NMSOperations(
                    new int[]{BATCH_SIZE, labelCount, boxCount},
                    CONFIDENCE_THRESHOLD,
                    IOU_THRESHOLD
            );

            return nms.nmsPerClass(detections);
        } catch (Exception e) {
            android.util.Log.e(TAG, "Detection failed: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    private ByteBuffer preprocessBitmap(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                // Normalize to [0, 1]
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

    private List<Data> postProcess(float[][][] output, int imageWidth, int imageHeight) {
        List<Data> detections = new ArrayList<>();
        
        if (output.length == 0 || output[0].length == 0) {
            return detections;
        }

        float[][] boxes = output[0]; // Get first batch

        for (float[] box : boxes) {
            if (box.length < 5 + labelCount) {
                continue;
            }

            // Assuming output format: [x_center, y_center, width, height, objectness, ...class_scores]
            // OR: [x1, y1, x2, y2, conf, ...class_scores]
            // Adjust based on your model's actual output format
            float xCenter = box[0];
            float yCenter = box[1];
            float width = box[2];
            float height = box[3];
            float objectness = box[4];

            if (objectness < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // Find class with highest score
            int bestClass = 0;
            float bestScore = box[5];
            for (int i = 1; i < labelCount; i++) {
                if (box[5 + i] > bestScore) {
                    bestScore = box[5 + i];
                    bestClass = i;
                }
            }

            float confidence = objectness * bestScore;
            if (confidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // Convert from center format to corner format (normalized coordinates)
            float left = (xCenter - width / 2) * imageWidth;
            float top = (yCenter - height / 2) * imageHeight;
            float right = (xCenter + width / 2) * imageWidth;
            float bottom = (yCenter + height / 2) * imageHeight;

            // Clamp to image bounds
            left = Math.max(0, Math.min(left, imageWidth));
            top = Math.max(0, Math.min(top, imageHeight));
            right = Math.max(0, Math.min(right, imageWidth));
            bottom = Math.max(0, Math.min(bottom, imageHeight));

            RectF rect = new RectF(left, top, right, bottom);
            detections.add(new Data(bestClass, bestClass, confidence, rect));
        }

        return detections;
    }

    private List<Data> postProcess2D(float[][] output, int imageWidth, int imageHeight) {
        List<Data> detections = new ArrayList<>();

        for (float[] box : output) {
            if (box.length < 5 + labelCount) {
                continue;
            }

            // Same processing as 3D version
            float xCenter = box[0];
            float yCenter = box[1];
            float width = box[2];
            float height = box[3];
            float objectness = box[4];

            if (objectness < CONFIDENCE_THRESHOLD) {
                continue;
            }

            int bestClass = 0;
            float bestScore = box[5];
            for (int i = 1; i < labelCount; i++) {
                if (box[5 + i] > bestScore) {
                    bestScore = box[5 + i];
                    bestClass = i;
                }
            }

            float confidence = objectness * bestScore;
            if (confidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            float left = (xCenter - width / 2) * imageWidth;
            float top = (yCenter - height / 2) * imageHeight;
            float right = (xCenter + width / 2) * imageWidth;
            float bottom = (yCenter + height / 2) * imageHeight;

            left = Math.max(0, Math.min(left, imageWidth));
            top = Math.max(0, Math.min(top, imageHeight));
            right = Math.max(0, Math.min(right, imageWidth));
            bottom = Math.max(0, Math.min(bottom, imageHeight));

            RectF rect = new RectF(left, top, right, bottom);
            detections.add(new Data(bestClass, bestClass, confidence, rect));
        }

        return detections;
    }
}

