package com.samsung.uielementtestapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.samsung.uielementtestapp.core.ModelUtility;
import com.samsung.uielementtestapp.detection.Data;
import com.samsung.uielementtestapp.detection.UIDetectionProcessor;
import com.samsung.uielementtestapp.view.BoundingBoxView;

import org.tensorflow.lite.Interpreter;

import java.io.InputStream;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int PICK_IMAGE_REQUEST = 1001;
    private static final String MODEL_NAME = "ui_model_0.7.tflite";
    private static final int LABEL_COUNT = 19;
    private static final int BOX_COUNT = 8400; // Common YOLO default, adjust based on your model

    private Button btnSelectImage;
    private Button btnTestDetection;
    private BoundingBoxView imageView;
    private TextView tvStatus;

    private Bitmap selectedBitmap;
    private Interpreter interpreter;
    private UIDetectionProcessor detectionProcessor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        loadModel();
        setupListeners();
    }

    private void initViews() {
        btnSelectImage = findViewById(R.id.btnSelectImage);
        btnTestDetection = findViewById(R.id.btnTestDetection);
        imageView = findViewById(R.id.imageView);
        tvStatus = findViewById(R.id.tvStatus);
    }

    private void loadModel() {
        new Thread(() -> {
            try {
                interpreter = ModelUtility.getInterpreterFromModel(
                        this,
                        MODEL_NAME,
                        ModelUtility.INTERPRETER_DELEGATE.CPU
                );

                if (interpreter != null) {
                    detectionProcessor = new UIDetectionProcessor(interpreter, LABEL_COUNT, BOX_COUNT);
                    runOnUiThread(() -> {
                        tvStatus.setText("Model loaded successfully");
                        Toast.makeText(this, "Model loaded", Toast.LENGTH_SHORT).show();
                    });
                } else {
                    runOnUiThread(() -> {
                        tvStatus.setText("Failed to load model. Place ui_model_0.7.tflite in assets folder.");
                        Toast.makeText(this, "Model loading failed", Toast.LENGTH_LONG).show();
                    });
                }
            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvStatus.setText("Error loading model: " + e.getMessage());
                    Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private void setupListeners() {
        btnSelectImage.setOnClickListener(v -> openImagePicker());

        btnTestDetection.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
                return;
            }

            if (detectionProcessor == null) {
                Toast.makeText(this, "Model not loaded yet", Toast.LENGTH_SHORT).show();
                return;
            }

            runDetection();
        });
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Select Image"), PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            if (imageUri != null) {
                try {
                    InputStream inputStream = getContentResolver().openInputStream(imageUri);
                    selectedBitmap = BitmapFactory.decodeStream(inputStream);
                    inputStream.close();

                    if (selectedBitmap != null) {
                        imageView.setBitmap(selectedBitmap);
                        imageView.setDetections(new java.util.ArrayList<>());
                        btnTestDetection.setEnabled(true);
                        tvStatus.setText("Image loaded. Click 'Test Detection' to detect UI elements.");
                    } else {
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                    }
                } catch (Exception e) {
                    Toast.makeText(this, "Error loading image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private void runDetection() {
        btnTestDetection.setEnabled(false);
        tvStatus.setText("Detecting...");

        new Thread(() -> {
            try {
                List<Data> detections = detectionProcessor.detect(selectedBitmap);

                new Handler(Looper.getMainLooper()).post(() -> {
                    imageView.setDetections(detections);
                    tvStatus.setText(String.format("Detection Complete: %d elements found", detections.size()));
                    btnTestDetection.setEnabled(true);
                });
            } catch (Exception e) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    tvStatus.setText("Detection failed: " + e.getMessage());
                    btnTestDetection.setEnabled(true);
                    Toast.makeText(this, "Detection error: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
        }
    }
}

