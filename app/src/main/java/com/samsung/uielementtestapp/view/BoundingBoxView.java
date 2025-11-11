package com.samsung.uielementtestapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.samsung.uielementtestapp.config.UILabels;
import com.samsung.uielementtestapp.detection.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BoundingBoxView extends View {
    private Bitmap bitmap;
    private List<Data> detections = new ArrayList<>();
    private Paint boxPaint;
    private Paint textPaint;
    private Paint textBgPaint;
    private float scaleX = 1.0f;
    private float scaleY = 1.0f;
    private float offsetX = 0.0f;
    private float offsetY = 0.0f;
    
    // Color map for different UI element classes
    private static final Map<Integer, Integer> COLOR_MAP = new HashMap<>();
    
    static {
        // Assign distinct colors for each UI element class
        COLOR_MAP.put(0, Color.argb(255, 128, 128, 128));  // Background - Gray
        COLOR_MAP.put(1, Color.argb(255, 255, 0, 0));     // Image - Red
        COLOR_MAP.put(2, Color.argb(255, 0, 255, 0));     // Icon - Green
        COLOR_MAP.put(3, Color.argb(255, 0, 0, 255));     // TEXT_INPUT - Blue
        COLOR_MAP.put(4, Color.argb(255, 255, 165, 0));   // BUTTON - Orange
        COLOR_MAP.put(5, Color.argb(255, 255, 0, 255));  // NAVIGATION_BAR - Magenta
        COLOR_MAP.put(6, Color.argb(255, 0, 255, 255));  // SLIDER - Cyan
        COLOR_MAP.put(7, Color.argb(255, 255, 192, 203)); // LIST - Pink
        COLOR_MAP.put(8, Color.argb(255, 128, 0, 128));  // DIALOGUE - Purple
        COLOR_MAP.put(9, Color.argb(255, 255, 255, 0));  // QRCODE - Yellow
        COLOR_MAP.put(10, Color.argb(255, 0, 128, 0));   // BARCODE - Dark Green
        COLOR_MAP.put(11, Color.argb(255, 255, 140, 0)); // CONTAINER - Dark Orange
        COLOR_MAP.put(12, Color.argb(255, 0, 191, 255)); // PROGRESS_BAR - Deep Sky Blue
        COLOR_MAP.put(13, Color.argb(255, 148, 0, 211)); // DATETIME - Dark Violet
        COLOR_MAP.put(14, Color.argb(255, 34, 139, 34)); // MAP - Forest Green
        COLOR_MAP.put(15, Color.argb(255, 255, 20, 147)); // KEYBOARD - Deep Pink
        COLOR_MAP.put(16, Color.argb(255, 70, 130, 180)); // CALENDAR - Steel Blue
        COLOR_MAP.put(17, Color.argb(255, 255, 69, 0));  // WEBVIEW - Red Orange
        COLOR_MAP.put(18, Color.argb(255, 50, 205, 50)); // GRID - Lime Green
    }

    public BoundingBoxView(Context context) {
        super(context);
        init();
    }

    public BoundingBoxView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public BoundingBoxView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5.0f); // Slightly thicker for better visibility
        boxPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(36.0f); // Slightly larger text
        textPaint.setAntiAlias(true);
        textPaint.setFakeBoldText(true); // Bold text for better readability

        textBgPaint = new Paint();
        textBgPaint.setStyle(Paint.Style.FILL);
    }
    
    private int getColorForLabel(int labelIndex) {
        return COLOR_MAP.getOrDefault(labelIndex, Color.WHITE);
    }

    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
        calculateScale();
        invalidate();
    }

    public void setDetections(List<Data> detections) {
        this.detections = detections != null ? new ArrayList<>(detections) : new ArrayList<>();
        invalidate();
    }

    private void calculateScale() {
        if (bitmap == null || getWidth() == 0 || getHeight() == 0) {
            return;
        }

        float viewWidth = getWidth();
        float viewHeight = getHeight();
        float bitmapWidth = bitmap.getWidth();
        float bitmapHeight = bitmap.getHeight();

        float scale = Math.min(viewWidth / bitmapWidth, viewHeight / bitmapHeight);
        scaleX = scale;
        scaleY = scale;

        offsetX = (viewWidth - bitmapWidth * scale) / 2.0f;
        offsetY = (viewHeight - bitmapHeight * scale) / 2.0f;
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        calculateScale();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (bitmap == null) {
            return;
        }

        canvas.save();
        canvas.translate(offsetX, offsetY);
        canvas.scale(scaleX, scaleY);
        canvas.drawBitmap(bitmap, 0, 0, null);
        canvas.restore();

        // Draw bounding boxes with different colors for each UI element
        for (Data detection : detections) {
            RectF rect = detection.getRect();
            RectF scaledRect = new RectF(
                    rect.left * scaleX + offsetX,
                    rect.top * scaleY + offsetY,
                    rect.right * scaleX + offsetX,
                    rect.bottom * scaleY + offsetY
            );

            // Get color for this UI element class
            int labelIndex = detection.getLabelName();
            int boxColor = getColorForLabel(labelIndex);
            boxPaint.setColor(boxColor);

            // Draw bounding box
            canvas.drawRect(scaledRect, boxPaint);

            // Prepare label and confidence text
            String label = UILabels.getLabelName(labelIndex);
            float confidence = detection.getConfidence();
            String text = String.format("%s: %.2f", label, confidence);

            // Calculate text dimensions
            float textWidth = textPaint.measureText(text);
            float textHeight = textPaint.getTextSize();
            float padding = 10.0f;

            // Draw text background with semi-transparent overlay
            // Use a darker version of the box color for text background
            int textBgColor = Color.argb(220, 
                    Color.red(boxColor), 
                    Color.green(boxColor), 
                    Color.blue(boxColor));
            textBgPaint.setColor(textBgColor);
            
            RectF textBg = new RectF(
                    scaledRect.left,
                    scaledRect.top - textHeight - padding * 2,
                    scaledRect.left + textWidth + padding * 2,
                    scaledRect.top
            );
            
            // Ensure text background doesn't go off-screen
            if (textBg.top < 0) {
                textBg.top = scaledRect.top;
                textBg.bottom = scaledRect.top + textHeight + padding * 2;
            }
            
            canvas.drawRect(textBg, textBgPaint);

            // Draw label and confidence text
            float textX = scaledRect.left + padding;
            float textY = scaledRect.top - padding;
            if (textY < textHeight) {
                textY = scaledRect.top + textHeight + padding;
            }
            canvas.drawText(text, textX, textY, textPaint);
        }
    }
}

