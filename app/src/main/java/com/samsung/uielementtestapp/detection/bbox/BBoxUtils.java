package com.samsung.uielementtestapp.detection.bbox;

import android.graphics.Bitmap;
import android.graphics.Rect;

public class BBoxUtils {
    public static Rect denormalizedBBox(float[] normalized_rect, Bitmap output_bitmap) {
        int count = 0;
        Rect rect = new Rect(
                (int) Math.ceil(normalized_rect[count++] * output_bitmap.getWidth()),
                (int) Math.ceil(normalized_rect[count++] * output_bitmap.getHeight()),
                (int) Math.ceil(normalized_rect[count++] * output_bitmap.getWidth()),
                (int) Math.ceil(normalized_rect[count++] * output_bitmap.getHeight())
        );

        return rect;
    }

    public static float[] normalizeBBox(Rect rect, Bitmap input_bitmap) {
        float[] normalized_values = new float[4];
        normalized_values[0] = rect.left / (float) input_bitmap.getWidth();
        normalized_values[1] = rect.top / (float) input_bitmap.getHeight();
        normalized_values[2] = rect.right / (float) input_bitmap.getWidth();
        normalized_values[3] = rect.bottom / (float) input_bitmap.getHeight();

        return normalized_values;
    }
}

