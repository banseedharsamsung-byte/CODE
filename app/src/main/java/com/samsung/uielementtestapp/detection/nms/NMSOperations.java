package com.samsung.uielementtestapp.detection.nms;

import android.graphics.RectF;

import com.samsung.uielementtestapp.detection.Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NMSOperations {
    private int[] shape;
    private float threshold;
    private float iouThreshold;

    public NMSOperations(int[] shape, float threshold, float iouThreshold) {
        this.shape = shape;
        this.threshold = threshold;
        this.iouThreshold = iouThreshold;
    }

    public List<Data> nmsAllClass(List<Data> detections) {
        if (detections == null || detections.isEmpty()) {
            return new ArrayList<>();
        }

        // Sort by confidence descending
        List<Data> sorted = new ArrayList<>(detections);
        Collections.sort(sorted, (a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

        List<Data> result = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            if (sorted.get(i).getConfidence() < threshold) continue;

            Data current = sorted.get(i);
            result.add(current);

            // Suppress overlapping boxes
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                if (calculateIoU(current.getRect(), sorted.get(j).getRect()) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    public List<Data> nmsPerClass(List<Data> detections) {
        if (detections == null || detections.isEmpty()) {
            return new ArrayList<>();
        }

        // Group by class
        Map<Integer, List<Data>> classMap = new HashMap<>();
        for (Data detection : detections) {
            if (detection.getConfidence() >= threshold) {
                int label = detection.getLabelName();
                classMap.putIfAbsent(label, new ArrayList<>());
                classMap.get(label).add(detection);
            }
        }

        List<Data> result = new ArrayList<>();

        // Apply NMS per class
        for (List<Data> classDetections : classMap.values()) {
            Collections.sort(classDetections, (a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

            boolean[] suppressed = new boolean[classDetections.size()];

            for (int i = 0; i < classDetections.size(); i++) {
                if (suppressed[i]) continue;

                Data current = classDetections.get(i);
                result.add(current);

                for (int j = i + 1; j < classDetections.size(); j++) {
                    if (suppressed[j]) continue;
                    if (calculateIoU(current.getRect(), classDetections.get(j).getRect()) > iouThreshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return result;
    }

    private float calculateIoU(RectF box1, RectF box2) {
        float left = Math.max(box1.left, box2.left);
        float top = Math.max(box1.top, box2.top);
        float right = Math.min(box1.right, box2.right);
        float bottom = Math.min(box1.bottom, box2.bottom);

        if (right < left || bottom < top) {
            return 0.0f;
        }

        float intersection = (right - left) * (bottom - top);
        float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
        float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);
        float union = area1 + area2 - intersection;

        return union > 0 ? intersection / union : 0.0f;
    }
}

