package com.samsung.uielementtestapp.detection;

import android.graphics.RectF;

public class Data {
    float labelId;
    int labelName;
    float confidence;
    RectF rect;

    public Data(float labelId, int labelName, float confidence, RectF r) {
        this.confidence = confidence;
        this.labelId = labelId;
        this.labelName = labelName;
        this.rect = r;
    }

    public float getConfidence() {
        return confidence;
    }

    public int getLabelName() {
        return labelName;
    }

    public float getLabelId() {
        return labelId;
    }

    public RectF getRect() {
        return rect;
    }
}

