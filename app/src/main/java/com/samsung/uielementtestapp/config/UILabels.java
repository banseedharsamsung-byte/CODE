package com.samsung.uielementtestapp.config;

import java.util.Arrays;
import java.util.List;

public class UILabels {
    public static final List<String> LABELS = Arrays.asList(
            "Background",
            "Image",
            "Icon",
            "TEXT_INPUT",
            "BUTTON",
            "NAVIGATION_BAR",
            "SLIDER",
            "LIST",
            "DIALOGUE",
            "QRCODE",
            "BARCODE",
            "CONTAINER",
            "PROGRESS_BAR",
            "DATETIME",
            "MAP",
            "KEYBOARD",
            "CALENDAR",
            "WEBVIEW",
            "GRID"
    );

    public static String getLabelName(int index) {
        if (index >= 0 && index < LABELS.size()) {
            return LABELS.get(index);
        }
        return "Unknown";
    }
}

