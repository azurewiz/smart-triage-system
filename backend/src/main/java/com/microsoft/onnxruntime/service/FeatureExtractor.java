package com.microsoft.onnxruntime.service;

import java.time.LocalTime;

public class FeatureExtractor {

    public static float[] extractFeatures(String text) {
        float length = Math.min(text.length() / 100.0f, 1.0f);
        float exclam = Math.min(text.chars().filter(ch -> ch == '!').count() / 5.0f, 1.0f);
        float caps = Math.min(text.chars().filter(Character::isUpperCase).count() / 10.0f, 1.0f);
        float urgent = (text.toLowerCase().contains("urgent") ? 1 : 0) +
                       (text.toLowerCase().contains("asap") ? 1 : 0);
        float hour = (float) LocalTime.now().getHour() / 24.0f;
        return new float[]{length, exclam, caps, urgent, hour};
    }
}