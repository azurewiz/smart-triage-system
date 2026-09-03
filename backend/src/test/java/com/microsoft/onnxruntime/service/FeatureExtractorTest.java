package com.microsoft.onnxruntime.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FeatureExtractorTest {

    @Test
    public void testHighUrgency() {
        String text = "URGENT! System is down ASAP!!!";
        float[] features = FeatureExtractor.extractFeatures(text);

        assertEquals(0.3f, features[0], 0.001f, "Length mismatch");
        assertEquals(0.8f, features[1], 0.001f, "Exclamation mismatch");  // 4 '!'
        assertEquals(1.0f, features[2], 0.001f, "Caps should be capped at 1.0");
        assertEquals(2.0f, features[3], 0.001f, "Urgency score should be 2");
        assertTrue(features[4] >= 0 && features[4] <= 1.0f, "Hour out of range");
    }

    @Test
    public void testLowUrgency() {
        String text = "Hello, I just wanted to ask about my account balance.";
        float[] features = FeatureExtractor.extractFeatures(text);

        assertTrue(features[0] > 0, "Length should be > 0");
        assertEquals(0.0f, features[1], 0.001f, "No exclamations");
        assertEquals(0.2f, features[2], 0.001f, "Caps count should be 2 (H and I) => 0.2");
        assertEquals(0.0f, features[3], 0.001f, "No urgency keywords");
        assertTrue(features[4] >= 0 && features[4] <= 1.0f, "Hour out of range");
    }

    @Test
    public void testCapsCapping() {
        String text = "AAAAA BBBBB CCCCC";
        float[] features = FeatureExtractor.extractFeatures(text);
        assertEquals(1.0f, features[2], 0.001f, "Caps should be capped at 1.0");
    }

    @Test
    public void testEmptyString() {
        String text = "";
        float[] features = FeatureExtractor.extractFeatures(text);
        assertEquals(0.0f, features[0]);
        assertEquals(0.0f, features[1]);
        assertEquals(0.0f, features[2]);
        assertEquals(0.0f, features[3]);
        assertTrue(features[4] >= 0 && features[4] <= 1.0f);
    }

    @Test
    public void testFeatureArrayLength() {
        float[] features = FeatureExtractor.extractFeatures("test");
        assertEquals(5, features.length, "Should return exactly 5 features");
    }
}