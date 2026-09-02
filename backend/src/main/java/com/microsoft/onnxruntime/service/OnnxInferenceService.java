package com.microsoft.onnxruntime.service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.io.InputStream;

@Service
public class OnnxInferenceService {

    private OrtEnvironment environment;
    private OrtSession session;

    @PostConstruct
    public void init() throws OrtException {
        environment = OrtEnvironment.getEnvironment();
        try (InputStream modelStream = getClass().getResourceAsStream("/priority_model.onnx")) {
            if (modelStream == null) {
                throw new RuntimeException("ONNX model not found in resources!");
            }
            session = environment.createSession(modelStream.readAllBytes());
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    public double predict(float[] features) throws OrtException {
        OnnxTensor tensor = OnnxTensor.createTensor(environment, new float[][]{features});
        try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("float_input", tensor))) {
            Object output = result.get(0).getValue();
            
            // Handle both probability (float[][]) and class (long[]) outputs
            if (output instanceof float[][]) {
                float[][] probs = (float[][]) output;
                return probs[0][1]; // probability of class 1 (HIGH)
            } else if (output instanceof long[]) {
                long[] classes = (long[]) output;
                return classes[0]; // 0 or 1
            } else {
                throw new RuntimeException("Unexpected output type: " + output.getClass().getName());
            }
        }
    }

    @PreDestroy
    public void close() throws OrtException {
        if (session != null) session.close();
        if (environment != null) environment.close();
    }
}