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
            // Wrap any checked exception (IO, etc.) into an unchecked RuntimeException
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    public double predict(float[] features) throws OrtException {
        OnnxTensor tensor = OnnxTensor.createTensor(environment, new float[][]{features});
        try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("float_input", tensor))) {
            float[][] output = (float[][]) result.get(0).getValue();
            return output[0][1];
        }
    }

    @PreDestroy
    public void close() throws OrtException {
        if (session != null) session.close();
        if (environment != null) environment.close();
    }
}