package com.microsoft.onnxruntime.controller;

import com.microsoft.onnxruntime.entity.TicketEntity;
import com.microsoft.onnxruntime.repo.TicketRepo;
import com.microsoft.onnxruntime.service.OnnxInferenceService;
import com.microsoft.onnxruntime.service.FeatureExtractor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import ai.onnxruntime.OrtException;

import java.time.LocalTime;
import java.util.Map;

@RestController
@RequestMapping("/api/triage")
public class TriageController {

    @Autowired
    private OnnxInferenceService onnxService;

    @Autowired
    private TicketRepo repo;

    @PostMapping
    public Map<String, Object> triage(@RequestBody Map<String, String> payload) throws OrtException {
        String text = payload.get("text");

        float length = Math.min(text.length() / 100.0f, 1.0f);
        float exclam = text.chars().filter(ch -> ch == '!').count() / 5.0f;
        float caps = text.chars().filter(Character::isUpperCase).count() / 10.0f;
        float urgent = (text.toLowerCase().contains("urgent") ? 1 : 0) +
                       (text.toLowerCase().contains("asap") ? 1 : 0);
        float hour = (float) LocalTime.now().getHour() / 24.0f;

       
        float[] features = FeatureExtractor.extractFeatures(text);
        double score = onnxService.predict(features);
        String priority = score > 0.5 ? "HIGH" : "LOW";

        TicketEntity entity = new TicketEntity();
        entity.setCustomerText(text);
        entity.setPriorityScore(score);
        entity.setPredictedPriority(priority);
        repo.save(entity);

        return Map.of("priority", priority, "score", score, "message", "Ticket logged successfully");
    }
}