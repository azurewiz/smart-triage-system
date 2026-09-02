package com.microsoft.onnxruntime.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
public class TicketEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(columnDefinition = "TEXT")
    private String customerText;

    private Double priorityScore;
    private String predictedPriority; // "HIGH" or "LOW"
    private LocalDateTime timestamp = LocalDateTime.now();
}