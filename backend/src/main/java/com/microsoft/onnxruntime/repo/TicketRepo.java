package com.microsoft.onnxruntime.repo;

import com.microsoft.onnxruntime.entity.TicketEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TicketRepo extends JpaRepository<TicketEntity, Long> {
}