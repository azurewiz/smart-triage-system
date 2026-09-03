# 🚀 Smart Triage System
**AI-powered support ticket priority classification microservice**

## Overview
This project bridges Data Science and Backend Engineering by deploying a scikit-learn classification model directly inside a Java Spring Boot application using **ONNX Runtime**. It eliminates the typical network overhead of calling a separate Python microservice, achieving sub-100ms inference latency.

## 📦 Tech Stack
- **Backend**: Java 17, Spring Boot 4.1.1
- **AI/ML**: Python, scikit-learn, ONNX Runtime 1.15.1
- **Database**: PostgreSQL 15
- **Containerization**: Docker, Docker Compose
- **Build Tools**: Maven (Java), pip (Python)

## ⚡ How It Works
1. **Feature Extraction**: The API extracts 5 heuristic features from raw text (urgency keywords, capitalization bursts, exclamation count, text length, and time of day).
2. **Inference**: The features are fed into an ONNX model loaded directly into the JVM. No inter-process communication means <50ms latency.
3. **Audit Logging**: Every prediction is persisted in PostgreSQL with a timestamp, enabling tracking of ticket volumes and priority distribution.

## 🛠️ Run Locally

### 1. Generate the ONNX Model (Optional)
If you want to retrain the model:
```bash
cd ml-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or just scikit-learn, pandas, skl2onnx
python train_model.py
cp priority_model.onnx ../backend/src/main/resources/



