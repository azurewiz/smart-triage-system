import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import datetime

# =============================================
# 1. FEATURE EXTRACTION LOGIC (Now testable!)
# =============================================
def extract_features(text: str) -> list:
    """
    Extracts 5 heuristic features from raw customer text.
    Matches the Java implementation in TriageController.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Normalized text length (capped at 100 chars)
    length = min(len(text) / 100.0, 1.0)
    
    # 2. Exclamation count (capped at 5)
    exclam = min(text.count('!') / 5.0, 1.0)
    
    # 3. Capitalization burst (capped at 10 caps)
    caps = min(sum(1 for c in text if c.isupper()) / 10.0, 1.0)
    
    # 4. Urgency keyword score (0, 1, or 2)
    urgent = (1 if 'urgent' in text.lower() else 0) + \
             (1 if 'asap' in text.lower() else 0)
    
    # 5. Time of day (0 to 1)
    hour = datetime.datetime.now().hour / 24.0
    
    return [length, exclam, caps, urgent, hour]


# =============================================
# 2. MODEL TRAINING (Uses random data)
# =============================================
def train_and_export_model():
    # Generate dummy dataset: 5 features
    np.random.seed(42)
    X_train = np.random.rand(1000, 5).astype(np.float32)
    y_train = ((X_train[:, 0] > 0.7) | (X_train[:, 2] > 0.8)).astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_scaled, y_train)

    joblib.dump(scaler, 'scaler.pkl')
    print(f"✅ Scaler saved. Mean: {scaler.mean_}, Scale: {scaler.scale_}")

    # Convert to ONNX with target_opset=12
    initial_type = [('float_input', FloatTensorType([None, 5]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

    with open("priority_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print("✅ ONNX model saved as 'priority_model.onnx'")

if __name__ == "__main__":
    train_and_export_model()