import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# Generate dummy dataset: 5 features
# [text_length, num_exclamation, num_caps, urgent_keyword_count, hour_of_day]
np.random.seed(42)
X_train = np.random.rand(1000, 5).astype(np.float32)
# Simple logic: high priority if feature_0 > 0.7 OR feature_2 > 0.8
y_train = ((X_train[:, 0] > 0.7) | (X_train[:, 2] > 0.8)).astype(np.int64)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y_train)

# Save scaler for Java to use (optional, but we'll just hardcode mean/std in Java for simplicity)
joblib.dump(scaler, 'scaler.pkl')
print(f"Scaler mean: {scaler.mean_}, Scale: {scaler.scale_}")

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 5]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

with open("priority_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
    
print("✅ ONNX model saved as 'priority_model.onnx'")