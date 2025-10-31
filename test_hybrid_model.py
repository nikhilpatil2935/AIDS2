"""
Test script for hybrid_model.h5
This model requires 2 inputs: tabular features (8) and text embeddings (50)
"""

import numpy as np
import tensorflow as tf
import joblib
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the hybrid model
print("\n1. Loading hybrid model...")
model = tf.keras.models.load_model('hybrid_model.h5')
print(f"   ✓ Model loaded successfully")
print(f"   Input shapes: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Load the scaler
print("\n2. Loading scaler...")
scaler = joblib.load('tabular_scaler.joblib')
print(f"   ✓ Scaler loaded (expects {scaler.n_features_in_} features)")

# Create sample tabular data (8 features)
print("\n3. Creating sample tabular input...")
# These would be: CGPA, Prev_Sem, Acad_Perf, IQ, Projects, Internship, Comm, Extra
sample_tabular = np.array([[
    8.5,   # CGPA
    8.0,   # Prev Sem Result
    8,     # Academic Performance
    115,   # IQ
    3,     # Projects
    1,     # Internship (1=Yes)
    8,     # Communication Skills
    7      # Extra Curricular
]])

print(f"   Raw tabular data shape: {sample_tabular.shape}")
print(f"   Values: {sample_tabular[0]}")

# Scale the tabular data
tabular_scaled = scaler.transform(sample_tabular)
print(f"   Scaled tabular data shape: {tabular_scaled.shape}")

# Create sample text embedding (50 features) - random for demo
print("\n4. Creating sample text embedding...")
# In a real scenario, this would come from encoding resume text
sample_embedding = np.random.randn(1, 50) * 0.1  # Small random values
print(f"   Text embedding shape: {sample_embedding.shape}")

# Make prediction
print("\n5. Making prediction with hybrid model...")
try:
    prediction = model.predict([tabular_scaled, sample_embedding], verbose=0)
    print(f"   ✓ Prediction successful!")
    print(f"   Output shape: {prediction.shape}")
    print(f"   Raw output: {prediction[0][0]:.4f}")
    
    # Interpret result
    prob = float(prediction[0][0])
    result = "PLACED" if prob >= 0.5 else "NOT PLACED"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    
    print(f"\n   RESULT: {result}")
    print(f"   Confidence: {confidence:.2f}%")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("EXPLANATION")
print("=" * 70)
print("""
The hybrid model architecture requires:
  Input 1: 8 tabular features (numerical data) - SCALED
  Input 2: 50 text embedding features (from resume text encoding)

This is why it cannot be used in the Streamlit app with just the 
12 engineered features. It was likely trained on a dataset that 
included both structured data AND text data (like resume descriptions).

To use this model in production, you would need to:
1. Extract text from resume (✓ already implemented)
2. Encode the text using a text embedding model (e.g., BERT, Word2Vec)
3. Generate a 50-dimensional embedding vector
4. Pass both tabular + embedding to the model

For now, use Random Forest or AdaBoost models which work perfectly
with the current feature engineering!
""")
print("=" * 70)
