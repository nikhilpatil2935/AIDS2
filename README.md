# Student Placement Predictor (Professional)

This Streamlit app provides a professional UI for predicting student placements using multiple models (Random Forest, AdaBoost, and a hybrid Keras model).

## ✨ NEW: Resume Scanner Feature
Upload your resume (PDF or TXT) and the app will automatically extract:
- CGPA/GPA and academic performance
- Number of projects completed
- Internship experience
- Extra-curricular activities
- Communication skills (estimated from resume quality)

All extracted values are auto-filled in the form and can be manually adjusted before prediction!

## ✅ Working Models
- **Random Forest** (`rf_model_engineered.joblib`) — ✓ Fully functional
- **AdaBoost** (`ada_model_engineered.joblib`) — ✓ Fully functional

## ⚠️ Hybrid Model Note
- **Hybrid NN** (`hybrid_model.h5`) — This model requires **multiple inputs**:
  - Tabular input: 8 numerical features (scaled)
  - Text embedding: 50 features from text data
  
  The current UI only provides the 12 engineered features, so this model cannot make predictions. Use Random Forest or AdaBoost instead.

## Features
- **Multi-model selector** in the sidebar (auto-loads joblib/.h5 models when available)
- **Clean, professional layout** with two-column design for inputs and detailed output
- **Per-prediction insights**:
  - Engineered features shown to explain what the model saw
  - Feature importance chart for tree-based models
  - Confidence scores with visual indicators
- **CSV batch upload** for batch predictions
- **Persistent prediction history** stored in `predictions_history.csv` with download option
- **Error handling** with detailed debug information when predictions fail

## Quick start (Windows PowerShell)

1. (Optional) Create and activate a virtual environment.

2. Install required packages:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run professional_app.py
```

4. Open your browser to the URL shown (usually http://localhost:8501)

## Using the App

### Method 1: Upload Resume (Recommended)
1. **Select a model** in the sidebar (Random Forest or AdaBoost recommended)
2. **Upload your resume** in PDF or TXT format
3. **Review extracted information** - the app will show what it found
4. **Verify/adjust the auto-filled values** in the sliders below
5. **Click "Predict Placement"** to see results

### Method 2: Manual Input
1. **Select a model** in the sidebar (Random Forest or AdaBoost recommended)
2. **Adjust input sliders** for the student profile:
   - CGPA, Academic Performance, Previous Semester Result
   - IQ, Projects Completed, Internship Experience
   - Communication Skills, Extra Curricular Score
3. **Click "Predict Placement"** to see results

### View Insights
After prediction, you'll see:
- Prediction result (PLACED / NOT PLACED)
- Confidence percentage
- Feature importance (what matters most)
- Engineered features (how input was transformed)

## Batch Predictions

1. Enable "CSV batch upload" in the sidebar
2. Upload a CSV with these columns:
   ```
   cgpa, academic_performance, prev_sem_result, iq, projects, 
   internship, comm_skills, extra_curricular
   ```
3. Results will be shown in a table and saved to history

## Files

- `professional_app.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `rf_model_engineered.joblib` — Random Forest model
- `ada_model_engineered.joblib` — AdaBoost model
- `hybrid_model.h5` — Hybrid Keras model (requires additional features)
- `tabular_scaler.joblib` — Scaler for hybrid model
- `predictions_history.csv` — Created at runtime to store prediction history

## Troubleshooting

**Error: "Model could not produce a prediction"**
- If using Hybrid NN model, switch to Random Forest or AdaBoost
- Check the debug information expander for detailed error messages
- Ensure all input values are within valid ranges

**No history showing**
- Make at least one prediction to create the history file
- History persists across sessions

**TensorFlow warnings**
- Info messages about oneDNN and CPU optimizations are normal and don't affect functionality
- Keras compile_metrics warnings are expected for pre-trained loaded models

## Technical Details

**Feature Engineering:**
The app automatically transforms raw inputs into 12 engineered features:
- 6 numerical features (kept as-is)
- 3 CGPA category bins (Low < 7.0 < Medium < 8.5 < High)
- 3 IQ category bins (Below < 90 <= Average <= 110 < Above)

**Model Expectations:**
- Random Forest: 12 features, binary classification
- AdaBoost: 12 features, binary classification
- Hybrid NN: 2 inputs (8 tabular + 50 embedding), NOT compatible with current UI