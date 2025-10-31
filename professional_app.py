import os
import json
import io
import re
from datetime import datetime, timezone
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Optional: PDF parsing for resume upload
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Optional: keras for .h5 models. If not available, Keras-based models won't be loadable.
try:
    # Suppress TensorFlow warnings before import
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # Use importlib to avoid static import errors in environments where tensorflow is not installed
    import importlib

    keras_load_model = None
    # Try tensorflow.keras first, then fall back to standalone keras
    try:
        keras_models = importlib.import_module("tensorflow.keras.models")
    except Exception:
        try:
            keras_models = importlib.import_module("keras.models")
        except Exception:
            keras_models = None

    if keras_models is not None and hasattr(keras_models, "load_model"):
        keras_load_model = keras_models.load_model
    else:
        keras_load_model = None
except Exception:
    keras_load_model = None


# -------------------------
# Config / Helpers
# -------------------------
APP_TITLE = "🎓 Student Placement Predictor — Professional"
HISTORY_FILE = "predictions_history.csv"
AVAILABLE_MODELS = {
    "Random Forest (rf_model_engineered) — ✓ Recommended": "rf_model_engineered.joblib",
    "AdaBoost (ada_model_engineered) — ✓ Recommended": "ada_model_engineered.joblib",
    "Hybrid NN (hybrid_model) — ⚠️ Needs extra features": "hybrid_model.h5",
}


def load_model_by_path(path: str):
    """Load a model by file extension. Returns (model, meta_dict)."""
    if not os.path.exists(path):
        return None, {"error": "file_not_found"}

    if path.endswith(".joblib"):
        m = joblib.load(path)
        meta = {"type": "sklearn", "path": path}
        return m, meta

    if path.endswith(".h5"):
        if keras_load_model is None:
            return None, {"error": "keras_not_available"}
        m = keras_load_model(path)
        meta = {"type": "keras", "path": path}
        return m, meta

    return None, {"error": "unsupported_format"}


def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["timestamp", "model", "inputs", "engineered", "prediction", "confidence"])
        df.to_csv(HISTORY_FILE, index=False)


def append_history(entry: dict):
    ensure_history_file()
    df = pd.DataFrame([entry])
    df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)


def engineer_features(raw_data: dict) -> pd.DataFrame:
    """Convert raw inputs (dictionary) into engineered DataFrame expected by models.

    Returns a one-row DataFrame.
    """
    feature_names = [
        "Prev_Sem_Result",
        "Academic_Performance",
        "Internship_Experience",
        "Extra_Curricular_Score",
        "Communication_Skills",
        "Projects_Completed",
        "CGPA_Category_CGPA_Low",
        "CGPA_Category_CGPA_Medium",
        "CGPA_Category_CGPA_High",
        "IQ_Category_IQ_Below_Average",
        "IQ_Category_IQ_Average",
        "IQ_Category_IQ_Above_Average",
    ]

    row = dict.fromkeys(feature_names, 0)

    # Base numeric features
    row["Prev_Sem_Result"] = raw_data.get("prev_sem_result", 0)
    row["Academic_Performance"] = raw_data.get("academic_performance", 0)
    row["Internship_Experience"] = 1 if raw_data.get("internship", "No") == "Yes" else 0
    row["Extra_Curricular_Score"] = raw_data.get("extra_curricular", 0)
    row["Communication_Skills"] = raw_data.get("comm_skills", 0)
    row["Projects_Completed"] = raw_data.get("projects", 0)

    # CGPA bins
    cgpa = raw_data.get("cgpa", 0)
    if cgpa < 7.0:
        row["CGPA_Category_CGPA_Low"] = 1
    elif cgpa < 8.5:
        row["CGPA_Category_CGPA_Medium"] = 1
    else:
        row["CGPA_Category_CGPA_High"] = 1

    # IQ bins
    iq = raw_data.get("iq", 0)
    if iq < 90:
        row["IQ_Category_IQ_Below_Average"] = 1
    elif iq <= 110:
        row["IQ_Category_IQ_Average"] = 1
    else:
        row["IQ_Category_IQ_Above_Average"] = 1

    return pd.DataFrame([row])


def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    if PyPDF2 is None:
        return None, "PyPDF2 not installed. Install with: pip install PyPDF2"
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text, None
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"


def parse_resume(text: str) -> dict:
    """Parse resume text and extract relevant information.
    
    Returns a dictionary with extracted values and confidence scores.
    """
    extracted = {
        "cgpa": None,
        "academic_performance": None,
        "prev_sem_result": None,
        "iq": None,
        "projects": None,
        "internship": None,
        "comm_skills": None,
        "extra_curricular": None,
        "confidence": {},
        "raw_findings": []
    }
    
    text_lower = text.lower()
    
    # Extract CGPA/GPA (patterns like "CGPA: 8.5", "GPA 3.5/4.0", "8.5 CGPA")
    cgpa_patterns = [
        r'cgpa[:\s]+([0-9]\.[0-9]{1,2})',
        r'gpa[:\s]+([0-9]\.[0-9]{1,2})',
        r'([0-9]\.[0-9]{1,2})\s*cgpa',
        r'([0-9]\.[0-9]{1,2})\s*/\s*10',
    ]
    for pattern in cgpa_patterns:
        match = re.search(pattern, text_lower)
        if match:
            cgpa_val = float(match.group(1))
            # Normalize if needed (assume 4.0 scale if < 5)
            if cgpa_val <= 4.0:
                cgpa_val = (cgpa_val / 4.0) * 10.0
            extracted["cgpa"] = min(cgpa_val, 10.0)
            extracted["confidence"]["cgpa"] = "high"
            extracted["raw_findings"].append(f"Found CGPA: {match.group(1)}")
            break
    
    # Extract percentage/grades (convert to 10-point scale)
    if extracted["cgpa"] is None:
        percentage_patterns = [
            r'(\d{2,3})\s*%',
            r'percentage[:\s]+(\d{2,3})',
        ]
        for pattern in percentage_patterns:
            match = re.search(pattern, text_lower)
            if match:
                percentage = float(match.group(1))
                if 0 <= percentage <= 100:
                    extracted["cgpa"] = (percentage / 100.0) * 10.0
                    extracted["confidence"]["cgpa"] = "medium"
                    extracted["raw_findings"].append(f"Found percentage: {percentage}%")
                    break
    
    # Count projects (look for keywords and numbered lists)
    project_count = 0
    
    # Look for "PROJECTS" section and count numbered items
    projects_section = re.search(r'projects?.*?(?=\n[A-Z]{2,}|\Z)', text, re.IGNORECASE | re.DOTALL)
    if projects_section:
        # Count numbered items (1., 2., etc.)
        numbered_items = len(re.findall(r'^\s*\d+[\.\)]\s+', projects_section.group(), re.MULTILINE))
        project_count = max(project_count, numbered_items)
        extracted["raw_findings"].append(f"Found {numbered_items} numbered projects in PROJECTS section")
    
    # Also count generic project mentions
    project_keywords = ['project', 'capstone', 'thesis']
    for keyword in project_keywords:
        project_count += len(re.findall(rf'\b{keyword}\b', text_lower))
    
    extracted["projects"] = min(project_count, 10)
    extracted["confidence"]["projects"] = "high" if projects_section else "medium" if project_count > 0 else "low"
    
    # Check for internship
    internship_keywords = ['intern', 'internship', 'trainee', 'co-op', 'work experience']
    has_internship = any(keyword in text_lower for keyword in internship_keywords)
    extracted["internship"] = "Yes" if has_internship else "No"
    extracted["confidence"]["internship"] = "high" if has_internship else "low"
    if has_internship:
        extracted["raw_findings"].append("Found internship/work experience")
    
    # Estimate communication skills based on content quality and structure
    word_count = len(text.split())
    has_summary = any(kw in text_lower for kw in ['summary', 'objective', 'about', 'profile'])
    has_sections = len(re.findall(r'\n[A-Z][A-Z\s]{3,}\n', text)) >= 3  # Well-structured sections
    
    # Better scoring based on resume quality
    comm_score = 5  # baseline
    if word_count > 300:
        comm_score += 2
    if word_count > 500:
        comm_score += 1
    if has_summary:
        comm_score += 1
    if has_sections:
        comm_score += 1
    
    extracted["comm_skills"] = min(comm_score, 10)
    extracted["confidence"]["comm_skills"] = "medium" if has_sections else "low"
    extracted["raw_findings"].append(f"Resume quality: {word_count} words, {'well-structured' if has_sections else 'basic structure'}")
    
    # Count extracurricular activities
    extracurr_keywords = ['volunteer', 'club', 'sport', 'leadership', 'award', 'certificate', 
                          'achievement', 'competition', 'hackathon', 'event']
    extracurr_count = sum(1 for kw in extracurr_keywords if kw in text_lower)
    extracted["extra_curricular"] = min(extracurr_count, 10)
    extracted["confidence"]["extra_curricular"] = "medium" if extracurr_count > 0 else "low"
    if extracurr_count > 0:
        extracted["raw_findings"].append(f"Found {extracurr_count} extracurricular indicators")
    
    # Academic performance (infer from CGPA if available)
    if extracted["cgpa"] is not None:
        if extracted["cgpa"] >= 8.5:
            extracted["academic_performance"] = 9
        elif extracted["cgpa"] >= 7.5:
            extracted["academic_performance"] = 8
        elif extracted["cgpa"] >= 6.5:
            extracted["academic_performance"] = 7
        else:
            extracted["academic_performance"] = 6
        extracted["confidence"]["academic_performance"] = "medium"
    
    # Previous semester result (default to CGPA if found)
    if extracted["cgpa"] is not None:
        extracted["prev_sem_result"] = extracted["cgpa"]
        extracted["confidence"]["prev_sem_result"] = "medium"
    
    # IQ - can't really extract, use default
    extracted["iq"] = 100  # Default average
    extracted["confidence"]["iq"] = "assumed"
    extracted["raw_findings"].append("IQ set to default (100)")
    
    return extracted


def generate_text_embedding(text: str, target_dim: int = 50) -> np.ndarray:
    """Generate a simple text embedding from resume text using TF-IDF-like approach.
    
    Args:
        text: Resume text content
        target_dim: Target dimensionality (default 50 for hybrid model)
    
    Returns:
        numpy array of shape (1, target_dim)
    """
    # Simple but effective: hash words and create a fixed-size embedding
    from hashlib import md5
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Create embedding vector
    embedding = np.zeros(target_dim)
    
    # For each word, hash it and add to corresponding dimensions
    for word in words:
        if len(word) > 2:  # Skip very short words
            hash_val = int(md5(word.encode()).hexdigest(), 16)
            idx = hash_val % target_dim
            embedding[idx] += 1.0
    
    # Normalize using L2 norm
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    # Add some statistical features
    if target_dim >= 10:
        embedding[-10] = len(words) / 1000.0  # Word count (normalized)
        embedding[-9] = len(set(words)) / len(words) if words else 0  # Vocabulary richness
        embedding[-8] = sum(1 for w in words if len(w) > 8) / len(words) if words else 0  # Complex words
        embedding[-7] = text.count('.') / 100.0  # Sentence count proxy
        embedding[-6] = text.count('\n') / 100.0  # Line breaks
    
    return embedding.reshape(1, -1)


def prepare_hybrid_inputs(raw_data: dict, resume_text: str = None):
    """Prepare inputs for hybrid model: tabular features (8) + text embedding (50).
    
    Args:
        raw_data: Dictionary with student profile data
        resume_text: Resume text for generating embeddings
    
    Returns:
        tuple: (tabular_input, text_embedding) both as numpy arrays
    """
    # Load scaler
    scaler = joblib.load('tabular_scaler.joblib')
    
    # Prepare 8 tabular features (not the 12 engineered ones)
    # Order: CGPA, Prev_Sem, Acad_Perf, IQ, Projects, Internship, Comm, Extra
    tabular_data = np.array([[
        raw_data.get('cgpa', 7.5),
        raw_data.get('prev_sem_result', 7.0),
        raw_data.get('academic_performance', 7),
        raw_data.get('iq', 100),
        raw_data.get('projects', 2),
        1 if raw_data.get('internship', 'No') == 'Yes' else 0,
        raw_data.get('comm_skills', 7),
        raw_data.get('extra_curricular', 5),
    ]])
    
    # Scale the tabular data
    tabular_scaled = scaler.transform(tabular_data)
    
    # Generate text embedding
    if resume_text:
        text_embedding = generate_text_embedding(resume_text, target_dim=50)
    else:
        # Create a default/placeholder embedding if no resume text
        # Use small random values
        text_embedding = np.random.randn(1, 50) * 0.01
    
    return tabular_scaled, text_embedding


# -------------------------
# Analysis & Insights Functions
# -------------------------
def analyze_strengths_weaknesses(raw_data: dict, pred_label: int, confidence: float):
    """Analyze student profile and provide detailed strengths/weaknesses report."""
    strengths = []
    weaknesses = []
    recommendations = []
    
    # CGPA Analysis
    cgpa = raw_data.get("cgpa", 0)
    if cgpa >= 8.5:
        strengths.append(f"🌟 **Excellent CGPA ({cgpa:.2f})** - Well above industry standards")
    elif cgpa >= 7.5:
        strengths.append(f"✅ **Good CGPA ({cgpa:.2f})** - Meets most placement criteria")
    elif cgpa >= 6.5:
        weaknesses.append(f"⚠️ **Average CGPA ({cgpa:.2f})** - Some companies may filter out")
        recommendations.append("📚 Focus on improving semester grades to boost overall CGPA")
    else:
        weaknesses.append(f"❌ **Low CGPA ({cgpa:.2f})** - Below most placement cutoffs")
        recommendations.append("🎯 Consider CGPA improvement courses and retake weak subjects")
    
    # Projects Analysis
    projects = raw_data.get("projects", 0)
    if projects >= 5:
        strengths.append(f"💼 **Extensive Project Portfolio ({projects} projects)** - Strong practical experience")
    elif projects >= 3:
        strengths.append(f"📂 **Good Project Work ({projects} projects)** - Demonstrates hands-on skills")
    elif projects >= 1:
        weaknesses.append(f"⚠️ **Limited Projects ({projects})** - More practical work needed")
        recommendations.append("💡 Build 2-3 substantial projects showcasing different technologies")
    else:
        weaknesses.append(f"❌ **No Projects Listed** - Critical gap in technical portfolio")
        recommendations.append("🚀 URGENT: Start building projects immediately (web app, ML model, etc.)")
    
    # Internship Analysis
    internship = raw_data.get("internship", "No")
    if internship == "Yes":
        strengths.append("🏢 **Industry Experience** - Internship adds significant value")
    else:
        weaknesses.append("⚠️ **No Internship Experience** - Missing real-world exposure")
        recommendations.append("🔍 Apply for internships/summer training programs")
    
    # Communication Skills
    comm = raw_data.get("comm_skills", 0)
    if comm >= 8:
        strengths.append(f"🗣️ **Strong Communication ({comm}/10)** - Key for interviews and teamwork")
    elif comm >= 6:
        strengths.append(f"✅ **Adequate Communication ({comm}/10)** - Room for improvement")
    else:
        weaknesses.append(f"⚠️ **Weak Communication ({comm}/10)** - May struggle in interviews")
        recommendations.append("🎤 Join debate club, practice mock interviews, improve English fluency")
    
    # Extra Curricular
    extra = raw_data.get("extra_curricular", 0)
    if extra >= 7:
        strengths.append(f"🏆 **Well-Rounded Profile ({extra}/10 extracurricular)** - Leadership & teamwork")
    elif extra >= 4:
        strengths.append(f"✅ **Some Activities ({extra}/10)** - Shows initiative beyond academics")
    else:
        weaknesses.append(f"⚠️ **Limited Extracurricular ({extra}/10)** - Profile needs diversification")
        recommendations.append("🎯 Participate in clubs, hackathons, volunteer work, or sports")
    
    # IQ/Aptitude
    iq = raw_data.get("iq", 100)
    if iq >= 120:
        strengths.append(f"🧠 **High Aptitude (IQ: {iq})** - Strong problem-solving ability")
    elif iq >= 110:
        strengths.append(f"✅ **Above Average Aptitude (IQ: {iq})**")
    elif iq < 90:
        weaknesses.append(f"⚠️ **Aptitude Score (IQ: {iq})** - Practice aptitude tests")
        recommendations.append("📖 Regularly solve quantitative, logical reasoning, and coding problems")
    
    # Academic Performance
    acad = raw_data.get("academic_performance", 0)
    if acad >= 8:
        strengths.append(f"📈 **Consistent Academic Performance ({acad}/10)**")
    elif acad < 6:
        weaknesses.append(f"⚠️ **Inconsistent Academics ({acad}/10)**")
        recommendations.append("📊 Maintain regular study schedule and attend all classes")
    
    # Overall Assessment
    overall_score = (
        (cgpa / 10) * 30 +  # CGPA: 30%
        (projects / 10) * 20 +  # Projects: 20%
        (1 if internship == "Yes" else 0) * 15 +  # Internship: 15%
        (comm / 10) * 15 +  # Communication: 15%
        (extra / 10) * 10 +  # Extra-curricular: 10%
        (acad / 10) * 10  # Academic Performance: 10%
    )  # Already out of 100, no need to multiply
    
    # Prediction Analysis
    placement_analysis = {
        "overall_score": round(overall_score, 1),
        "grade": "A+" if overall_score >= 90 else "A" if overall_score >= 80 else "B+" if overall_score >= 70 else "B" if overall_score >= 60 else "C" if overall_score >= 50 else "D",
        "prediction": "PLACED" if pred_label == 1 else "NOT PLACED",
        "confidence": round(confidence, 1),
    }
    
    # Why this prediction?
    reasons = []
    if pred_label == 1:  # Placed
        if cgpa >= 7.5:
            reasons.append(f"✓ CGPA ({cgpa:.2f}) meets industry standards")
        if projects >= 3:
            reasons.append(f"✓ Strong project portfolio ({projects} projects)")
        if internship == "Yes":
            reasons.append("✓ Valuable internship experience")
        if comm >= 7:
            reasons.append(f"✓ Good communication skills ({comm}/10)")
        if extra >= 5:
            reasons.append(f"✓ Well-rounded profile ({extra}/10 activities)")
    else:  # Not Placed
        if cgpa < 7.0:
            reasons.append(f"✗ CGPA ({cgpa:.2f}) below most cutoffs")
        if projects < 2:
            reasons.append(f"✗ Insufficient projects ({projects})")
        if internship == "No":
            reasons.append("✗ No industry experience")
        if comm < 6:
            reasons.append(f"✗ Weak communication ({comm}/10)")
        if extra < 3:
            reasons.append(f"✗ Limited extracurricular involvement ({extra}/10)")
    
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "placement_analysis": placement_analysis,
        "reasons": reasons
    }


def create_radar_chart_data(raw_data: dict):
    """Create data for radar/spider chart visualization."""
    categories = ['CGPA', 'Projects', 'Internship', 'Communication', 'Extra-Curricular', 'Academics']
    values = [
        (raw_data.get("cgpa", 0) / 10) * 100,
        (raw_data.get("projects", 0) / 10) * 100,
        100 if raw_data.get("internship", "No") == "Yes" else 0,
        (raw_data.get("comm_skills", 0) / 10) * 100,
        (raw_data.get("extra_curricular", 0) / 10) * 100,
        (raw_data.get("academic_performance", 0) / 10) * 100,
    ]
    
    return pd.DataFrame({
        'Category': categories,
        'Score': values
    })


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .strength-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .weakness-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f44336;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .strength-item {
        color: #2e7d32;
        font-weight: 500;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .weakness-item {
        color: #c62828;
        font-weight: 500;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .recommendation-item {
        color: #1565c0;
        font-weight: 500;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .prediction-badge-placed {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 1rem;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        margin: 1rem 0;
    }
    .prediction-badge-not-placed {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 1rem;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
        margin: 1rem 0;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .reason-box {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎓 AI-Powered Placement Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive placement prediction with detailed insights, strength/weakness analysis, and actionable recommendations**")
st.markdown("---")

# Sidebar: model selection and general settings
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/student-male.png", width=150)
    st.markdown("### ⚙️ Configuration")
    
    model_choice = st.selectbox("🤖 Choose AI Model", list(AVAILABLE_MODELS.keys()),
                               help="Select the machine learning model for prediction")
    model_path = AVAILABLE_MODELS[model_choice]
    
    # Show loading spinner while loading model
    with st.spinner("Loading model..."):
        model_obj, model_meta = load_model_by_path(model_path)

    if model_obj is None:
        if model_meta.get("error") == "file_not_found":
            st.error("❌ Model file not found in the workspace.")
        elif model_meta.get("error") == "keras_not_available":
            st.error("❌ Keras/TensorFlow not available to load .h5 model.")
        else:
            st.error("❌ Unable to load model: " + str(model_meta))
    else:
        st.success("✅ Model loaded successfully")
        
        # Check if it's a multi-input Keras model and show appropriate message
        if model_meta.get("type") == "keras" and hasattr(model_obj, 'input_shape') and isinstance(model_obj.input_shape, list):
            st.success("✅ **Hybrid Neural Network Active**")
            st.info("📄 **Next Step:** Scroll down to upload your resume in the **Resume Scanner** section for full model functionality.")
        else:
            st.info("💡 Upload a resume for auto-fill, or manually adjust the sliders below.")

    st.markdown("---")
    st.markdown("### 📊 Display Options")
    batch_mode = st.checkbox("📁 Batch CSV Upload", value=False)
    show_feature_importance = st.checkbox("📈 Feature Importance", value=True)
    show_detailed_analysis = st.checkbox("🔍 Detailed Analysis", value=True)
    
    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    with st.expander("How to use this app"):
        st.markdown("""
        **For Random Forest & AdaBoost Models:**
        1. Adjust profile sliders below  
        2. Click 'Predict Placement'  
        3. Review results & insights
        
        **For Hybrid Neural Network:**
        1. **Upload resume** in Resume Scanner section ⬇️  
        2. Auto-extracted data fills the sliders  
        3. Click 'Predict Placement'  
        4. Get advanced AI analysis
        
        ---
        
        **Model Comparison:**
        - **Random Forest**: Fast & accurate (no resume needed) ⭐
        - **AdaBoost**: Ensemble learning (no resume needed)
        - **Hybrid NN**: Most advanced (resume required) 🚀
        """)
    
    st.markdown("---")
    st.markdown("**Model Details:**")
    st.code(f"File: {model_path}", language="text")
    if model_meta.get("type"):
        st.code(f"Type: {model_meta.get('type')}", language="text")


# Resume Upload Section
st.markdown("### 📄 Smart Resume Scanner")
st.markdown("**Upload your resume to automatically extract profile information and enable AI-powered analysis**")

resume_col1, resume_col2 = st.columns([1, 1])

with resume_col1:
    uploaded_resume = st.file_uploader(
        "📎 Choose Resume File", 
        type=["pdf", "txt"], 
        help="Upload your resume in PDF or TXT format for automatic data extraction"
    )

resume_data = None
resume_text = None  # Store resume text globally for hybrid model

if uploaded_resume is not None:
    with resume_col2:
        with st.spinner("🔍 Scanning resume..."):
            # Extract text
            if uploaded_resume.type == "application/pdf":
                resume_text, error = extract_text_from_pdf(uploaded_resume)
                if error:
                    st.error(error)
                    if PyPDF2 is None:
                        st.info("💡 Install PyPDF2: `pip install PyPDF2`")
            else:  # text file
                resume_text = uploaded_resume.read().decode("utf-8")
                error = None
            
            if resume_text and not error:
                # Parse the resume
                resume_data = parse_resume(resume_text)
                
                st.success("✅ Resume scanned successfully!")
                st.metric("Words Analyzed", len(resume_text.split()))

if resume_data:
    with st.expander("📊 Extracted Information & AI Insights", expanded=True):
        st.markdown("**🤖 AI Analysis Results:**")
        
        # Show findings in a nice grid
        findings_to_show = resume_data["raw_findings"][:6]  # Show top 6
        if findings_to_show:
            for finding in findings_to_show:
                st.markdown(f"✓ {finding}")
        
        st.markdown("---")
        st.markdown("**📈 Extracted Profile Data:**")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            if resume_data["cgpa"]:
                confidence_emoji = "🟢" if resume_data["confidence"].get("cgpa") == "high" else "🟡" if resume_data["confidence"].get("cgpa") == "medium" else "🔴"
                st.metric("CGPA", f"{resume_data['cgpa']:.2f}", delta=f"{confidence_emoji} {resume_data['confidence'].get('cgpa', 'low')}")
        with metric_col2:
            if resume_data["projects"] is not None:
                st.metric("Projects", resume_data["projects"], delta="Detected")
        with metric_col3:
            if resume_data["internship"]:
                intern_icon = "✅" if resume_data["internship"] == "Yes" else "❌"
                st.metric("Internship", f"{intern_icon} {resume_data['internship']}")
        with metric_col4:
            if resume_data["extra_curricular"] is not None:
                st.metric("Activities", resume_data["extra_curricular"], delta="Extracted")
        
        st.info("💡 The sliders below have been auto-filled. Review and adjust before prediction.")

st.markdown("---")

# Main layout: inputs and outputs
main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.markdown("### 👤 Student Profile")
    
    # Use resume data if available, otherwise use defaults
    default_cgpa = resume_data["cgpa"] if resume_data and resume_data["cgpa"] else 7.5
    default_acad = resume_data["academic_performance"] if resume_data and resume_data["academic_performance"] else 7
    default_prev = resume_data["prev_sem_result"] if resume_data and resume_data["prev_sem_result"] else 7.0
    default_iq = resume_data["iq"] if resume_data and resume_data["iq"] else 100
    default_projects = resume_data["projects"] if resume_data and resume_data["projects"] is not None else 2
    default_internship = resume_data["internship"] if resume_data and resume_data["internship"] else "No"
    default_comm = resume_data["comm_skills"] if resume_data and resume_data["comm_skills"] else 7
    default_extra = resume_data["extra_curricular"] if resume_data and resume_data["extra_curricular"] is not None else 5
    
    st.markdown("**📚 Academic Metrics**")
    raw_cgpa = st.slider("CGPA (Cumulative Grade Point Average)", min_value=0.0, max_value=10.0, value=float(default_cgpa), step=0.1,
                         help="Your overall CGPA on a 10-point scale" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["cgpa"] else ""))
    
    raw_prev_sem = st.slider("Previous Semester Result", min_value=0.0, max_value=10.0, value=float(default_prev), step=0.1,
                             help="Your GPA/percentage from last semester" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["prev_sem_result"] else ""))
    
    raw_acad_perf = st.slider("Academic Performance Score (1-10)", min_value=1, max_value=10, value=int(default_acad),
                              help="Overall academic consistency and performance" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["academic_performance"] else ""))
    
    st.markdown("**🧠 Aptitude & Skills**")
    raw_iq = st.slider("IQ / Aptitude Score", min_value=60, max_value=160, value=int(default_iq),
                       help="Your aptitude test score or IQ estimate")
    
    raw_comm_skills = st.slider("Communication Skills (1-10)", min_value=1, max_value=10, value=int(default_comm),
                                help="Your verbal and written communication ability" + (" 🔵 Estimated from resume quality" if resume_data and resume_data["comm_skills"] else ""))
    
    st.markdown("**💼 Experience & Activities**")
    raw_projects = st.slider("Projects Completed", min_value=0, max_value=15, value=int(default_projects),
                            help="Number of academic/personal projects" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["projects"] is not None else ""))
    
    raw_internship = st.selectbox("Internship Experience", ("No", "Yes"), 
                                  index=0 if default_internship == "No" else 1,
                                  help="Have you completed any internship?" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["internship"] else ""))
    
    raw_extra_curr = st.slider("Extra Curricular Score (0-10)", min_value=0, max_value=10, value=int(default_extra),
                              help="Sports, clubs, volunteering, leadership" + (" 🔵 Auto-filled from resume" if resume_data and resume_data["extra_curricular"] is not None else ""))

    st.markdown("---")
    
    # Batch mode option
    if batch_mode:
        uploaded = st.file_uploader(
            "📂 Upload CSV for batch prediction", 
            type=["csv"],
            help="CSV must have: cgpa, academic_performance, prev_sem_result, iq, projects, internship, comm_skills, extra_curricular"
        )
    else:
        uploaded = None

    st.markdown("")
    predict_button = st.button("🎯 Predict Placement", type="primary", use_container_width=True)

with main_col2:
    st.markdown("### 📊 Prediction Results & Insights")
    
    # Check if hybrid model is selected and show warning if no resume
    if "Hybrid" in model_choice and not resume_text and not predict_button:
        st.warning("⚠️ **Action Required: Upload Resume for Hybrid Model**")
        st.info("""
        ### 📄 How to Use the Hybrid Neural Network:
        
        This advanced model combines **tabular data** + **resume text analysis** for better predictions.
        
        **Step 1:** Scroll up to the **"Smart Resume Scanner"** section  
        **Step 2:** Upload your resume (PDF or TXT format)  
        **Step 3:** The app will automatically extract your information  
        **Step 4:** Come back here and click **'Predict Placement'**
        
        **Don't have a resume?** Switch to **Random Forest** or **AdaBoost** models - they work great without resume upload!
        """)
        
        # Add helpful button to scroll up
        st.markdown("⬆️ **Look for the Resume Scanner section above** ⬆️")
    
    # Placeholder for results
    result_container = st.container()
    
    with result_container:
        if not predict_button:
            if not ("Hybrid" in model_choice and not resume_text):
                st.info("👈 Fill in your profile and click **'Predict Placement'** to get AI-powered insights!")
                
                # Show sample visualization
                st.markdown("**Sample Profile Analysis:**")
                sample_data = {
                    'Category': ['CGPA', 'Projects', 'Internship', 'Communication', 'Extra-Curricular', 'Academics'],
                    'Score': [75, 40, 0, 70, 50, 70]
                }
                st.bar_chart(pd.DataFrame(sample_data).set_index('Category'))


def predict_single(model, meta, raw_inputs: dict, resume_text: str = None):
    engineered = engineer_features(raw_inputs)
    pred_label = None
    confidence = None
    error_msg = None

    # sklearn models
    if meta.get("type") == "sklearn":
        try:
            pred = model.predict(engineered)
            pred_label = int(pred[0])
        except Exception as e:
            error_msg = f"sklearn predict error: {str(e)}"
            pred_label = None
        try:
            proba = model.predict_proba(engineered)
            confidence = float(np.max(proba) * 100)
        except Exception as e:
            if error_msg is None:
                error_msg = f"sklearn predict_proba error: {str(e)}"
            confidence = None

    # keras models (assume binary classification with sigmoid or probability output)
    elif meta.get("type") == "keras":
        try:
            # Check if model expects multiple inputs
            if hasattr(model, 'input_shape') and isinstance(model.input_shape, list):
                # Multi-input hybrid model - needs tabular + text embedding
                if resume_text:
                    # We have resume text - can generate embedding
                    tabular_input, text_embedding = prepare_hybrid_inputs(raw_inputs, resume_text)
                    proba = model.predict([tabular_input, text_embedding], verbose=0)
                    p = float(proba.ravel()[0])
                    pred_label = 1 if p >= 0.5 else 0
                    confidence = float(p * 100) if p >= 0.5 else float((1 - p) * 100)
                else:
                    # No resume text - cannot use hybrid model properly
                    error_msg = f"Hybrid model requires resume text for embeddings. Please upload a resume (PDF/TXT) to use this model, or switch to Random Forest/AdaBoost."
                    pred_label = None
                    confidence = None
            else:
                # Single input model
                X = engineered.values
                proba = model.predict(X, verbose=0)
                # handle shape variations
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    # probabilities for classes
                    p = proba[0]
                    pred_label = int(np.argmax(p))
                    confidence = float(np.max(p) * 100)
                else:
                    p = float(proba.ravel()[0])
                    pred_label = 1 if p >= 0.5 else 0
                    confidence = float(p * 100)
        except Exception as e:
            error_msg = f"keras predict error: {str(e)}"
            pred_label = None
            confidence = None

    return pred_label, confidence, engineered, error_msg


# Handle prediction actions
if predict_button:
    if model_obj is None:
        st.error("❌ No model is available to make predictions. Check the sidebar.")
    else:
        # Single or batch
        if uploaded is not None:
            # Batch mode (existing code)
            try:
                df_raw = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read uploaded CSV: {e}")
                df_raw = None

            if df_raw is not None:
                required_cols = [
                    "cgpa",
                    "academic_performance",
                    "prev_sem_result",
                    "iq",
                    "projects",
                    "internship",
                    "comm_skills",
                    "extra_curricular",
                ]
                missing = [c for c in required_cols if c not in df_raw.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: {missing}")
                else:
                    results = []
                    for _, r in df_raw.iterrows():
                        raw = {
                            "cgpa": float(r["cgpa"]),
                            "academic_performance": float(r["academic_performance"]),
                            "prev_sem_result": float(r["prev_sem_result"]),
                            "iq": float(r["iq"]),
                            "projects": int(r["projects"]),
                            "internship": str(r["internship"]),
                            "comm_skills": float(r["comm_skills"]),
                            "extra_curricular": float(r["extra_curricular"]),
                        }
                        pred_label, confidence, engineered, error_msg = predict_single(model_obj, model_meta, raw, resume_text)
                        results.append({"raw": raw, "engineered": engineered.to_dict(orient="records")[0], "prediction": pred_label, "confidence": confidence, "error": error_msg})

                    st.success(f"✅ Batch predictions completed: {len(results)} rows")
                    st.write(pd.DataFrame([{"prediction": r["prediction"], "confidence": r["confidence"]} for r in results]))

                    # Append batch to history
                    for r in results:
                        entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model_choice,
                            "inputs": json.dumps(r["raw"]),
                            "engineered": json.dumps(r["engineered"]),
                            "prediction": r["prediction"],
                            "confidence": r["confidence"],
                        }
                        append_history(entry)

        else:
            # Single prediction mode
            user_raw = {
                "cgpa": float(raw_cgpa),
                "academic_performance": float(raw_acad_perf),
                "prev_sem_result": float(raw_prev_sem),
                "iq": float(raw_iq),
                "projects": int(raw_projects),
                "internship": str(raw_internship),
                "comm_skills": float(raw_comm_skills),
                "extra_curricular": float(raw_extra_curr),
            }

            pred_label, confidence, engineered, error_msg = predict_single(model_obj, model_meta, user_raw, resume_text)

            if pred_label is None:
                with main_col2:
                    st.error("❌ Model could not produce a prediction for the provided input.")
                    if error_msg:
                        st.error(f"**Error details:** {error_msg}")
                    
                    # Show debug info
                    with st.expander("🔧 Debug Information"):
                        st.write("**Model metadata:**", model_meta)
                        st.write("**Engineered input shape:**", engineered.shape)
                        st.write("**Engineered input:**")
                        st.dataframe(engineered)
            else:
                # Success! Show results
                with main_col2:
                    # Main prediction result with beautiful badge
                    if pred_label == 1:
                        st.markdown(f'<div class="prediction-badge-placed">✅ PLACED</div>', unsafe_allow_html=True)
                        st.metric("Confidence Level", f"{confidence:.1f}%", delta="High Confidence" if confidence >= 80 else "Medium Confidence" if confidence >= 60 else "Low Confidence")
                        st.balloons()
                    else:
                        st.markdown(f'<div class="prediction-badge-not-placed">❌ NOT PLACED</div>', unsafe_allow_html=True)
                        st.metric("Confidence Level", f"{confidence:.1f}%", delta="High Confidence" if confidence >= 80 else "Medium Confidence" if confidence >= 60 else "Low Confidence")
                    
                    st.markdown("---")
                    
                    # Detailed analysis
                    if show_detailed_analysis:
                        analysis = analyze_strengths_weaknesses(user_raw, pred_label, confidence)
                        
                        # Overall Score Card with gradient
                        st.markdown('<p class="section-header">📈 Overall Profile Assessment</p>', unsafe_allow_html=True)
                        
                        score_col1, score_col2, score_col3 = st.columns(3)
                        with score_col1:
                            st.markdown(f"""
                            <div class="score-card">
                                <div style="font-size: 0.9rem; opacity: 0.9;">Profile Score</div>
                                <div style="font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;">{analysis['placement_analysis']['overall_score']}</div>
                                <div style="font-size: 0.9rem; opacity: 0.9;">out of 100</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with score_col2:
                            grade = analysis['placement_analysis']['grade']
                            grade_color = "#4caf50" if grade in ["A+", "A"] else "#ff9800" if grade in ["B+", "B"] else "#f44336"
                            st.markdown(f"""
                            <div style="background: white; padding: 1.5rem; border-radius: 1rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                                <div style="font-size: 0.9rem; color: #666;">Grade</div>
                                <div style="font-size: 2.5rem; font-weight: 700; color: {grade_color}; margin: 0.5rem 0;">{grade}</div>
                                <div style="font-size: 0.9rem; color: #666;">Letter Grade</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with score_col3:
                            pred_color = "#4caf50" if pred_label == 1 else "#f44336"
                            st.markdown(f"""
                            <div style="background: white; padding: 1.5rem; border-radius: 1rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                                <div style="font-size: 0.9rem; color: #666;">Prediction</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: {pred_color}; margin: 0.5rem 0;">{analysis['placement_analysis']['prediction']}</div>
                                <div style="font-size: 0.9rem; color: #666;">AI Verdict</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Why this prediction?
                        st.markdown('<p class="section-header">🎯 Prediction Reasoning</p>', unsafe_allow_html=True)
                        if analysis['reasons']:
                            for reason in analysis['reasons']:
                                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)
                        else:
                            st.info("Borderline case - multiple factors contributed equally.")
                        
                        # Profile visualization
                        st.markdown('<p class="section-header">📊 Profile Breakdown</p>', unsafe_allow_html=True)
                        radar_data = create_radar_chart_data(user_raw)
                        st.bar_chart(radar_data.set_index('Category'), use_container_width=True)
                        
                        # Strengths, Weaknesses, Recommendations
                        st.markdown("---")
                        strength_col, weakness_col = st.columns(2)
                        
                        with strength_col:
                            st.markdown('<p class="section-header">✅ Key Strengths</p>', unsafe_allow_html=True)
                            if analysis['strengths']:
                                for strength in analysis['strengths']:
                                    st.markdown(f'<div class="strength-box"><p class="strength-item">{strength}</p></div>', unsafe_allow_html=True)
                            else:
                                st.info("💪 Work on building your strengths!")
                        
                        with weakness_col:
                            st.markdown('<p class="section-header">⚠️ Areas to Improve</p>', unsafe_allow_html=True)
                            if analysis['weaknesses']:
                                for weakness in analysis['weaknesses']:
                                    st.markdown(f'<div class="weakness-box"><p class="weakness-item">{weakness}</p></div>', unsafe_allow_html=True)
                            else:
                                st.success("🎉 Great! No major weaknesses detected.")
                        
                        # Recommendations
                        if analysis['recommendations']:
                            st.markdown('<p class="section-header">💡 Actionable Recommendations</p>', unsafe_allow_html=True)
                            for i, rec in enumerate(analysis['recommendations'], 1):
                                st.markdown(f'<div class="recommendation-box"><p class="recommendation-item"><strong>{i}.</strong> {rec}</p></div>', unsafe_allow_html=True)
                    
                    # Technical details
                    with st.expander("🔬 Technical Details & Feature Engineering"):
                        st.markdown("**Engineered Features (Model Input):**")
                        st.dataframe(engineered.T, use_container_width=True)

                        # Feature importance for sklearn tree-based models
                        if show_feature_importance and hasattr(model_obj, "feature_importances_"):
                            try:
                                fi = pd.Series(model_obj.feature_importances_, index=engineered.columns).sort_values(ascending=False)
                                st.markdown("**Feature Importance Analysis:**")
                                st.bar_chart(fi)
                                
                                st.markdown("**Top 5 Most Important Features:**")
                                for i, (feat, importance) in enumerate(fi.head(5).items(), 1):
                                    st.write(f"{i}. **{feat}**: {importance:.4f}")
                            except Exception:
                                st.info("Could not compute feature importances for this model.")

                        # Show model output details
                        st.markdown("**Raw Model Output:**")
                        st.json({"prediction": int(pred_label), "confidence_score": round(confidence, 2), "model": model_choice})

                # Append to history
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model_choice,
                    "inputs": json.dumps(user_raw),
                    "engineered": json.dumps(engineered.to_dict(orient="records")[0]),
                    "prediction": int(pred_label),
                    "confidence": float(confidence) if confidence is not None else None,
                }
                append_history(entry)

# Show history and allow export
st.markdown("---")
st.markdown("### 📜 Prediction History")
st.markdown("Track all your past predictions and download the complete history.")

if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    # parse json columns for nicer display
    if not hist.empty:
        hist_display = hist.copy()
        
        # Show summary metrics
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Total Predictions", len(hist))
        with summary_col2:
            placed_count = hist[hist['prediction'] == 1].shape[0]
            st.metric("Placed", placed_count)
        with summary_col3:
            not_placed_count = hist[hist['prediction'] == 0].shape[0]
            st.metric("Not Placed", not_placed_count)
        with summary_col4:
            avg_confidence = hist['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        st.markdown("**Recent Predictions:**")
        
        # Format timestamp - handle ISO8601 format with timezone
        try:
            hist_display['timestamp'] = pd.to_datetime(hist_display['timestamp'], format='ISO8601', utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        except:
            # Fallback: try without format specification
            hist_display['timestamp'] = pd.to_datetime(hist_display['timestamp'], utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        
        # Show most recent 10
        display_cols = ['timestamp', 'model', 'prediction', 'confidence']
        st.dataframe(
            hist_display[display_cols].sort_values("timestamp", ascending=False).head(10).reset_index(drop=True),
            use_container_width=True
        )
        
        # Download button
        csv = hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Complete History (CSV)", 
            data=csv, 
            file_name="predictions_history.csv", 
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No past predictions found. Make your first prediction above!")
else:
    st.info("No history file yet. Make a prediction to start tracking history.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🎓 AI-Powered Placement Predictor</strong></p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
    <p style='font-size: 0.8rem;'>Upload resume → Auto-fill profile → Get AI insights → Improve your placement chances</p>
</div>
""", unsafe_allow_html=True)
