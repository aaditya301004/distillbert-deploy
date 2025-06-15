import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"

import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from evaluator import RobustJobMismatchEvaluator
import plotly.graph_objects as go

# ----------------- Glassmorphism CSS -----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background: radial-gradient(circle at top left, #0f172a, #1e293b);
    color: #f1f5f9;
}

.stApp {
    background: transparent;
}

/* Title Styling */
h1 {
    font-weight: 800 !important;
    color: #e0f2fe;
    text-align: center;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
}

/* Page Container Padding */
.block-container {
    padding: 2rem 2rem 3rem 2rem;
}

/* Glassy Form Styling */
.stForm {
    background: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeInUp 0.8s ease-out;
}

/* Form Label */
label {
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    margin-bottom: 0.25rem;
    display: inline-block;
}

/* Input Fields */
input, textarea, select {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 1rem !important;
    margin-top: 0.3rem;
    margin-bottom: 1rem;
    box-shadow: inset 1px 1px 3px rgba(0,0,0,0.8), inset -1px -1px 3px rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}

input:hover, textarea:hover, select:hover {
    transform: scale(1.02);
    box-shadow: 0 0 0 1px #3b82f6, 0 0 8px #3b82f6;
}

/* Primary Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: bold;
    font-size: 1rem;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.6);
    transition: all 0.3s ease-in-out;
    cursor: pointer;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    transform: translateY(-2px);
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.7), 0 4px 20px rgba(0,0,0,0.5);
}

/* Fade-in Animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Mobile Responsive Adjustments */
@media screen and (max-width: 768px) {
    .stForm {
        padding: 1.2rem !important;
    }

    input, textarea, select {
        font-size: 0.95rem !important;
    }

    h1 {
        font-size: 1.75rem !important;
    }
}
</style>

""", unsafe_allow_html=True)

# ----------------- Load Model & Evaluator -----------------
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("Model_folder", local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained("Model_folder", local_files_only=True)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_evaluator():
    return RobustJobMismatchEvaluator()

model, tokenizer = load_model_and_tokenizer()
evaluator = load_evaluator()

# ----------------- Gauge Drawing Function -----------------
def draw_gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 1,
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef9c3'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Initialize Session -----------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ----------------- UI -----------------
st.markdown("""
    <h1 style='text-align: center;'> Job Fraud & Mismatch Evaluator</h1>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

if not st.session_state.show_results:
    with st.form("job_form"):
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("###  Required Job Details")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input(" Job Title *", placeholder="e.g., Data Analyst")
        with col2:
            employment_type = st.selectbox(" Employment Type *", [
                "Full-time", "Internship", "Internship & Graduate", "Other Part-time", "Temporary"
            ])

        job_description = st.text_area(" Job Description *", height=150)
        skill_desc = st.text_area(" Skills Required *", height=100)
        location = st.text_input(" Location *", placeholder="e.g., Bangalore, India")

        st.markdown("###  Optional Details")
        salary_range = st.text_input(" Salary Range", placeholder="e.g., 4-6 LPA")
        industry = st.text_input(" Industry", placeholder="e.g., IT Services")
        company_profile = st.text_area(" Company Profile", height=100)

        submitted = st.form_submit_button(" Evaluate", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not all([job_title.strip(), job_description.strip(), skill_desc.strip(), location.strip(), employment_type.strip()]):
            st.error("⚠ Please fill all the required fields marked with *.")
        else:
            st.session_state.job_inputs = {
                "Job Title": job_title,
                "Employment Type": employment_type,
                "Job Description": job_description,
                "Skills Required": skill_desc,
                "Location": location,
                "Salary Range": salary_range,
                "Industry": industry,
                "Company Profile": company_profile,
            }
            st.session_state.show_results = True
            st.experimental_set_query_params(page="results")

# ----------------- Results Page -----------------
if st.session_state.show_results:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("##  Prediction Results")

    inputs = st.session_state.get("job_inputs", {})
    combined_text = f"{inputs['Job Title']} {inputs['Job Description']} {inputs['Skills Required']} {inputs['Employment Type']} {inputs['Location']}"

    temperature = 2.0
    tokens = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits / temperature, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
        class_labels = [" Real Job Posting", " Fake Job Posting"]
        prediction_label = class_labels[pred_idx]

    mismatch_score = evaluator.evaluate([inputs["Job Title"]], [inputs["Job Description"]])

    col1, col2 = st.columns(2)
    with col1:
        draw_gauge("Model Confidence", round(confidence, 2))
    with col2:
        draw_gauge("Job-Role Match Score", round(mismatch_score, 2))

    st.markdown(f"###  Prediction: *{prediction_label}*")

    with st.expander(" Job Summary (Your Input)", expanded=True):
        for key, val in inputs.items():
            if val.strip():
                st.markdown(f"{key}:** {val}")

    if st.button("🔄 Evaluate Another Job"):
        st.session_state.show_results = False
        st.experimental_set_query_params()

    st.markdown("</div>", unsafe_allow_html=True)
