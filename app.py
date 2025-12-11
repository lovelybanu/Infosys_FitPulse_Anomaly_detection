import streamlit as st

st.set_page_config(
    page_title="FitPulse – Anomaly Detection",
    page_icon="🏃‍♂️",
    layout="wide",
)

st.title("🏃‍♂️ FitPulse – Anomaly Detection")
st.write(
    """
Welcome to the FitPulse anomaly detection system.

Use the sidebar to navigate:

- **Milestone 1:** Data Upload → Cleaning → Resampling  
- **Milestone 2:** Feature Extraction → Trend Modeling → Clustering  
- **Milestone 3:** Anomaly Detection  
- **Milestone 4:** Dashboard & Insights  
"""
)

st.info("Go to the **left sidebar** and select a Milestone page to get started.")
