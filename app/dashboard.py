import streamlit as st

st.set_page_config(
    page_title="Q-ROUTE AI",
    page_icon="🚗",
    layout="wide"
)

st.title("Q-ROUTE AI")
st.subheader(
    "On-Device Intelligent Traffic Prediction "
    "and Quantum-Inspired Route Optimization"
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Traffic Level", "LOW")

with col2:
    st.metric("Vehicles", "0")

with col3:
    st.metric("Route Status", "Optimal")

st.divider()

st.info(
    "Q-ROUTE AI combines traffic prediction, "
    "QPSO optimization and dynamic re-routing."
)