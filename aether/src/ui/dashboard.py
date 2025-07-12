import streamlit as st
from aether.src.orchestrator.langgraph_flow import build_aether_graph

st.title("AETHER-AI Cloud Monitoring Dashboard")

st.header("Pipeline Output")
try:
    graph = build_aether_graph()
    output = graph.invoke({})
    
    st.subheader("Logs")
    for log in output["logs"]:
        st.write(f"ID: {log['id']}, Log: {log['log']}, Timestamp: {log['timestamp']}")
    
    st.subheader("Anomalies")
    st.write(output["anomalies"])
    
    st.subheader("Root Cause Analysis")
    st.write(output["rca"])
    
    st.subheader("Recommendations")
    st.write(output["recommendation"])
    
    print("✅ Dashboard loaded successfully.")
except Exception as e:
    st.error(f"❌ Dashboard failed to load: {str(e)}")
    print(f"❌ Dashboard error: {str(e)}")
