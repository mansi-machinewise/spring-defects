"""Streamlit view layer; run with `streamlit run src/views/dashboard.py`."""
import json
from urllib.request import urlopen, Request
import streamlit as st

BASE_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
st.title("Spring Inspection Dashboard")
if st.button("Run simulated inspection"):
    request = Request(f"{BASE_URL}/api/v1/inspections/simulate", method="POST")
    with urlopen(request) as response:
        st.session_state["latest"] = json.loads(response.read())
latest = st.session_state.get("latest")
if latest:
    status = latest["decision"]
    (st.success if status == "PASS" else st.error)(f"{status} — {latest['inspection_id']}")
    st.json(latest)
st.subheader("Recent inspections")
try:
    with urlopen(f"{BASE_URL}/api/v1/inspections/latest") as response: st.dataframe(json.loads(response.read()), use_container_width=True)
except Exception:
    st.info("Start the FastAPI service to view inspection history.")
