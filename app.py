import os
import sys

# Ensure the script directory is in the Python path for seamless importing
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import and execute the primary streamlit_app.py file
try:
    import streamlit_app
except Exception as e:
    import streamlit as st
    st.error(f"Failed to load the primary application file (streamlit_app.py): {e}")
