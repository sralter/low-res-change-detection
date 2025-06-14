# Home.py

import streamlit as st

st.set_page_config(
    page_title='Home',
    page_icon=':material/domain:'
)

st.markdown(
    """
    # Low-Resolution Satellite Imgagery Change Detection
    """
	)
st.markdown(
	"""
	#### This tool allows the user to interact with a PyTorch-based model \
    that has been trained using satellite imagery from the Dallas, TX area.
	
	***
	
	#### To use the tool, please navigate to the "Demo" page in the sidebar.
	
	***
	
	#### Tool overview:
	##### 1. Select a location (5-character geohash)
	##### 2. Pick two dates
	##### 3. See how things changed
	"""
)