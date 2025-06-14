# 1_Only_Presentation.py

import streamlit as st
from streamlit.components.v1 import html
def main():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    st.header("Low-Resolution Satellite Change Detection | Samuel Alter | Reveal Global Consulting")

    st.subheader("Presentation")
    GOOGLE_SLIDES_EMBED_SRC = (
        "https://docs.google.com/presentation/d/e/"
        "2PACX-1vSoyXNvBjV7c5gyXdsRQPOvn_irAoEBtAaFYOiGk1EthHOYoLoWUr70hsPEe-"
        "V9tNeo3jkDcPDVgRg9/pubembed?start=true&loop=false&delayms=60000"
    )
    iframe_code = f"""
        <iframe
            src="{GOOGLE_SLIDES_EMBED_SRC}"
            frameborder="0"
            width="100%"
            height="750px"
            allowfullscreen="true"
            mozallowfullscreen="true"
            webkitallowfullscreen="true">
        </iframe>
    """
    html(iframe_code, height=800)

if __name__ == '__main__':
    main()
