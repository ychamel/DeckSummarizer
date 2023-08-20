import streamlit as st

from Main.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your password below🔑\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )

        password = st.text_input(
            "Access Password",
            type="password",
            placeholder="enter the password to access this app",
            help="you can send us an email at https://www.synapse-analytics.io/contact to get access",  # noqa: E501
            value=None
            or st.session_state.get("password", ""),
        )

        api_key_input = None
        if password == os.environ.get("password"):
            api_key_input = os.environ.get("OPENAI_API_KEY", None)

        st.session_state["OCR_ENABLED"] = st.checkbox("OCR Enabled")

        st.session_state["OPENAI_API_KEY"] = api_key_input

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖Synapse DeckSummarizer allows you to ask questions about your "
            "documents and get accurate answers with instant citations. "
        )
        st.markdown("---")

        faq()
