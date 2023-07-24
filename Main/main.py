import streamlit as st
from PIL import Image

from Main.components.sidebar import sidebar

from Main.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from Main.core.caching import bootstrap_caching

from Main.core.parsing import read_file
from Main.core.chunking import chunk_file
from Main.core.embedding import embed_files
from Main.core.qa import query_folder

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL = "openai"

# For testing
# EMBEDDING, VECTOR_STORE, MODEL = ["debug"] * 3

image = Image.open('Main/assets/logo.png')
st.image(image)
# Title
st.set_page_config(page_title="Synapse Deck Summarizer", page_icon="📖", layout="wide")
st.header("Synapse Deck Summarizer")

# Enable caching for expensive functions
bootstrap_caching()

# sidebar
sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")


if not openai_api_key:
    st.warning(
        "please enter a password to access the app!"
    )

# uploader
uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

if not uploaded_file or not openai_api_key:
    st.stop()

# read files
try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e)

# chunk files
chunked_file = chunk_file(file, chunk_size=1000, chunk_overlap=20)

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai_api_key):
    st.stop()

# save chunks to temp db
with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=openai_api_key,
    )

# open chat area
with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

# options to show more info
with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

# option to show raw read data
if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

# when chat sent
if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        model=MODEL,
        openai_api_key=openai_api_key,
        temperature=0,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
