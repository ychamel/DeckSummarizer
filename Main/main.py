import streamlit as st
from PIL import Image

from Main.components.sidebar import sidebar
from Main.core.summary import write_report, store_txt

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

# init
st.set_page_config(page_title="Synapse Deck Summarizer", page_icon="📖", layout="wide")

st.markdown(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-3TZSGJBX3W"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
    
      gtag('config', 'G-3TZSGJBX3W');
    </script>
    """
    ,unsafe_allow_html=True
)
# image
image = Image.open('Main/assets/logo.png')
st.image(image)
# Title
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
uploaded_files = st.file_uploader(
    "Upload file of the following format: pdf, docx, pptx or txt",
    type=["pdf", "docx", "txt", "pptx"],
    help="Scanned documents are not supported yet!",
    accept_multiple_files=True
)

if not uploaded_files or not openai_api_key:
    st.stop()

update_btn = st.button('Update Files')
files = st.session_state.get("FILES")
# read files
if update_btn:
    files = []
    for uploaded_file in uploaded_files:
        try:
            file = read_file(uploaded_file)
            files.append(file)
        except Exception as e:
            display_file_read_error(e)
    st.session_state["FILES"] = files
elif not files:
    st.stop()
# chunk files
chunked_files = []
for file in files:
    chunked_file = chunk_file(file, chunk_size=1000, chunk_overlap=50)
    chunked_files.append(chunked_file)

for file in files:
    if not is_file_valid(file):
        st.stop()

if not is_open_ai_key_valid(openai_api_key):
    st.stop()

# save chunks to temp db
with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=chunked_files,
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=openai_api_key,
    )
    pinecone_index = store_txt(chunked_files)

# open chat area
with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

# options to show more info
with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")
    generate_summary = st.button("Generate Summary")

# generate summary
if generate_summary:
    result = write_report(pinecone_index)
    st.download_button("Download Report", result)

# option to show raw read data
if show_full_doc:
    with st.expander("Document"):
        docs = []
        for file in files:
            docs.extend(file.docs)
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(docs)}</p>", unsafe_allow_html=True)

# setup new chat
if not st.session_state.get("messages"):
    st.session_state.messages = []

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
    # add answer
    st.session_state.get("messages").append({"role": "user", "content": query})
    st.session_state.get("messages").append({"role": "assistant", "content": result.answer})
    with answer_col:
        st.markdown("#### Answer")
        for msg in reversed(st.session_state.get("messages")):
            st.chat_message(msg["role"]).write(msg["content"])
        # st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.write(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
