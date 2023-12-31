import streamlit as st
from PIL import Image

from Main.components.sidebar import sidebar
from Main.core.Analytics import set_data
from Main.core.summary import write_report, store_txt, write_RSM, get_summary

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
from Main.core.qa import query_folder, get_query_answer

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL = "openai"

# For testing
# EMBEDDING, VECTOR_STORE, MODEL = ["debug"] * 3

# init
st.set_page_config(page_title="Synapse Deck Summarizer", page_icon="📖", layout="wide")
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
    "Upload file of the following format: pdf, docx, pptx, xlsx or txt",
    type=["pdf", "docx", "txt", "pptx", "xlsx"],
    help="Scanned documents are not supported yet!",
    accept_multiple_files=True
)

if not uploaded_files or not openai_api_key:
    st.stop()

update_btn = st.button('Update Files')

files = st.session_state.get("FILES")
chunked_files = st.session_state.get("CHUNKED_FILES")
folder_index = st.session_state.get("FOLDER_INDEX")
summary = st.session_state.get("SUMMARY")

# read files
if update_btn:
    # turn uploaded files into file objects
    files = []
    for uploaded_file in uploaded_files:
        try:
            file = read_file(uploaded_file)
            files.append(file)
        except Exception as e:
            display_file_read_error(e)
    st.session_state["FILES"] = files

    # chunk files
    chunked_files = []
    for file in files:
        chunked_file = chunk_file(file, chunk_size=400, chunk_overlap=50)
        chunked_files.append(chunked_file)
    st.session_state["CHUNKED_FILES"] = chunked_files

    # save chunks to temp db
    with st.spinner("Indexing document... This may take a while⏳"):
        folder_index = embed_files(
            files=chunked_files,
            embedding=EMBEDDING,
            vector_store=VECTOR_STORE,
            openai_api_key=openai_api_key,
        )
    st.session_state["FOLDER_INDEX"] = folder_index

    # create a summary
    if len(chunked_files) > 0:
        summary = get_summary(chunked_files[0])
        st.session_state["SUMMARY"] = summary

elif not files:
    st.stop()

for file in files:
    if not is_file_valid(file):
        st.stop()

if not is_open_ai_key_valid(openai_api_key):
    st.stop()

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
    with st.spinner("Generating Report... This may take a while⏳"):
        # pinecone_index = store_txt(files)
        # result = write_report(pinecone_index)
        result = write_RSM(files)
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

    # get updated query
    search_query = query
    if summary:
        search_query += get_query_answer(query, summary)
    # Output Columns
    answer_col, sources_col = st.columns(2)

    result = query_folder(
        folder_index=folder_index,
        query=search_query,
        return_all=return_all_chunks,
        model=MODEL,
        openai_api_key=openai_api_key,
        temperature=0,
    )
    # add answer
    st.session_state.get("messages").append({"role": "user", "content": query})
    st.session_state.get("messages").append({"role": "assistant", "content": result.answer})
    # save to analytics
    set_data(query, result.answer)
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
