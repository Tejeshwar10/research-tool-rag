import streamlit as st
from rag import process_sources, generate_answer, save_uploaded_file

st.title("Real Estate Research Tool" )

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


placeholder = st.empty()
process_sources_button = st.sidebar.button("Process Sources")
if process_sources_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]

    pdf_paths = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            saved_path = save_uploaded_file(uploaded_file)
            pdf_paths.append(saved_path)

    if len(urls) == 0 and len(pdf_paths) == 0:
        placeholder.text("Please provide at least one valid URL or upload one PDF")
    else:
        for status in process_sources(urls=urls, pdf_paths=pdf_paths):
            placeholder.text(status)


query  = st.text_input("Enter the question")

if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError:
        placeholder.text("You must process URLs or PDFs first")