import streamlit as st
from rag import process_sources, generate_answer, save_uploaded_file

st.set_page_config(
    page_title="RAG Research Tool",
    page_icon="📘",
    layout="wide",
)

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            color: #9aa0a6;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .section-card {
            padding: 1.25rem 1.25rem 0.75rem 1.25rem;
            border: 1px solid rgba(250,250,250,0.08);
            border-radius: 14px;
            background-color: rgba(255,255,255,0.02);
            margin-bottom: 1rem;
        }
        .small-note {
            color: #9aa0a6;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="main-title">RAG-Based Research Tool</div>
    <div class="sub-title">Ask questions from URLs and uploaded PDF documents</div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Source Ingestion")
    st.markdown(
        "Add one or more URLs, upload PDF documents, then click **Process Sources**."
    )

    url1 = st.text_input("URL 1", placeholder="Paste a web article URL")
    url2 = st.text_input("URL 2", placeholder="Paste another URL")
    url3 = st.text_input("URL 3", placeholder="Paste another URL")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or more PDF documents.",
    )

    process_sources_button = st.button("Process Sources", use_container_width=True)

# ---------- Main layout ----------
left_spacer, center_col, right_spacer = st.columns([1, 2.4, 1])

with center_col:
    status_placeholder = st.empty()

    st.markdown(
        """
        <div class="section-card">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                Ask a Question
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Enter your question",
        placeholder="Example: Summarize the key points from the uploaded PDF and URLs",
        label_visibility="collapsed",
    )

    st.markdown(
        "<div class='small-note'>Process sources first, then ask questions grounded in those documents.</div>",
        unsafe_allow_html=True,
    )

    if process_sources_button:
        urls = [url for url in (url1, url2, url3) if url.strip() != ""]

        pdf_paths = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                saved_path = save_uploaded_file(uploaded_file)
                pdf_paths.append(saved_path)

        if len(urls) == 0 and len(pdf_paths) == 0:
            status_placeholder.error(
                "Please provide at least one valid URL or upload one PDF."
            )
        else:
            try:
                logs = []

                with st.spinner("Processing sources and building the knowledge base..."):
                    for status in process_sources(urls=urls, pdf_paths=pdf_paths):
                        logs.append(status)
                        status_placeholder.info("\n".join(logs[-8:]))

                status_placeholder.success(
                    "Sources processed successfully. You can now ask questions."
                )
            except Exception as e:
                status_placeholder.error(f"Error while processing sources: {str(e)}")

    if query:
        try:
            answer, sources = generate_answer(query)

            st.markdown(
                """
                <div class="section-card">
                    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem;">
                        Answer
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(answer)

            if sources:
                with st.expander("Sources", expanded=True):
                    for source in sources.split("\n"):
                        if source.strip():
                            st.markdown(f"- {source}")

        except RuntimeError:
            status_placeholder.error("You must process URLs or PDFs first.")
        except Exception as e:
            status_placeholder.error(f"Error while generating answer: {str(e)}")