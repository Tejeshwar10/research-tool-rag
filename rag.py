import time
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain

load_dotenv()

# constants
CHUNK_SIZE = 1200
EMBEdding_Model = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "research_tool"
UPLOAD_DIR = Path(__file__).parent / "resources/uploads"

llm = None
vector_store = None


def intialise_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            #model="llama-3.1-8b-instant",
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.7,
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEdding_Model,
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )


def load_pdf_documents(pdf_paths):
    documents = []

    for pdf_path in pdf_paths:
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_file}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def process_sources(urls=None, pdf_paths=None):
    yield "initialising vector Database and LLM"
    intialise_components()

    yield "empting the vector data base"
    vector_store.reset_collection()

    all_docs = []

    if urls:
        yield "loading data from URLs"
        loader = WebBaseLoader(urls)
        url_docs = loader.load()
        all_docs.extend(url_docs)

    if pdf_paths:
        yield "loading data from PDFs"
        pdf_docs = load_pdf_documents(pdf_paths)
        all_docs.extend(pdf_docs)

    if not all_docs:
        raise ValueError("No URLs or PDFs were provided for processing.")

    yield(f"Total loaded documents: {len(all_docs)}")

    yield "Performing Chunking"
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
    )

    docs = splitter.split_documents(all_docs)
    yield(f"Total chunks created: {len(docs)}")

    yield "Creating indexes for the chunks"
    uuids = [str(uuid4()) for _ in range(len(docs))]

    yield "adding the chunks to the vector database"

    start_time = time.time()
    batch_size = 25

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_ids = uuids[i:i + batch_size]

        vector_store.add_documents(batch_docs, ids=batch_ids)
        yield(f"Processed {min(i + batch_size, len(docs))}/{len(docs)} chunks")

    elapsed = time.time() - start_time
    yield(f"Vector DB add completed in {elapsed:.2f} seconds")
    yield("Processing complete")


def save_uploaded_file(uploaded_file):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIR / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("vector DB is not initialised")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result["answer"], sources


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for step in process_sources(urls=urls):
        print(step)

    prompt = "tell me what was the 30 year fixed mortgage rate along with the date?"
    answer, sources = generate_answer(prompt)

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")