from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

#from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain

load_dotenv()

##constants
CHUNK_SIZE = 1000
EMBEdding_Model = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def intialise_components():
    global llm,vector_store
    # initialissing the LLM 
    if llm is None:

        llm = ChatGroq(model = "llama-3.3-70b-versatile",max_tokens=500,temperature=0.9)
    

    #initialising the vector database
    if vector_store is None:

        ef = HuggingFaceEmbeddings(
            model_name = EMBEdding_Model,
            model_kwargs = {"trust_remote_code":True}
        )
        
        
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),

        )




def process_urls(urls):

    #initialising vector data Base and llm 
    yield("initialising vector Database and LLM")
    intialise_components()

    yield("empting the vector data base")
    #empting the vector data base
    vector_store.reset_collection()

    yield("loading the data")
    #loading the data from the urls
    loader = WebBaseLoader(urls)
    data = loader.load()
    

    yield("Performing Chunking")
    ## chunkinhg the data 
    splitter = RecursiveCharacterTextSplitter(
        
            separators=["\n\n","\n","."," "],
            chunk_size = CHUNK_SIZE
        
    )
    docs = splitter.split_documents(data)

    yield("Creating indexes for the chunks")
    # Creating indexes for the docs with unique identifier
    uuids = [str(uuid4()) for _ in range(len(docs))]


    yield("adding the chunks to the vector database")
    #adding the documents into the vector database collection
    vector_store.add_documents(docs,ids=uuids)


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("vector DB is not initialised")
    chain =RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources =result.get("sources", "")
    
    return result["answer"], sources




if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    process_urls(urls)

    prompt = "tell me what was the 30 year fixed mortgage rate along with the date?"
    answer,sources = generate_answer(prompt)

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")