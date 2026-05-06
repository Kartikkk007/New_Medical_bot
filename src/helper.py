from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#extract the data from the pdf
def load_pdf_file(data):
    loader= DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#split the data into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_huggingface_embeddings():
    embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings