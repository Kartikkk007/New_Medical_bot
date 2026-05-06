from src.helper import (
    load_pdf_file,
    text_split,
    download_huggingface_embeddings
)

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY is not None, "PINECONE_API_KEY not found"

# Load and process PDFs
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)  

#  Load embeddings
embeddings = download_huggingface_embeddings()

#  Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

#  Create index ONLY if it does not exist
existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Store embeddings in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(" Pinecone index created and documents stored successfully")
