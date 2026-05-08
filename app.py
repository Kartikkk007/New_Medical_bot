from flask import Flask, render_template, jsonify, request, session
from src.helper import download_huggingface_embeddings, load_pdf_file, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not configured")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
app.secret_key = FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

embeddings = download_huggingface_embeddings()

index_name = "medicalbot"

# 1. Initialize permanent vector store from existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Keep the global retriever and chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

ALLOWED_EXTENSIONS = {"pdf"}

# In-memory custom chains are keyed by browser session. For a multi-worker
# deployment, move this state to a shared store or rebuild from persisted docs.
custom_rag_chains = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_id():
    if "chat_session_id" not in session:
        session["chat_session_id"] = os.urandom(16).hex()
    return session["chat_session_id"]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Only PDF uploads are supported"}), 400

    with tempfile.TemporaryDirectory() as upload_dir:
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        try:
            # Load & split custom uploaded file
            extracted_data = load_pdf_file(data=upload_dir)
            text_chunks = text_split(extracted_data)
            if not text_chunks:
                return jsonify({"error": "No readable text found in the PDF"}), 400

            # Use an in-memory vector store for the session to prevent Pinecone rate limits
            vectorstore = FAISS.from_documents(text_chunks, embeddings)
            custom_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
            custom_rag_chains[get_session_id()] = create_retrieval_chain(custom_retriever, question_answer_chain)

            return jsonify({"message": f"Successfully processed {filename}. Chatbot will now query your document."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/reset_context", methods=["POST"])
def reset_context():
    custom_rag_chains.pop(get_session_id(), None)
    return jsonify({"message": "Context reset to default knowledge base"}), 200

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message.", 400
    
    # Check if a custom document is active, if so use it, otherwise use the global Pinecone base
    chain_to_use = custom_rag_chains.get(get_session_id(), rag_chain)
    response = chain_to_use.invoke({"input": msg})
    
    return str(response["answer"])

if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8080"))
    app.run(host=host, port=port, debug=debug)
