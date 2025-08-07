import os
import io
import base64
import mimetypes
import time
import email
import requests
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pypdf import PdfReader
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load .env variables
load_dotenv()

# ENV vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db_session = SessionLocal()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    content = Column(Text)
    answer = Column(Text)

Base.metadata.create_all(engine)

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "docs"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

vector_index = pc.Index(index_name)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Helpers ----------

def extract_text(file, mimetype):
    if "pdf" in mimetype:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif "image" in mimetype:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    elif "text" in mimetype or "message" in mimetype or getattr(file, "name", "").endswith(".eml"):
        msg = email.message_from_bytes(file.read())
        if msg.is_multipart():
            parts = [part.get_payload(decode=True) for part in msg.walk() if part.get_content_type() == "text/plain"]
            return "\n".join([p.decode("utf-8", errors="ignore") for p in parts if p])
        else:
            return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    return ""

def chunk_text(text, max_chunk_size=500):
    words = text.split()
    chunks, current_chunk, current_len = [], [], 0
    for word in words:
        current_len += len(word) + 1
        current_chunk.append(word)
        if current_len >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

import json  # Make sure this is imported

def call_groq_llm_json(context, question):
    prompt = f"""
You are a helpful assistant analyzing an insurance policy document.

Answer the question strictly using the context provided.

Return the answer in the following JSON format ONLY:
{{
  "answer": "...",
  "explanation": "...",
  "clause_match": "..."
}}

If no answer is found, return:
{{
  "answer": "Not found",
  "explanation": "The document does not contain sufficient information.",
  "clause_match": ""
}}

Context:
{context}

Question: {question}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": prompt}]

    try:
        response = requests.post(url, headers=headers, json={
            "model": "llama3-8b-8192",
            "messages": messages
        })
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()

        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from triple-backtick code block
        if "```json" in text:
            try:
                json_part = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_part)
            except Exception:
                pass

        # Try from generic code block
        if "```" in text:
            try:
                json_part = text.split("```")[1].strip()
                return json.loads(json_part)
            except Exception:
                pass

        # Last resort: return raw string
        return {
            "answer": text,
            "explanation": "⚠️ Unable to parse structured format. Model may have returned plain text.",
            "clause_match": ""
        }

    except Exception as e:
        return {
            "answer": f"❌ Groq API Error: {e}",
            "explanation": "",
            "clause_match": ""
        }


def download_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        filename = os.path.basename(urlparse(url).path)
        file_like = io.BytesIO(response.content)
        file_like.name = filename
        return file_like, filename, response.headers.get("Content-Type")
    except Exception as e:
        raise RuntimeError(f"Failed to download file: {e}")

# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if not uploaded_file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(filepath)

        mimetype = mimetypes.guess_type(filename)[0] or uploaded_file.mimetype
        with open(filepath, "rb") as f:
            content = extract_text(f, mimetype)

        if not content.strip():
            return jsonify({"error": "Unsupported or empty file content"}), 400

        chunks = chunk_text(content)
        vectors = []
        for i, chunk in enumerate(chunks):
            vec = embedder.encode(chunk).tolist()
            vectors.append((f"{filename}_chunk_{i}", vec, {"text": chunk}))

        unique_ns = f"{filename}_{int(time.time())}"
        session['namespace'] = unique_ns
        session['chat_history'] = []
        vector_index.upsert(vectors=vectors, namespace=unique_ns)

        doc = Document(filename=filename, content=content[:1000], answer="")
        db_session.add(doc)
        db_session.commit()

        return jsonify({"status": "processed"})

    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    namespace = session.get("namespace")
    if not namespace:
        return jsonify({"answer": "❌ No document uploaded yet."}), 400

    try:
        query_vec = embedder.encode(question).tolist()
        results = vector_index.query(
            vector=query_vec,
            top_k=3,
            include_metadata=True,
            namespace=namespace
        )
        context = "\n\n".join([match["metadata"]["text"] for match in results.get("matches", [])])
    except Exception as e:
        return jsonify({"answer": f"❌ Pinecone query error: {e}"}), 500

    chat_history = session.get("chat_history", [])
    answer = call_groq_llm_json(context, question)["answer"]

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history

    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "reset"})

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    try:
        data = request.get_json()
        doc_url = data.get("documents")
        questions = data.get("questions", [])

        if not doc_url or not questions:
            return jsonify({"error": "Missing documents or questions"}), 400

        file, filename, mimetype = download_file_from_url(doc_url)
        content = extract_text(file, mimetype or "application/pdf")

        if not content.strip():
            return jsonify({"error": "Empty or unsupported file content"}), 400

        chunks = chunk_text(content)
        vectors = []
        for i, chunk in enumerate(chunks):
            vec = embedder.encode(chunk).tolist()
            vectors.append((f"{filename}_chunk_{i}", vec, {"text": chunk}))

        namespace = f"{filename}_{int(time.time())}"
        vector_index.upsert(vectors=vectors, namespace=namespace)

        results = []
        for q in questions:
            query_vec = embedder.encode(q).tolist()
            matches = vector_index.query(vector=query_vec, top_k=3, include_metadata=True, namespace=namespace)
            context = "\n\n".join([m["metadata"]["text"] for m in matches.get("matches", [])])
            structured = call_groq_llm_json(context, q)
            structured["question"] = q
            results.append(structured)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------

if __name__ == "__main__":
    app.run(debug=True)
