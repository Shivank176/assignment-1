pip install fastapi uvicorn chromadb sentence-transformers python-multipart
RAG_Server/
├── main.py             # FastAPI application
├── db.py               # ChromaDB client setup
├── embeddings.py       # Embeddings model setup
├── models.py           # Pydantic models for request and response
└── requirements.txt    # Dependencies
# db.py
import chromadb

def get_chromadb_client():
    # Configure ChromaDB with persistence
    client = chromadb.Client(persistent=True)
    return client

def ingest_document(client, doc_id, embeddings):
    # Insert or update a document with its embeddings in the database
    client.upsert(doc_id, embeddings)

def query_documents(client, query_embedding):
    # Perform a similarity search in ChromaDB
    results = client.search(query_embedding, top_k=5)
    return results
# embeddings.py
from sentence_transformers import SentenceTransformer

def load_embeddings_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def generate_embeddings(model, text):
    embeddings = model.encode(text, show_progress_bar=False)
    return embeddings
# models.py
from pydantic import BaseModel
from typing import List

class DocumentIngestionRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    doc_id: str
    score: float
# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from db import get_chromadb_client, ingest_document, query_documents
from embeddings import load_embeddings_model, generate_embeddings
from models import DocumentIngestionRequest, QueryRequest, QueryResponse
from typing import List
import asyncio

app = FastAPI()
client = get_chromadb_client()
model = load_embeddings_model()

@app.post("/ingest/")
async def ingest_document_endpoint(request: DocumentIngestionRequest):
    embeddings = generate_embeddings(model, request.text)
    ingest_document(client, request.doc_id, embeddings)
    return {"message": "Document ingested successfully"}

@app.post("/query/", response_model=List[QueryResponse])
async def query_documents_endpoint(request: QueryRequest):
    query_embedding = generate_embeddings(model, request.query)
    results = query_documents(client, query_embedding)
    response = [{"doc_id": res['id'], "score": res['score']} for res in results]
    return response
uvicorn main:app --host 0.0.0.0 --port 8000
curl -X 'POST' \
  'http://127.0.0.1:8000/ingest/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_id": "doc1",
    "text": "Sample document text for ingestion."
  }'
curl -X 'POST' \
  'http://127.0.0.1:8000/query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "relevant query text"
  }'
