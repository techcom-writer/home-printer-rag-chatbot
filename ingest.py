"""
Document Ingestion Script for Home Printer Documentation
Processes markdown files and uploads embeddings to Pinecone.

Usage:
    python ingest.py

This script should be run once to populate your vector database.
It will:
1. Read all markdown files from the docs/ directory
2. Split them into chunks
3. Generate embeddings using Hugging Face
4. Upload to Pinecone
"""

import os
import re
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple
from pinecone import Pinecone
from tqdm import tqdm
import argparse

from dotenv import load_dotenv
load_dotenv()
PINECONE_INDEX = "home-printer-docs"
# ============= CONFIGURATION =============
DOCS_DIR = Path(__file__).parent.parent / "docs"
# HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "home-printer-docs")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# HF_EMBED_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# Chunk size for splitting documents
CHUNK_SIZE = 500  # tokens (approximate)
CHUNK_OVERLAP = 50

# ============= HELPER FUNCTIONS =============

def load_markdown_files() -> Dict[str, str]:
    """Load all markdown files from docs directory."""
    documents = {}
    
    if not DOCS_DIR.exists():
        print(f"Error: Docs directory not found at {DOCS_DIR}")
        return documents
    
    for md_file in DOCS_DIR.glob("*.md"):
        if md_file.name == "index.md":
            continue  # Skip index
        
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
                    documents[md_file.stem] = body
                else:
                    documents[md_file.stem] = content
            else:
                documents[md_file.stem] = content
    
    print(f"Loaded {len(documents)} markdown files")
    return documents

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    # Split by sentences or paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            if len(current_chunk) > overlap // 5:
                current_chunk = current_chunk[-(overlap // 5):]
                current_length = len(" ".join(current_chunk).split())
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += len(words)
    
    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks

# def get_embeddings(texts: List[str]) -> List[List[float]]:
#     """Get embeddings from Hugging Face."""
#     if not HUGGINGFACE_API_KEY:
#         raise ValueError("HUGGINGFACE_API_KEY not set")
    
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#     embeddings = []
    
#     for text in tqdm(texts, desc="Generating embeddings"):
#         try:
#             response = requests.post(
#                 HF_EMBED_URL,
#                 headers=headers,
#                 json={"inputs": text},
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 embedding = response.json()
#                 embeddings.append(embedding)
#                 time.sleep(0.1)  # Rate limiting
#             else:
#                 print(f"Error for text chunk: {response.status_code}")
#                 embeddings.append([0.0] * 384)  # Default dimension for bge-small-en
        
#         except Exception as e:
#             print(f"Error getting embedding: {e}")
#             embeddings.append([0.0] * 384)
    
#     return embeddings

from sentence_transformers import SentenceTransformer

# Load model once globally
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def get_embeddings(texts):
    """Generate embeddings locally."""
    
    embeddings = embedding_model.encode(texts)

    return [embedding.tolist() for embedding in embeddings]


def upload_to_pinecone(documents: Dict[str, str]) -> None:
    """Upload documents and embeddings to Pinecone."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        index = pc.Index(PINECONE_INDEX)
        print(f"Connected to Pinecone index: {PINECONE_INDEX}")
    except Exception as e:
        print(f"Error connecting to index: {e}")
        print(f"Please create index '{PINECONE_INDEX}' in Pinecone console first")
        return
    
    # Prepare vectors for upload
    vectors = []
    
    for doc_id, content in tqdm(documents.items(), desc="Processing documents"):
        chunks = split_into_chunks(content)
        
        # Get embeddings for this document's chunks
        embeddings = get_embeddings(chunks)
        
        # Create vectors with metadata
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{doc_id}_chunk_{i}"
            metadata = {
                "document": doc_id,
                "chunk_index": i,
                "text": chunk[:1000],  # Store first 1000 chars for reference
                "full_text": chunk
            }
            vectors.append((vector_id, embedding, metadata))
    
    # Upload to Pinecone in batches
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    print(f"\nSuccessfully uploaded {len(vectors)} vectors to Pinecone")
    print(f"Index: {PINECONE_INDEX}")

        
    pc = Pinecone(api_key=PINECONE_API_KEY)

    try:
        index = pc.Index(PINECONE_INDEX)

        print(f"Connected to Pinecone index: {PINECONE_INDEX}")

        # DEBUG INDEX STATS
        stats = index.describe_index_stats()

        print("\n===== INDEX STATS =====")
        print(stats)
        print("=======================\n")

    except Exception as e:
        print(f"Error connecting to index: {e}")
        return



    
# ============= MAIN =============

def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest Home Printer docs to Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload, just show what would happen")
    args = parser.parse_args()
    
    print("Starting Document Ingestion Pipeline")
    print(f"Docs directory: {DOCS_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Pinecone index: {PINECONE_INDEX}")
    print()
    
    # Load documents
    documents = load_markdown_files()
    
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Split into chunks
    all_chunks = {}
    total_chunks = 0
    for doc_id, content in documents.items():
        chunks = split_into_chunks(content)
        all_chunks[doc_id] = chunks
        total_chunks += len(chunks)
        print(f"  {doc_id}: {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {total_chunks}")
    
    if args.dry_run:
        print("DRY RUN MODE - Not uploading to Pinecone")
        return
    
    # Upload to Pinecone
    print("\nUploading to Pinecone...")
    upload_to_pinecone(documents)
    print("\nIngestion complete!")

if __name__ == "__main__":
    main()
