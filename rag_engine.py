"""
rag_engine.py — RAG (Retrieval-Augmented Generation) Engine
Uses:
  - sentence-transformers  → local embeddings (free, no API key needed)
  - chromadb               → in-memory vector store (free, runs locally)
  - groq                   → LLM generation via Llama-3 (free tier)
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path

_RAG_BASE_DIR = Path(__file__).parent


# ── 1. BUILD DOCUMENTS ─────────────────────────────────────────────────────────

def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each booking row into a rich text document for indexing.
    We create two types:
      a) Complaint documents  — for complaint-related queries
      b) Booking summary docs — for general booking queries
    """
    docs = []

    for _, row in df.iterrows():
        # ── Booking summary (all rows) ─────────────────────────────────────────
        booking_text = (
            f"Booking {row['booking_id']}: "
            f"{row['airline']} flight from {row['origin']} to {row['destination']} "
            f"({row['haul_type']}, {row['cabin_class']}, {row['trip_type']}). "
            f"Booked on {row['booking_date']} via {row['booking_channel']} ({row['booking_source']}). "
            f"Lead time: {row['lead_time_days']} days. "
            f"Passengers: {row['pax_count']}. "
            f"Total fare: ₹{row['total_fare_inr']:,.0f}. "
            f"Status: {row['booking_status']}. "
            f"Customer type: {'Repeat' if row['prior_bookings'] > 0 else 'New'} "
            f"(prior bookings: {row['prior_bookings']}). "
            f"Satisfaction score: {row['satisfaction_score'] if pd.notna(row['satisfaction_score']) else 'N/A'}."
        )
        docs.append({
            "id":       f"booking_{row['booking_id']}",
            "text":     booking_text,
            "metadata": {
                "type":           "booking",
                "booking_id":     str(row['booking_id']),
                "airline":        str(row['airline']),
                "origin":         str(row['origin']),
                "destination":    str(row['destination']),
                "booking_status": str(row['booking_status']),
                "cabin_class":    str(row['cabin_class']),
                "booking_channel":str(row['booking_channel']),
                "total_fare_inr": str(round(row['total_fare_inr'], 2)),
            }
        })

        # ── Complaint document (only if complaint exists) ──────────────────────
        complaint = str(row.get('customer_complaint', '')).strip()
        if complaint and complaint.lower() not in ('nan', 'none', ''):
            complaint_text = (
                f"Complaint from booking {row['booking_id']} "
                f"({row['airline']}, {row['booking_channel']}, "
                f"status: {row['booking_status']}): {complaint}"
            )
            docs.append({
                "id":       f"complaint_{row['booking_id']}",
                "text":     complaint_text,
                "metadata": {
                    "type":           "complaint",
                    "booking_id":     str(row['booking_id']),
                    "airline":        str(row['airline']),
                    "booking_channel":str(row['booking_channel']),
                    "booking_status": str(row['booking_status']),
                    "complaint_text": complaint[:200],
                }
            })

    return docs


# ── 2. RAG ENGINE CLASS ────────────────────────────────────────────────────────

class ClarityRAG:
    """
    Full RAG pipeline for the Clarity TTS booking dataset.

    Flow:
      index(df)      → embed all booking rows → store in ChromaDB
      query(q, key)  → embed question → retrieve top-k → ask Groq LLM → return answer
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"   # Fast, lightweight, 384-dim embeddings
    GROQ_MODEL  = "llama-3.1-8b-instant" # Free Groq model — fast & capable
    TOP_K       = 8                      # Number of documents to retrieve

    SYSTEM_PROMPT = """You are a data analyst assistant for Clarity Travel Technology Solutions (Clarity TTS),
a B2B travel technology company that processes airline bookings via NDC and GDS systems.

You have access to retrieved booking records from the Clarity dataset. 
Use ONLY the information provided in the retrieved context to answer the user's question.
If the context does not contain enough information to answer, say so clearly.

Be concise, professional, and data-driven. Format numbers clearly (e.g., ₹25,000, 32%).
When relevant, identify patterns or business implications of the data."""

    def __init__(self):
        self._embedder   = None
        self._chroma     = None
        self._collection = None
        self._indexed    = False

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.EMBED_MODEL)
        return self._embedder

    def _get_collection(self):
        if self._chroma is None:
            self._chroma = chromadb.PersistentClient(path=str(_RAG_BASE_DIR / "chroma_db"))
            self._collection = self._chroma.get_or_create_collection(
                name="clarity_bookings",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def is_indexed(self, df: pd.DataFrame) -> bool:
        """Check if the vector database is fully populated."""
        if self._indexed:
            return True
        collection = self._get_collection()
        # Fast expected count: 1 per booking + 1 per valid complaint
        expected = len(df) + df['customer_complaint'].apply(lambda c: 1 if pd.notna(c) and str(c).strip().lower() not in ('nan', 'none', '') else 0).sum()
        if collection.count() >= expected:
            self._indexed = True
            return True
        return False

    # ── Index ──────────────────────────────────────────────────────────────────
    def index(self, df: pd.DataFrame, progress_callback=None) -> int:
        """Build vector index from the dataframe. Returns number of documents indexed."""
        # Fast-path check
        if self.is_indexed(df):
            if progress_callback: progress_callback(1.0)
            return self._get_collection().count()

        docs = build_documents(df)
        collection = self._get_collection()
        embedder   = self._get_embedder()

        # Embed in batches
        BATCH = 128
        total = len(docs)
        for start in range(0, total, BATCH):
            batch = docs[start:start + BATCH]
            texts = [d["text"]     for d in batch]
            ids   = [d["id"]       for d in batch]
            metas = [d["metadata"] for d in batch]

            embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
            collection.upsert(ids=ids, embeddings=embeddings,
                              documents=texts, metadatas=metas)

            if progress_callback:
                progress_callback(min(start + BATCH, total) / total)

        self._indexed = True
        return total

    # ── Retrieve ───────────────────────────────────────────────────────────────
    def retrieve(self, question: str) -> list[dict]:
        """Embed the question and return top-k similar documents."""
        embedder   = self._get_embedder()
        collection = self._get_collection()

        q_vec = embedder.encode([question], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=q_vec,
            n_results=self.TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 4)   # cosine similarity
            })
        return retrieved

    # ── Generate ───────────────────────────────────────────────────────────────
    def generate(self, question: str, context_docs: list[dict], api_key: str) -> str:
        """Call Groq LLM with retrieved context to generate a grounded answer."""
        client = Groq(api_key=api_key)

        context_str = "\n\n".join(
            f"[Record {i+1} | similarity={d['similarity']}]\n{d['text']}"
            for i, d in enumerate(context_docs)
        )

        user_message = (
            f"Question: {question}\n\n"
            f"Retrieved Context (top {len(context_docs)} relevant records):\n"
            f"{'='*60}\n{context_str}\n{'='*60}\n\n"
            f"Please answer the question using only the context above."
        )

        response = client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=[
                {"role": "system",  "content": self.SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    # ── Full pipeline ──────────────────────────────────────────────────────────
    def ask(self, question: str, api_key: str) -> dict:
        """
        Full RAG pipeline: retrieve → generate.
        Returns: {"answer": str, "sources": list[dict]}
        """
        if not self._indexed:
            raise RuntimeError("Index not built yet. Call .index(df) first.")

        sources = self.retrieve(question)
        answer  = self.generate(question, sources, api_key)
        return {"answer": answer, "sources": sources}


# ── Singleton ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_rag() -> ClarityRAG:
    """Returns a globally cached instance of the RAG engine."""
    return ClarityRAG()
