"""
All settings in one place, so experiments in Stage 2 only need edits here.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# ---- Data & storage ----
PAPERS_DIR = "papers"        # put your PDF files here
DB_DIR = "chroma_db"         # ChromaDB saves the vector index here (on disk)
COLLECTION = "papers"        # like a "table" name inside ChromaDB

# ---- Embeddings ----
# bge-small: small, free, runs on CPU, handles up to 512 tokens per chunk.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models retrieve better when QUESTIONS (not documents) get this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- Chunking ----
CHUNK_SIZE = 250     # words per chunk (~330 tokens, safely under 512)
CHUNK_OVERLAP = 50   # words shared between neighbouring chunks

# ---- Retrieval ----
TOP_K = 5            # how many chunks to hand to the LLM

# ---- LLM ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = 0.1    # low = factual and consistent, which is what RAG wants
# Reasoning models (like gpt-oss) "think" before answering, and the thinking
# counts against MAX_TOKENS. Too small a budget -> empty answer.
MAX_TOKENS = 2048
REASONING_EFFORT = "low"   # only sent to gpt-oss models; ignored otherwise

# ---- LLM-as-a-judge (judge.py) ----
# Use a STRONGER model than the one being graded, and temperature 0 for consistency.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
JUDGE_MAX_TOKENS = 4096          # the judge lists every claim, so it needs room
JUDGE_REASONING_EFFORT = "medium"
