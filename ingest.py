"""
STAGE 1 - PART A: INDEXING
PDFs  ->  text  ->  chunks  ->  embeddings  ->  ChromaDB

Run once (and again whenever you add papers or change chunk settings):
    python ingest.py
"""
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------
# Step 1: LOAD - pull raw text out of every PDF, page by page.
# We keep the file name and page number as "metadata" so the
# final answer can cite where each fact came from.
# ---------------------------------------------------------------
def load_pdfs(folder):
    pages = []
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = " ".join(text.split())  # collapse newlines and extra spaces
            if len(text) > 50:             # skip blank / figure-only pages
                pages.append({"text": text, "source": pdf_path.name, "page": i + 1})
        print(f"  loaded {pdf_path.name}  ({len(reader.pages)} pages)")
    return pages


# ---------------------------------------------------------------
# Step 2: CHUNK - split long text into overlapping windows of words.
#
#   words:   [w1 ... w250][w201 ... w450][w401 ... w650] ...
#                        ^^^^ overlap of 50 words ^^^^
#
# Overlap means a sentence cut at a boundary still appears whole
# in at least one chunk.
# ---------------------------------------------------------------
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):  # last window reached the end
            break
    return chunks


def main():
    print("Step 1: loading PDFs ...")
    pages = load_pdfs(config.PAPERS_DIR)
    if not pages:
        print(f"No PDFs found in '{config.PAPERS_DIR}/'. Add some papers and retry.")
        return

    print("Step 2: chunking ...")
    chunks, metadatas, ids = [], [], []
    for p in pages:
        for j, chunk in enumerate(chunk_text(p["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(chunk)
            metadatas.append({"source": p["source"], "page": p["page"]})
            ids.append(f'{p["source"]}-p{p["page"]}-c{j}')  # unique id per chunk
    print(f"  {len(pages)} pages -> {len(chunks)} chunks")

    # -----------------------------------------------------------
    # Step 3: EMBED - turn every chunk into a vector.
    # normalize_embeddings=True scales each vector to length 1,
    # which makes distance comparisons equivalent to cosine similarity.
    # (First run downloads the model, ~130 MB.)
    # -----------------------------------------------------------
    print("Step 3: embedding (first run downloads the model) ...")
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True)
    print(f"  embedding shape: {embeddings.shape}  (chunks x dimensions)")

    # -----------------------------------------------------------
    # Step 4: STORE - save vectors + text + metadata in ChromaDB.
    # We rebuild the collection from scratch each run so old chunks
    # (from different chunk settings) never mix with new ones.
    # -----------------------------------------------------------
    print("Step 4: storing in ChromaDB ...")
    client = chromadb.PersistentClient(path=config.DB_DIR)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # collection didn't exist yet
    collection = client.create_collection(config.COLLECTION)

    batch = 1000  # Chroma limits how many items one add() call can take
    for s in range(0, len(chunks), batch):
        collection.add(
            ids=ids[s:s + batch],
            documents=chunks[s:s + batch],
            embeddings=embeddings[s:s + batch].tolist(),
            metadatas=metadatas[s:s + batch],
        )
    print(f"Done. {collection.count()} chunks stored in '{config.DB_DIR}/'.")


if __name__ == "__main__":
    main()
