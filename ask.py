"""
STAGE 1 - PART B: RETRIEVE + GENERATE
question -> embed -> find similar chunks -> prompt the LLM -> cited answer

Run:
    python ask.py                 # interactive Q&A
    python ask.py --show-chunks   # also print the retrieved text (great for debugging)
"""
import argparse

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """You are a research assistant answering questions about research papers.
Rules:
- Use ONLY the information in the provided context. Do not use outside knowledge.
- If the context does not contain the answer, reply: "I couldn't find this in the papers."
- After each fact, cite its source in the form [file.pdf, p.N].
- Be concise and precise."""

# Load everything once at startup (loading models is slow; querying is fast).
embedder = SentenceTransformer(config.EMBED_MODEL)
collection = chromadb.PersistentClient(path=config.DB_DIR).get_collection(config.COLLECTION)
llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


# ---------------------------------------------------------------
# RETRIEVE: embed the question with the SAME model used for chunks,
# then ask Chroma for the k nearest chunk vectors.
# ---------------------------------------------------------------
def retrieve(question, k=config.TOP_K):
    q_vec = embedder.encode(config.QUERY_PREFIX + question, normalize_embeddings=True)
    res = collection.query(query_embeddings=[q_vec.tolist()], n_results=k)

    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        # Chroma returns squared L2 distance. For unit-length vectors:
        #   distance = 2 - 2*cosine   =>   cosine = 1 - distance/2
        hits.append({"text": text, "source": meta["source"],
                     "page": meta["page"], "score": 1 - dist / 2})
    return hits


# ---------------------------------------------------------------
# AUGMENT: paste the retrieved chunks into the prompt, each labelled
# with its source so the LLM can cite it.
# ---------------------------------------------------------------
def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{h['source']}, p.{h['page']}]\n{h['text']}" for h in hits
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


# ---------------------------------------------------------------
# GENERATE: send system rules + augmented prompt to the LLM.
# ---------------------------------------------------------------
def generate(prompt):
    # Reasoning models accept a "reasoning_effort" setting; other models
    # might reject it, so we only send it when the model is gpt-oss.
    extra = {}
    if "gpt-oss" in config.LLM_MODEL:
        extra["reasoning_effort"] = config.REASONING_EFFORT

    response = llm.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body=extra,
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        # Explain empty answers instead of failing silently.
        # finish_reason "length" = token budget ran out (often used up by reasoning).
        return (f"[Empty answer - finish_reason: {choice.finish_reason}, "
                f"tokens used: {response.usage.completion_tokens}]")
    return content


def answer(question, show_chunks=False):
    hits = retrieve(question)

    print("\nRetrieved chunks:")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['source']} p.{h['page']}  (similarity {h['score']:.3f})")
        if show_chunks:
            print(f"     {h['text'][:300]}...\n")

    print("\nAnswer:\n" + generate(build_prompt(question, hits)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    print(f"Loaded {collection.count()} chunks. Model: {config.LLM_MODEL}")
    print("Ask a question about your papers (type 'exit' to quit).")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer(q, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
