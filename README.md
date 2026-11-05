# RAG Paper Assistant

A retrieval-augmented generation (RAG) system that answers questions about a collection of research papers, with cited sources, plus a hand-built benchmark and an evaluation pipeline that measures retrieval quality, refusal behaviour, answer correctness and faithfulness.

The pipeline is written from scratch (no LangChain or LlamaIndex) so that every stage, from chunking to retrieval to prompting, is visible and can be measured and changed independently.

This is Stage 1–2 of a larger project. The next stage adds a knowledge graph over the same papers, to test whether graph-based retrieval fixes the failure modes found here (see [Roadmap](#roadmap)).

## How it works

**Indexing (`ingest.py`, run once):** PDFs are read page by page with `pypdf`, split into 250-word chunks with 50-word overlap, embedded with `BAAI/bge-small-en-v1.5`, and stored with their file name and page number in ChromaDB.

**Querying (`ask.py`):** the question is embedded with the same model, the top-k most similar chunks are retrieved, and they are passed to an LLM (`gpt-oss-20b` via Groq) with instructions to answer only from the context, cite every fact as `[file.pdf, p.N]`, and refuse when the answer is not in the papers.

| Component | Choice |
|---|---|
| PDF parsing | pypdf |
| Chunking | 250 words, 50-word overlap, per page |
| Embeddings | BAAI/bge-small-en-v1.5 (384 dimensions, normalized) |
| Vector store | ChromaDB (persistent, on disk) |
| Answer LLM | openai/gpt-oss-20b via Groq, temperature 0.1 |
| Judge LLM | openai/gpt-oss-120b via Groq, temperature 0 |

The LLM is called through the OpenAI-compatible API, so switching to Ollama, Gemini or OpenAI only requires editing `.env`.

## Setup

```bash
git clone https://github.com/SahithReddy-GitHub/rag-paper-assistant.git
cd rag-paper-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your API key
```

Put PDF files in a `papers/` folder (papers are not included in this repository; see [Corpus](#corpus)).

```bash
python ingest.py                   # build the index
python ask.py --show-chunks        # interactive Q&A, showing retrieved chunks
```

## Evaluation

```bash
python evaluate.py --name baseline             # retrieval metrics at k = 3, 5, 10
python evaluate.py --generate --name baseline  # also generate answers and check refusals
python judge.py eval/results/<run>_answers.json             # LLM-as-judge grading
python judge.py eval/results/<run>_judged.csv --agreement   # judge vs. human grades
```

### Benchmark

`eval/questions.json` contains 26 questions written by hand, each with the expected source paper(s), the pages that contain the answer, and a reference answer written from the PDFs:

| Type | Count | Tests |
|---|---|---|
| Single-paper | 15 | Methods and contributions of one paper (6 on the RADIANT, AutoFLUKA and LFR digital-twin papers) |
| Numeric | 4 | Exact figures, often from tables |
| Multi-paper | 4 | Comparisons and aggregation across papers |
| Out-of-scope | 3 | Unrelated, related-but-absent, and near-miss questions; the correct behaviour is refusal |

### Metrics

- **PaperHit@k / PageHit@k:** whether a correct paper / page appears in the top k chunks.
- **Coverage:** fraction of required papers retrieved (multi-paper questions).
- **MRR:** mean reciprocal rank of the first relevant chunk.
- **Refusal rates:** out-of-scope questions should be refused; in-scope refusals are false refusals.
- **Correctness (judge):** 0/1/2 against the reference answer, reported on a 0–1 scale.
- **Faithfulness (judge):** fraction of the answer's claims supported by the retrieved chunks. Claims that are true but come from the model's own memory count as unsupported.
- **Citation accuracy (judge):** fraction of citations that point to a chunk supporting the claim.

## Results (baseline)

### Retrieval, all in-scope questions (n = 23)

| k | PaperHit | PageHit | Coverage | MRR |
|---|---|---|---|---|
| 3 | 0.957 | 0.652 | 0.928 | 0.957 |
| 5 | 1.000 | 0.783 | 1.000 | 0.965 |
| 10 | 1.000 | 0.957 | 1.000 | 0.965 |

PageHit by question type at k = 5: single-paper 0.867, multi-paper 0.750, numeric 0.500.

### Refusals

| k | Correct refusals (out-of-scope) | False refusals (in-scope) |
|---|---|---|
| 3 | 3/3 | 6/23 (26%) |
| 5 | 3/3 | 3/23 (13%) |

### Answer quality at k = 5 (LLM judge)

| Metric | Value |
|---|---|
| Correctness (in-scope, 0–1) | 0.61 |
| Fully correct (in-scope) | 10/23 (43%) |
| Faithfulness (answered questions) | 0.90 |
| Judge–human agreement | TODO: X/10 exact agreement |

## Findings

1. **Paper-level retrieval is easy; page-level retrieval is not.** The right paper is retrieved for almost every question, but the page that actually contains the answer is missed for about 1 in 5 questions at k = 5, and for half of the numeric questions.

2. **Partial evidence triggers hallucination; missing evidence triggers refusal.** On a three-paper comparison, when only one paper's chunks were retrieved (k = 3) the model refused. When two of the three were retrieved (k = 5, in early manual tests), it answered confidently about all three, including the one it never saw. Partial context appears to give the model licence to fill gaps from memory.

3. **Similarity scores cannot separate relevant from irrelevant retrieval.** An out-of-scope question ("What does the Llama 3 paper say about RAG?") received a higher top similarity (0.693) than a genuinely relevant question about RADIANT (0.653), so a fixed similarity threshold cannot be used to detect unanswerable questions.

4. **Tables cause confident wrong numbers.** For DPR's top-20 accuracy (q17), the model reported incorrect figures (correctness 0). The results table extracts from the PDF as an unstructured run of numbers, and the model misreads it. A wrong number stated confidently is worse than a refusal.

5. **Correct is not the same as grounded.** Some answers are correct but only half supported by the retrieved text (q09, faithfulness 0.5): the model knows famous papers from training and fills in from memory, despite instructions to use only the context.

6. **References sections act as "hub" chunks.** The same late pages (reference lists and appendices) are retrieved across unrelated questions, because they are dense with paper titles and dataset names, and they displace real content.

7. **Chunks don't know which paper they belong to.** "What index does the original RAG paper use?" ranks the RAG paper 5th, behind other papers that discuss RAG in general, because the chunk containing the answer never mentions the paper's identity.

8. **Aggregation questions don't fit top-k retrieval.** "Which papers evaluate on HotpotQA?" requires checking every paper, but top-k retrieval returns the five most similar chunks, mostly reference lists that merely mention the dataset. This is a structural limitation rather than a tuning problem.

9. **More context lowers false refusals without adding false answers.** Moving from k = 3 to k = 5 halved false refusals (26% to 13%) while out-of-scope refusals stayed at 100%.

## Limitations

- Small benchmark (26 questions; 4 numeric and 4 multi-paper), so per-type results move by 25 points per question.
- Figures are ignored; tables and equations extract poorly with pypdf.
- Expected pages are hand-labelled, and deciding which pages "contain" an answer involves judgment.
- Most questions name the method they ask about, which makes paper-level retrieval easier than in real use.
- Refusals are detected by string matching.
- The judge is itself an LLM; its grades are being validated against human grades on a subset.

## Roadmap

- [ ] Remove references sections at ingestion and measure the effect
- [ ] Prepend paper titles to chunks (contextual chunk headers)
- [ ] Add a cross-encoder reranker
- [ ] Build a knowledge graph of methods, datasets, models and their relations (Neo4j), extracted with an LLM
- [ ] Hybrid GraphRAG: combine graph lookups with vector retrieval
- [ ] Compare vector RAG and GraphRAG on this benchmark, especially multi-paper and aggregation questions (findings 2 and 8)

## 📚 Research Papers

### Jian Tao / Nuclear Engineering and AI Research

1. **RADIANT-LLM: An Agentic Retrieval Augmented Generation Framework for Reliable Decision Support in Safety-Critical Nuclear Engineering**
2. **AutoFLUKA: A Large Language Model Based Framework for Automating Monte Carlo Simulations in FLUKA**
3. **Automating Monte Carlo Simulations in Nuclear Engineering with Domain Knowledge-Embedded Large Language Model Agents**
4. **Hierarchical Autoencoder-Based Lossy Compression for Large-Scale High-Resolution Scientific Data**
5. **Automating Monte Carlo Simulations in Nuclear Engineering with Domain Knowledge-Embedded Large Language Model Agents**
6. **Large Language Model-Assisted Digital Twin for Remote Monitoring and Control of Advanced Reactors**

### Foundational RAG and Knowledge Graph Papers

7. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
8. **Dense Passage Retrieval for Open-Domain Question Answering**
9. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**
10. **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**
11. **LightRAG: Simple and Fast Retrieval-Augmented Generation**
12. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
13. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
14. **Corrective Retrieval Augmented Generation**

## Repository structure

```
ingest.py          build the vector index from papers/
ask.py             interactive question answering
evaluate.py        retrieval metrics and answer generation over the benchmark
judge.py           LLM-as-judge grading and judge-human agreement
config.py          all settings in one place
eval/questions.json   the benchmark
observation.md     manual test notes from Stage 1
```
