# RAG Stage 2 Observations

## Test 1: RADIANT-LLM Overview

**Question:** What is RADIANT, what problem does it address, and what are its main contributions?

- Retrieved source: `Jian_Tao_RADIANT_LLM.pdf`
- Retrieved pages: 32, 8, 37, 31, 22
- Similarity scores: 0.653, 0.630, 0.619, 0.590, 0.590
- Correct source retrieved: Yes
- Answer relevance: Good
- Citations provided: Yes
- Claim-level factual verification: Pending

**Observation:** The RAG system retrieved relevant RADIANT-LLM content and generated a structured answer covering local-first architecture, multimodal retrieval, provenance, and nuclear 3S applications. Experimental performance claims require verification against the paper's results.

## Test 2: RADIANT-LLM Methodology

**Question:** What specific methodology does the RADIANT paper propose, and what are the key components of its framework?

- Retrieved source: `Jian_Tao_RADIANT_LLM.pdf`
- Retrieved pages: 37, 32, 22, 8, 7
- Similarity scores: 0.707, 0.694, 0.670, 0.655, 0.648
- Correct source retrieved: Yes
- Answer relevance: Good
- Citations provided: Yes
- Claim-level factual verification: Pending

**Observation:** The system retrieved relevant methodological content and organized the answer into framework components. Some security-related statements may be stronger than the retrieved evidence. Detailed methodology claims should be verified directly against the PDF.

**Overall:** Initial retrieval performance for RADIANT-LLM is promising. Further evaluation should focus on claim-level grounding, citation accuracy, and whether retrieved chunks contain sufficient evidence for the generated answer.

## Test 3: Out-of-Scope Question — FIFA World Cup

**Question:** Who won the 2022 FIFA World Cup?

- Retrieved papers: Self-RAG, HippoRAG, Corrective RAG, Jian Tao's Progress in Nuclear Energy paper
- Similarity scores: 0.627, 0.567, 0.554, 0.515, 0.514
- Relevant source retrieved: No
- Answer: "I couldn't find this in the papers."
- Hallucination observed: No
- Refusal behavior: Correct

**Observation:** The retriever returned unrelated chunks, including football biographies and unrelated research references. Despite irrelevant retrieval, the LLM correctly refused to answer. This indicates promising grounding and refusal behavior.

## Test 4: Related but Absent Question — Llama 3

**Question:** What does the Llama 3 paper say about RAG?

- Retrieved papers: LightRAG, Corrective RAG, GraphRAG, RADIANT-LLM
- Similarity scores: 0.693, 0.689, 0.671, 0.670, 0.669
- Relevant source retrieved: No exact Llama 3 source
- Answer: "I couldn't find this in the papers."
- Hallucination observed: No
- Refusal behavior: Correct

**Observation:** The retriever found papers discussing RAG generally, but not the requested Llama 3 paper. The assistant correctly refused to infer an answer from related but insufficient evidence.

**Overall:** Both out-of-scope tests demonstrate promising refusal behavior. The main weakness is that irrelevant chunks can receive relatively high similarity scores. Future experiments should evaluate retrieval thresholds and top-k settings.

## Test 5: Query Sensitivity

### Question A — Vague Query
**Question:** Explain RAG.

- Top similarity score: 0.696
- Retrieved papers: GraphRAG, LightRAG, Energy and AI
- Relevant source retrieved: Yes
- Answer quality: Good
- Citation accuracy: Partially verified

**Observation:** The vague query retrieved relevant RAG-related content. GraphRAG page 2 directly discusses RAG and context-window limitations. However, the Energy and AI citation may not directly support the complete RAG definition.

### Question B — Specific Query
**Question:** Explain how Retrieval-Augmented Generation works, including how documents are chunked, converted into embeddings, stored in a vector database, retrieved using a user query, and passed to an LLM to generate a grounded answer.

- Top similarity score: 0.838
- Retrieved papers: LightRAG and Corrective RAG
- Relevant source retrieved: Yes
- Answer quality: Good
- Citation accuracy: Partially verified

**Observation:** The detailed query produced higher similarity scores and a structured five-stage explanation covering chunking, embeddings, vector storage, retrieval, and generation. Some technical examples, such as FAISS and Milvus, may have been added from general model knowledge and require evidence verification.

**Overall:** Specific queries appear to retrieve more focused RAG-related content. However, similarity scores should not be treated as a direct measure of answer correctness. Citation-level verification is still required.


## Test 6: Multi-Paper Comparison

**Question:** How do GraphRAG, HippoRAG, and LightRAG differ in their approaches to knowledge graph construction and retrieval?

- Top similarity score: 0.805
- Retrieved papers: LightRAG and HippoRAG
- GraphRAG paper retrieved: No
- HippoRAG paper retrieved: Yes
- LightRAG paper retrieved: Yes
- Unique papers retrieved: 2 out of 3
- Answer structure: Good
- Claim-level grounding: Weak / partially verified

**Observation:** The retriever returned four chunks from LightRAG and one chunk from HippoRAG. The requested GraphRAG paper was absent from the top 5. The generated answer contained detailed descriptions of all three methods, but several claims about HippoRAG and GraphRAG were not directly supported by the retrieved chunks.

**Failure mode:** Multi-document retrieval imbalance and unsupported synthesis.

**Potential improvement:** Increase TOP_K, use query decomposition into separate subqueries for each paper, or add a reranking step to improve diversity and relevance across documents.



## Test 7: TOP_K = 10 — Multi-Paper Retrieval

**Question:** How do GraphRAG, HippoRAG, and LightRAG differ in their approaches to knowledge graph construction and retrieval?

- TOP_K: 10
- Retrieved papers: GraphRAG, HippoRAG, LightRAG
- GraphRAG paper retrieved: Yes
- HippoRAG paper retrieved: Yes
- LightRAG paper retrieved: Yes
- All requested papers retrieved: Yes
- Answer structure: Good
- Claim-level grounding: Partially verified

**Comparison with TOP_K = 5:** At TOP_K = 5, GraphRAG was absent from the retrieved results. At TOP_K = 10, GraphRAG appeared in the top 10, and all three requested papers were retrieved.

**Observation:** Increasing TOP_K improved multi-paper coverage. However, LightRAG still dominates the retrieved chunks. Some detailed claims about HippoRAG and GraphRAG are not directly supported by the supplied excerpts.

**Conclusion:** Increasing TOP_K improves recall/document coverage for this question, but does not necessarily guarantee more accurate or fully grounded answers. More retrieved chunks may also introduce irrelevant context.


### Experiment: Multi-paper comparison with TOP_K = 3

**Question:** How do GraphRAG, HippoRAG, and LightRAG differ in their approaches to knowledge graph construction and retrieval?

**Retrieved Sources:**
- LightRAG paper, pages 6, 8, and 11
- GraphRAG: Not retrieved
- HippoRAG: Not retrieved

**Observation:**
With TOP_K set to 3, all retrieved chunks came from the LightRAG paper. The retrieval results did not cover GraphRAG or HippoRAG, which were essential for answering the comparative question. The LLM responded with “I couldn't find this in the papers,” demonstrating cautious behavior and avoiding unsupported claims.

**Evaluation:**
The experiment showed poor retrieval coverage for multi-paper comparative questions. Increasing TOP_K from 3 to 5 and then 10 improved source diversity, with all three papers appearing at TOP_K = 10. However, the higher TOP_K result still requires claim-level verification to ensure the answer is grounded in the retrieved passages.

**Conclusion:**
TOP_K influences retrieval coverage and answerability. For comparative questions involving multiple papers, retrieval should ideally include relevant evidence from each requested source, rather than relying only on the highest-similarity chunks.



the k=5 retrieval row (PaperHit 1.000, PageHit 0.783, MRR 0.965) and generation at k=5 (correct refusals 1.0, false refusals 0.13). Every improvement from here gets compared against those numbers.

