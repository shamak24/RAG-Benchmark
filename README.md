# RAG Benchmarking (Capstone Phase 1)

Compare retrieval pipelines on a fixed academic PDF: how chunking, embeddings, and vector stores affect **finding the exact policy sentence** you marked as ground truth. The notebook also scores **chunk boundary quality** (SBI, MSCR) and whether gold evidence fits in a single chunk (ESI).

This is a **research benchmark**, not a chat product. The main score is **retrieval** (`evidence_hit_rate`), not fluent answers from a large LLM.

### Results at a glance (handbook run)

Screenshots below are exported from notebook **section 15** after a full 18-config run on `riverside_university_handbook.pdf` and `handbook_questions.json`.

| Figure file | What it shows |
|-------------|----------------|
| `evidence hit rate.png` | Mean `evidence_hit_rate` by chunker, embedder, and vector store |
| `chunk boundary integrity.png` | Mean `sbi_pct` vs `mscr_pct` by chunker |
| `config.png` | `evidence_hit_rate` heatmap: chunker × embedder (mean over FAISS and Chroma) |
| `sbi vs evidence hit rate.png` | Each of the 18 configs: SBI vs `evidence_hit_rate` |
| `correlation.png` | Per-config `evidence_hit_rate` vs local `context_recall` |

![Mean evidence hit rate by chunker, embedder, and vector DB](<evidence hit rate.png>)

![SBI and MSCR by chunker](<chunk boundary integrity.png>)

![Evidence hit rate heatmap](<config.png>)

![SBI vs evidence hit rate for all configs](<sbi vs evidence hit rate.png>)

![Evidence hit rate vs context recall per config](<correlation.png>)

---

## Use case

Universities and teams ship RAG over handbooks, manuals, and policy PDFs. Those documents are long, repetitive, and easy to split badly at chunk boundaries. If a rule is cut across two chunks, search may return the wrong paragraph even when the embedding model is strong.

**Use case for this repo:** test many RAG retrieval setups on the **same** handbook and **same** questions, so you can say which chunker + embedder + database works best for **exact-span** retrieval on dense policy text.

Typical questions this project answers:

- Does semantic chunking help retrieval if it looks “cleaner” on paper?
- Do FAISS and Chroma differ if embeddings are identical?
- Is MiniLM enough on a 60+ page domain PDF, or do BGE / E5 win clearly?

---

## Why this project exists

Standard RAG demos focus on generated answers. This capstone instead:

1. Holds the **document and questions constant**.
2. Swaps only **chunking**, **embedding model**, and **vector store** (18 combinations).
3. Measures whether the **annotated evidence string** appears in the **top-k** retrieved chunks.
4. Relates **chunk boundary metrics** to retrieval, without assuming they always align.

That matches a common capstone goal: controlled comparison on a realistic PDF corpus, not a one-off demo on a tiny FAQ.

---

## What is in the repository

| File | Purpose |
|------|---------|
| `RAG_Integrity_Benchmark.ipynb` | **Main deliverable.** Run the full experiment in Colab or Jupyter. |
| `riverside_university_handbook.pdf` | ~61-page synthetic **Riverside State University** academic handbook (2026-2027). Domain test corpus. |
| `handbook_questions.json` | **40** questions with `evidence_text` copied from the handbook (substring match after whitespace collapse). |
| `evidence hit rate.png`, `chunk boundary integrity.png`, `config.png`, `sbi vs evidence hit rate.png`, `correlation.png` | Result figures for reports and README (from section 15). |


---

## Architecture (how it works)

Same PDF and same `handbook_questions.json` for every run. Only the pipeline knobs change.

![Mean evidence hit rate by chunker, embedder, and vector DB](<arch diag.png>)


### The 18 configurations

| Axis | Values |
|------|--------|
| Chunker | `fixed` (500 tokens, 50 overlap), `recursive` (1000 chars, 200 overlap), `semantic` (MiniLM breakpoints, 90th percentile) |
| Embedder | `minilm` (all-MiniLM-L6-v2), `bge` (bge-small-en-v1.5), `e5` (e5-small-v2; uses `query:` / `passage:` prefixes) |
| Vector store | `faiss`, `chroma` |

3 x 3 x 2 = **18** rows in `results.csv`.

### What we do not run by default

- No official **RAGAS** `evaluate()` and no Groq judge on the full handbook (API limits; not needed for the main metric).
- QA columns in the CSV use **`local_overlap`**: token overlap plus MiniLM cosine, with RAGAS-like **names** only as proxies.

---

## Metrics (formulas in plain language)

All percentage integrity scores use the handbook text after PDF extract. Retrieval scores use **40** questions unless `SMOKE_TEST` is on.

### Preprocessing

- **Whitespace collapse:** all runs of spaces/newlines become one space before substring checks.

### Chunk boundary (per chunker)

- **SBI (`sbi_pct`):** Among chunk start/end edges that align in the document, what percent sit on an NLTK sentence boundary (within 8 characters slack)?
- **MSCR (`mscr_pct`):** `100 - SBI`. Higher means more mid-sentence cuts.
- **`mid_start_pct` / `mid_end_pct`:** Percent of chunks whose start or end is a mid-sentence cut.

### Evidence in chunks (per chunker)

- **ESI (`esi_pct`):** Percent of questions where the full `evidence_text` is contained in **one** chunk (any chunk in that chunker’s list). Does not depend on embedder or DB.

### Retrieval (per config; primary result)

- **`evidence_hit_rate`:** For each question, retrieve **k=3** chunks. Count a hit if `evidence_text` is a substring of the combined retrieved text (collapsed). Average over questions.
- **`retrieved_esi_rate`:** Same, but the evidence must fit entirely inside **one** of the top-k chunks, not across chunks merged.

### Local QA proxies (per config; secondary)

- **`context_precision`:** Average over questions: fraction of top-k chunks that share at least one token with gold/evidence.
- **`context_recall`:** Token recall of gold/evidence in retrieved text.
- **`faithfulness`:** Share of answer tokens that appear in retrieved text.
- **`answer_relevancy`:** Cosine similarity between MiniLM embeddings of question and answer, clipped to [0, 1].

### Timing

- **`index_time_s`:** Build index for that config.
- **`avg_query_time_s`:** Mean search time per question.

### Analysis

- Group means by chunker, embedder, or vector store.
- **Spearman** correlation (exploratory) between columns across 18 rows, e.g. SBI vs `evidence_hit_rate`. If a column is constant (ESI was 100% on the handbook), correlation is undefined.

---

## Handbook corpus (main test set)

- **Document:** `riverside_university_handbook.pdf` (fictional Riverside State University, academic rules, ~61 pages).
- **Questions:** `handbook_questions.json` (40 items). Each item should include:
  - `question`
  - `answer` / `ground_truth`
  - **`evidence_text`:** exact supporting sentence(s) from the PDF extract (required for ESI and hit rate).

The handbook is **synthetic** but written like continuous policy prose (not a FAQ sheet), so chunking and `pypdf` noise (e.g. page headers) matter.

---

## How to run

### Google Colab (recommended)

1. Upload `RAG_Integrity_Benchmark.ipynb`.
2. Run cells from the top. Install cell runs once per session.
3. In **Settings**, set `SMOKE_TEST = False` for the full 18-config run. Use `True` only for a quick check (2 configs, 2 questions).
4. In **Load PDF**, upload `riverside_university_handbook.pdf` and `handbook_questions.json`, or set Drive paths with `USE_DRIVE = True`.
5. Keep `USE_EXTRACTIVE_ONLY = True` unless you want FLAN-T5 answers (slower; retrieval remains the focus).
6. Section **13** runs the loop and writes `/content/results.csv` (download from Colab).

Expect the full handbook run to take noticeable time (embedding 18 indexes over the full chunk sets).

### Local Jupyter

Same notebook. Install dependencies from section 1. Set `PDF_PATH` and `QUESTIONS_PATH` to local paths. Colab-only `files.upload()` / `files.download()` will need to be skipped or adapted.

### Smoke test

- `SMOKE_TEST = True`
- Use `sample_plain_prose.pdf` and built-in or `questions.json` (6 questions), not the handbook file.

---

## Example results (handbook, full run)

The tables match the charts in the screenshots at the top of this README. Your numbers may vary slightly if you change the PDF or questions; these are from the completed handbook experiment:

| By chunker | Mean `evidence_hit_rate` |
|------------|---------------------------|
| recursive | ~0.78 |
| fixed | ~0.77 |
| semantic | ~0.68 |

| By embedder | Mean `evidence_hit_rate` |
|-------------|---------------------------|
| e5 | ~0.80 |
| bge | ~0.77 |
| minilm | ~0.67 |

| Vector store | Mean `evidence_hit_rate` |
|--------------|---------------------------|
| faiss / chroma | ~0.74 each (tie) |

**Best single config:** `fixed + e5 + faiss` or `chroma` at about **0.85** hit rate (34/40 questions).

**Integrity note:** Semantic chunking had **100% SBI** but not the highest hit rate. Fixed/recursive had low SBI (~14-17%) but strong retrieval with E5/BGE. **ESI was 100%** for all chunkers on this question set (every evidence span fit inside some chunk; that does not mean retrieval always picks that chunk).

---

## Project outcomes

1. A reproducible **18-way** benchmark on a **50+ page** academic-style PDF.
2. A grounded question set with **substring-verified** evidence.
3. Evidence that **embedding choice** mattered most in your runs; **FAISS vs Chroma** did not.
4. Evidence that **high sentence-boundary integrity does not guarantee** high evidence hit rate on this corpus.
5. Open notebook + CSV + figures suitable for a viva or report.

---

## Limitations

- Synthetic handbook, not a scanned real university PDF.
- `evidence_hit_rate` is strict substring match; page headers in extract can break spans if questions are not written carefully.
- QA metric columns are **local proxies**, not certified RAGAS scores.
- Answers are extractive-by-default; the report should center **retrieval**, not generation quality.
- 40 questions is enough to compare configs but not a large-scale NLP benchmark.

---
