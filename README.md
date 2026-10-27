# Basic HelpDeskQA RAG — ChromaDB + Gemini

A small, explainable Retrieval-Augmented Generation (RAG) project. It has five knowledge-base sections, semantic retrieval with persistent ChromaDB, a rejection threshold, a Streamlit UI, and a retrieval evaluation script.

## Project structure

```text
basic_rag_project/
├── app.py                         # Streamlit chat interface
├── evaluate.py                    # Hit@3, MRR@10, threshold tuning
├── requirements.txt
├── .env.example                   # Copy to .env and add your key
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths and environment configuration
│   ├── ingest.py                  # Chunk, embed, and index articles
│   └── rag.py                     # Retrieve, threshold, and generate answer
└── data/
    └── kb_docs/
        ├── wifi/                  # 5 articles
        ├── passwords/             # 5 articles
        ├── software/              # 5 articles
        ├── hardware/              # 5 articles
        └── network_access/        # 5 articles
```

`chroma_db/` is created after indexing. It is ignored by Git because it is generated data.

## 1. Prerequisites

- Python 3.11 recommended.
- A Gemini API key for answer generation. Retrieval evaluation does not call Gemini.
- Internet access the first time you install packages and download the embedding model.

## 2. Create and activate a virtual environment (Windows PowerShell)

Open PowerShell in the folder containing `basic_rag_project` and run:

```powershell
cd basic_rag_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 3. Configure Gemini

Copy `.env.example` to a new file named `.env` in the project root. Put your actual key in `.env`:

```text
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
SIMILARITY_THRESHOLD=0.30
TOP_K=3
```

Never commit `.env` or share your API key. If you only want to build the index and evaluate retrieval, a Gemini key is not needed.

## 4. Build the ChromaDB index

Run from the project root:

```powershell
python -m src.ingest
```

The script:
1. Reads all 25 Markdown articles.
2. Splits them by subsection/paragraph.
3. Prepends the article title and section to each chunk.
4. Embeds each chunk using `all-MiniLM-L6-v2`.
5. Stores vectors and metadata (`section`, `article_title`, `article_id`, source file, and chunk index) in a persistent ChromaDB collection using cosine distance.

Re-run ingestion whenever you edit or add knowledge-base articles. It rebuilds the collection to avoid stale chunks.

## 5. Run retrieval evaluation and tune the threshold

```powershell
python evaluate.py
```

This uses 28 in-scope and 4 out-of-scope questions. It does not call Gemini. It reports:

- **Hit@3:** percentage of in-scope queries where the expected article ID appears among the three highest-ranked chunks.
- **MRR@10:** mean reciprocal rank of the first chunk from the expected article, considering up to ten retrieved chunks. A first-place result contributes 1, second place 1/2, third place 1/3, and a missing article contributes 0.
- **Threshold tuning:** tries boundaries between observed best-match similarity scores and selects the threshold with the best balanced accuracy for separating in-scope and out-of-scope questions. The selected threshold is written to `data/similarity_threshold.json` and is used by the app.

The included questions are a starter test set, not independent production validation. High scores can be optimistic because these examples were authored alongside the demo knowledge base. Add new, unseen questions and review false accepts/false rejects before relying on the result.

## 6. Launch the UI

After indexing and (optionally) running evaluation:

```powershell
streamlit run app.py
```

Ask a question in the chat box. The app shows the generated answer, article titles and IDs returned by Gemini, best retrieval similarity, active threshold, and an expandable view of retrieved chunks. If the top result is below the threshold, it returns exactly: `I couldn’t find this.`

## How the RAG pipeline works

1. **Chunking:** article subsections/paragraphs become chunks. Each chunk starts with its article title and section.
2. **Embedding:** Sentence Transformers converts each chunk to a numeric vector.
3. **Vector database:** ChromaDB persists vectors, text, and metadata. Its collection uses cosine distance.
4. **Retrieval:** the question is embedded with the same model; ChromaDB returns the top-K most similar chunks.
5. **Rejection threshold:** if the best cosine similarity is below the tuned threshold, generation is skipped.
6. **Generation:** Gemini gets only the retrieved context and is instructed to return an answer plus the article IDs used.
7. **Citations:** the UI resolves those IDs to article titles from retrieved metadata.
8. **Evaluation:** labelled test questions measure ranking quality and tune the threshold for answerability detection.

## Interview-ready explanation

“This is a small IT helpdesk RAG system. I wrote a five-section knowledge base and split each article into paragraph-level chunks while preserving article titles and IDs. Sentence Transformers embeds the chunks, and ChromaDB stores the vectors and metadata using cosine distance. At query time, I embed the question, retrieve the top-K chunks, and reject the query if its best similarity falls below a threshold. Otherwise, Gemini receives the question and retrieved context and returns a grounded response with article IDs. I evaluate retrieval using Hit@3 and MRR, and tune the rejection threshold using labelled in-scope and out-of-scope questions.”

## Limitations to state honestly

- The KB is a demonstration set and should be replaced or reviewed for a real organization.
- Retrieval metrics do not prove that Gemini's generated answer is factually correct.
- Threshold tuning on a small set can overfit. Keep a separate unseen validation set for a stronger project evaluation.
- Similarity is a heuristic, not a probability that an answer is correct.
