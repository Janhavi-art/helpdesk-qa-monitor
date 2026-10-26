# HelpDeskQA Monitor MVP

A small, explainable Retrieval-Augmented Generation (RAG) helpdesk app. It searches a local Markdown knowledge base with MiniLM + ChromaDB, sends a compact context to Gemini only when local retrieval is relevant, and tracks token usage and latency.

## What it does

- Loads the real IT ticket dataset (`Document`, `Topic_group`) when you add it to `data/tickets_raw.csv`; otherwise uses an explicitly labelled, small demo CSV so the app can run immediately.
- Profiles ticket categories and recurring phrases locally with Pandas and scikit-learn. This does **not** call Gemini.
- Supports one-click generation of a short knowledge-article draft from aggregate category keywords. The draft is marked unverified and is not indexed automatically.
- Indexes reviewed Markdown articles in local ChromaDB using `all-MiniLM-L6-v2` embeddings.
- Retrieves locally, applies a relevance threshold, limits context and output size, checks a local request/token budget, caches repeat questions, and records Gemini usage metadata.
- Includes a small retrieval/abstention evaluation set and a Streamlit monitoring view.

## Important data limitation

The ticket dataset has descriptions and categories but no verified resolution column. Ticket text tells us what people report, not which fix is correct. Therefore:

1. Use ticket profiles to decide which article topics to write.
2. Use the draft feature only as a writing aid.
3. Verify procedures against official vendor/organization documentation.
4. Move only reviewed articles into `data/kb_docs/`; `data/kb_drafts/` is not indexed.

The ZIP includes `data/demo_tickets.csv` with 32 clearly labelled example rows so the app and data-insights screen run immediately. It does **not** include the full 47,837-row file because that file was not attached to the chat. To use your real data, copy your CSV to `data/tickets_raw.csv`. It should contain `Document` and `Topic_group` columns (or already-normalized `text` and `category` columns). The program detects it automatically.

## 1. Install (Windows / VS Code)

Use Python 3.11 or 3.12. You can run `setup_windows.bat` to create `.venv` and install dependencies, or follow the manual commands below.

Open PowerShell at the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, you can run the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Add your Gemini API key

Run `Copy-Item .env.example .env` in PowerShell (or let `setup_windows.bat` create it), then edit `.env`:

```dotenv
GEMINI_API_KEY=your_real_api_key_here
```

Never share or commit `.env`. The code uses `GEMINI_MODEL=gemini-3.5-flash-lite` by default. If that model is not available for your API key/project, change it to an enabled model shown in Google AI Studio. Current model availability and limits can change; check https://ai.google.dev/gemini-api/docs/models.

## 3. (Recommended) Add the real ticket data

Copy your 47,837-row CSV to:

```text
data/tickets_raw.csv
```

The application uses this file instead of the demo data. Raw CSVs are ignored by Git to reduce the risk of publishing ticket contents.

## 4. Profile the tickets locally

```powershell
python -m src.profile_tickets
```

This creates:

- `data/processed/tickets_clean.csv`
- `data/processed/topic_profile.json`

It reports category counts and recurring phrases. It makes no Gemini API call.

## 5. Build the local vector index

```powershell
python -m src.ingest
```

This reads the generic starter Markdown articles under `data/kb_docs/`, splits them at headings/paragraphs where possible, creates MiniLM embeddings locally, and stores them in `chroma_db/`. Review and adapt the starter articles to official vendor/organization documentation before treating them as authoritative. The first run downloads the embedding model from Hugging Face and therefore requires an internet connection.

The app also creates the index automatically on the first relevant question if no index exists.

## 6. Start the application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (normally `http://localhost:8501`).

The tabs are:

- **Ask helpdesk** — answer a question and inspect retrieved passages and per-request usage.
- **Dataset insights** — inspect categories and run local phrase profiling.
- **Knowledge-base drafts** — generate a draft from aggregate keywords. One draft invokes Gemini once; draft output must be reviewed and is never indexed automatically.
- **Usage monitor** — view API calls, reported tokens, cache hits, local abstentions, latency, and evaluation results.

## Usage optimization implemented

1. **Local semantic retrieval first:** MiniLM + ChromaDB are used before any Gemini request.
2. **Local abstention:** if the best cosine similarity is below `MIN_RELEVANCE_SCORE`, the app returns a no-match message without calling Gemini.
3. **Context cap:** at most `RETRIEVAL_TOP_K` passages are selected, up to `MAX_CONTEXT_CHARS` characters.
4. **Output cap:** each answer request uses `MAX_OUTPUT_TOKENS`.
5. **Response cache:** repeated identical questions with the same experiment settings reuse the cached answer and make zero new Gemini calls.
6. **Local rate pacing:** requests are spaced by `API_MIN_INTERVAL_SECONDS` across app reruns on this machine.
7. **Application-side budget guard:** `MAX_DAILY_API_CALLS` limits API attempts logged by this project. `DAILY_TOKEN_BUDGET` compares prior reported usage with a rough local estimate of the next prompt plus output allowance before making a request.
8. **Usage metadata:** successful responses log input/output/thinking/cached/tool/total token fields when Gemini returns them.

The local token pre-check is an **estimate**, not a strict provider-enforced token cap. Exact token counts are known after the response. Provider-side quota and billing limits are separate. API failures are logged but may not return usage metadata.

## Baseline versus optimized experiment

Use the same `data/eval_set.json` for both runs.

### A. Baseline settings

In `.env`, set the following values. Keep the remaining key and budget settings intact:

```dotenv
EXPERIMENT_LABEL=baseline
RETRIEVAL_TOP_K=4
MAX_CONTEXT_CHARS=6000
MAX_OUTPUT_TOKENS=320
MIN_RELEVANCE_SCORE=0.0
ENABLE_RESPONSE_CACHE=false
```

Restart Streamlit if it's running, then run the evaluation with a fresh cache/result file:

```powershell
python -m src.run_eval --fresh --no-cache
```

### B. Optimized settings

Edit `.env` and change these experiment values:

```dotenv
EXPERIMENT_LABEL=optimized
RETRIEVAL_TOP_K=2
MAX_CONTEXT_CHARS=2500
MAX_OUTPUT_TOKENS=180
MIN_RELEVANCE_SCORE=0.30
ENABLE_RESPONSE_CACHE=true
```

Run:

```powershell
python -m src.run_eval --fresh --no-cache
```

Each experiment saves to its own file under `data/processed/`, such as `evaluation_results_baseline.csv` and `evaluation_results_optimized.csv`. Set appropriate request/token budgets for your API quota. The runner resumes completed questions; `--fresh` replaces the current experiment's results. `--no-cache` avoids reusing answers. Local pacing defaults to 12 seconds between API attempts; change only if your provider's actual limit safely permits more requests.

Compare **mean total Gemini tokens per answerable query**, **retrieval hit rate**, **correct abstention rate**, **latency**, and a manual review of a few answers. This automated runner does not claim to judge factual correctness; use human review against verified article content. An optimization only counts as a success if token use falls without unacceptable loss in answer quality.

Note: `--no-cache` disables cache reads/writes for the run. If you want to test repeat-query caching in the UI, leave `ENABLE_RESPONSE_CACHE=true` and ask the same question twice; the second response should show a cache hit and zero tokens for that interaction.

## Useful commands

```powershell
# Analyze the available ticket file locally
python -m src.profile_tickets

# Rebuild verified knowledge articles' index
python -m src.ingest

# Run evaluation and resume if interrupted
python -m src.run_eval

# Start the UI
streamlit run app.py

# Local unit tests (do not call Gemini)
python -m unittest discover -s tests
```

## Project structure

```text
HelpDeskQA_Monitor_MVP/
├── app.py
├── setup_windows.bat
├── run_app.bat
├── requirements.txt
├── .env.example
├── data/
│   ├── demo_tickets.csv
│   ├── eval_set.json
│   ├── kb_docs/                  # verified starter articles
│   ├── kb_drafts/                # unverified drafts; not indexed
│   ├── cache/                    # local response cache
│   ├── logs/                     # JSONL usage and interaction logs
│   └── processed/                # local profiles and evaluations
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── profile_tickets.py
│   ├── chunking.py
│   ├── ingest.py
│   ├── llm.py
│   ├── rag.py
│   ├── kb_drafts.py
│   └── telemetry.py
└── tests/
    └── test_helpers.py
```

## Dataset reference

The 47,837-row IT Service Ticket Classification dataset is described as containing `Document` (ticket text) and `Topic_group` (category), with eight categories. See the public dataset listing and original repository for attribution/context:

- Kaggle dataset: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset
- Public repository describing the dataset schema: https://github.com/Lunalykaya/IT-Service-Ticket-Classification

Use the dataset under its applicable licence and do not send confidential ticket data to an external LLM without authorization.
