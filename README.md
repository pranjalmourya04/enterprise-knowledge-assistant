# Enterprise AI Knowledge Assistant

RAG-based internal document Q&A system with role-based access control and
answer verification. Built as a 4-week interview-prep project.

## Status: Week 1, Day 1-2 complete

- [x] FastAPI project structure
- [x] PDF upload endpoint
- [x] Text extraction (pdfplumber)
- [x] Chunking (per-page, word-window based, with overlap)
- [ ] Embeddings + ChromaDB storage (Day 3-4)
- [ ] Query/retrieval endpoint (Day 5)
- [ ] LLM answer generation (Day 6)
- [ ] Citations (Day 7)

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive API docs at
`http://127.0.0.1:8000/docs`.

## Design decisions

**Chunking stays within page boundaries.** Each chunk is built from a single
page's text (with word-window overlap within that page), rather than
chunking across the whole document. This keeps every chunk's page-number
citation exact. Trade-off: a paragraph spanning a page break may end up
split across two chunks — acceptable for this project's scope.
