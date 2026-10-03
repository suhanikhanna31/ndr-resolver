# NDR Reply Resolver (v2)

Turns a customer's free-text or voice-transcript reply to a failed delivery (NDR) into a **safe, carrier-ready action**,
and **learns from human corrections**. A prototype of the "LLM decision making + evals + feedback loop" layer behind
an NDR voice agent like ClickPost's Parth.

**Scope:** one decision. Given `(ndr_reason, customer reply)`, return `reattempt | update_address | update_phone | rto | escalate_human`.

## Architecture

```
                  webhook                            ┌───────────── Postgres + pgvector ─────────────┐
 voice agent ───► POST /v1/ndr/jobs ─► Redis (arq) ─►│ decisions (PK awb+attempt = idempotency key)   │
      │              (job id = awb:attempt)   │      │ resolved_examples (embeddings, LangChain)      │
      │ sync                                  ▼      └───────────────▲───────────────────▲────────────┘
      └──► POST /v1/ndr/resolve ────► DecisionService                │ similar cases     │ human resolutions
                                          │                          │                   │
                       retrieve similar ──┘                          │      POST /v1/ndr/feedback
                       human-resolved cases ──► LLM extractor ──► validation ──► policy engine ──► Decision
                                                (understands)                     (decides, pure code)
                                                      │ any failure                      │
                                                      └──────────────► escalate_human ◄──┘
```

| Layer | Responsibility | File |
|---|---|---|
| Extractor | Language understanding only. Forced tool call, schema-validated, one bounded retry | `app/llm.py`, `app/prompts.py` |
| Retrieval (LangChain) | Embed human resolutions, fetch similar ones as few-shot examples | `app/retrieval.py` |
| Policy | All business rules: confidence floors, window, attempt cap, validation | `app/resolver.py` |
| Persistence | Decision log, idempotency, SQL analytics | `app/db.py` |
| Service | One code path for sync API and queue worker | `app/service.py` |
| Queue worker | arq on Redis, retry with backoff on DB errors | `app/worker.py` |
| API | Resolve, jobs, feedback, stats | `app/main.py` |
| Evals | Golden set + regex baseline + CI gate | `evals/` |

## Endpoints
| Method + path | Purpose |
|---|---|
| `POST /v1/ndr/resolve` | Synchronous decision (latency-sensitive callers) |
| `POST /v1/ndr/jobs` | Async decision. `202 queued`, or `200 duplicate` for a repeated `awb:attempt` |
| `GET /v1/ndr/jobs/{awb}:{attempt}` | `queued / in_progress / complete (+decision) / failed` |
| `POST /v1/ndr/feedback` | A human resolved or overrode a reply. Embeds it for future retrieval |
| `GET /v1/stats?days=7` | Escalation rate, avg confidence, p95 latency per intent (SQL) |

## Run it
```bash
cp .env.example .env            # add your ANTHROPIC_API_KEY
docker compose up --build       # postgres(pgvector) + redis + api + worker
```
```bash
# 1. a human resolves a reply -> it becomes a retrievable example
curl -s localhost:8000/v1/ndr/feedback -H 'content-type: application/json' -d '{
  "awb":"A1","ndr_reason":"customer_unavailable","customer_utterance":"kal shaam ko bhej do",
  "intent":"reschedule","date_expression":"tomorrow","time_slot":"evening"}'

# 2. async decision, then poll
curl -s localhost:8000/v1/ndr/jobs -H 'content-type: application/json' -d '{
  "awb":"B2","ndr_reason":"customer_unavailable","customer_utterance":"Bhai cancel mat karna, kal shaam ko bhej do"}'
curl -s localhost:8000/v1/ndr/jobs/B2:1
curl -s localhost:8000/v1/stats
```
Without Docker: start Postgres (with the `vector` extension) and Redis yourself, `pip install -r requirements.txt`, then
`uvicorn app.main:app` and, in another terminal, `arq app.worker.WorkerSettings`. If `DATABASE_URL` / `REDIS_URL` are unset the
service still works as a stateless zero-shot resolver (`/jobs`, `/feedback`, `/stats` return 503).

```bash
pytest -q       # 24 offline tests; +3 integration tests when DATABASE_URL and REDIS_URL are set
python -m evals.run_evals --provider baseline --no-gate   # free regex baseline
python -m evals.run_evals --provider anthropic            # real model, gated
```

## Key design choices
- **LLM extracts, code decides.** Rules are unit-testable and auditable; prompts are not.
- **Dates resolved in code.** The model emits `tomorrow` / `weekday:monday`, never a calendar date.
- **Asymmetric thresholds.** RTO is irreversible: 0.90 confidence. A reattempt: 0.75.
- **Fail safe.** API error, schema violation, missing tool call, or retrieval outage all degrade to a human or to zero-shot, never to a wrong action.
- **LangChain only where it earns its place:** the embeddings + vector-store abstraction (swap pgvector, or FastEmbed for OpenAI, by config). Extraction stays on the raw SDK: one structured call needs no framework.
- **pgvector in the same Postgres** as the decision log: one transactional store, no extra system. Revisit at millions of vectors.
- **Two-layer idempotency.** arq job id `awb:attempt` dedups within the result window; the `(awb, attempt)` primary key covers everything older, and a repeated call is answered from Postgres with no LLM call.
- **arq over Celery/Kafka.** asyncio-native, so one worker holds many in-flight LLM calls instead of one process each.
- **Local multilingual embeddings** (fastembed, ONNX, CPU). Anthropic has no embeddings endpoint, and this adds no second vendor key. Replies never leave the network for embedding.
- **Retrieved text is untrusted.** Customer-written examples are rendered as data, tag characters are stripped, and the system prompt says they never override the rules.
- **PII.** Phone numbers and pincodes are redacted before embedding; free-text addresses are never indexed. Logs carry decisions, not utterances.
- **Feedback endpoint is a poisoning surface.** Whoever can POST feedback controls few-shot examples, so set `API_KEY` in any shared environment.

## Verified vs not verified
| | Status |
|---|---|
| Unit tests, policy, dates, redaction, prompt safety | Pass |
| Postgres 16 + pgvector 0.6 + Redis 7: feedback → retrieval, idempotency, queue dedup, worker, stats | Pass (integration tests, real services) |
| `uvicorn` + `arq` as separate processes, end to end | Run manually, fail-safe path verified with a dummy API key |
| Call kwargs checked against the installed Anthropic SDK signature | Pass (regression test) |
| **Real Claude responses, retrieval quality on real Hinglish, `docker compose up --build`** | **Not run by the author.** Run the evals with your key |

Known limits: `RETRIEVAL_MAX_DISTANCE=0.35` is untuned. Tune it on held-out data, and keep golden eval cases out of the feedback
store or the evals will be inflated. Changing the embedding model means re-embedding (drop `resolved_examples`). Add an HNSW index
(`aapply_vector_index`) once the table passes ~10k rows.

## Eval results (30 cases, 8 adversarial: negation, self-correction, implicit addresses)
| System | Accuracy | Wrong auto-actions |
|---|---|---|
| Regex baseline | 73.3% | 13.3% (wrongly cancels "don't cancel" replies) |
| LLM + policy | *run `evals.run_evals --provider anthropic` and paste here* | gate: 0% |

## Next steps
Review queue for escalated cases, HNSW index + threshold tuning, retrieval-on/off eval comparison, Prometheus metrics, Hindi-script and regional-language cases.
