# NDR Reply Resolver

Turns a customer's free-text or voice-transcript reply to a failed delivery (NDR) into a **safe, carrier-ready action**.
Built as a prototype of the "LLM-powered decision making + evals" layer behind an NDR voice agent like ClickPost's Parth.

**Scope:** one decision. Given `(ndr_reason, customer reply)`, return `reattempt | update_address | update_phone | rto | escalate_human`.

## Architecture

```
reply (Hinglish/English) ──► LLM extractor ──► Pydantic validation ──► policy engine ──► Decision
                              (understands)     (schema + date grammar)  (decides, pure code)
                                   │ any failure                              │
                                   └──────────────► escalate_human ◄──────────┘
```

| Layer | Responsibility | File |
|---|---|---|
| Extractor | Language understanding only. Forced tool call, temperature 0 | `app/llm.py`, `app/prompts.py` |
| Date grammar | LLM emits `tomorrow` / `weekday:monday`, never a calendar date | `app/schemas.py`, `app/dates.py` |
| Policy | All business rules: confidence floors, window, attempt cap, validation | `app/resolver.py` |
| API | `POST /v1/ndr/resolve` | `app/main.py` |
| Evals | Golden set + regex baseline + CI gate | `evals/` |

## Key design choices
- **LLM extracts, code decides.** Rules are unit-testable and auditable; prompts are not.
- **Dates resolved in code.** Models are unreliable at calendar math, and "kal" is ambiguous in Hindi.
- **Asymmetric thresholds.** RTO is irreversible so it needs 0.90 confidence; a reattempt needs 0.75.
- **Fail safe.** API error, schema violation or missing tool call all route to a human.
- **Address changes always go to human review** (fraud vector) and need a pincode.
- **Prompt-injection aware.** The reply is delimited as untrusted data; injection attempts become `unclear`.
- **Eval gate on *unsafe auto-actions*, not just accuracy.**

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env && export ANTHROPIC_API_KEY=sk-ant-...
pytest -q                                              # 13 offline tests, no API key needed
python -m evals.run_evals --provider baseline --no-gate   # free regex baseline
python -m evals.run_evals --provider anthropic         # real model, gated
uvicorn app.main:app --reload
```
```bash
curl -s localhost:8000/v1/ndr/resolve -H 'content-type: application/json' -d '{
  "awb":"123","ndr_reason":"customer_unavailable",
  "customer_utterance":"Bhai cancel mat karna, kal shaam ko bhej do"}'
```

## Eval results (30 cases, 8 of them adversarial: negation, self-correction, implicit addresses)
| System | Accuracy | Wrong auto-actions |
|---|---|---|
| Regex baseline | 73.3% | 13.3% (wrongly cancels "don't cancel" replies) |
| LLM + policy | *run `evals.run_evals --provider anthropic` and paste here* | gate: 0% |

## Next steps
Sample real Parth transcripts into the golden set, add Hindi-script and regional-language cases, log human overrides as new eval cases, A/B models via `--model`, add a confidence calibration plot.
