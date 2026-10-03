"""Run the golden set through the full pipeline (LLM + policy) and gate on accuracy AND safety.

    python -m evals.run_evals --provider anthropic --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.resolver import resolve
from app.schemas import CarrierAction, NDRRequest

HERE = Path(__file__).parent


def make_extractor(provider: str, model: str | None):
    if provider == "baseline":
        from .baseline import BaselineExtractor
        return BaselineExtractor()
    from app.llm import AnthropicExtractor
    return AnthropicExtractor(model=model)


def score(case, decision):
    exp = case["expected"]
    action_ok = decision.action.value == exp["action"]
    date_ok = ("reattempt_date" not in exp) or (
        decision.reattempt_date is not None and decision.reattempt_date.isoformat() == exp["reattempt_date"])
    correct = action_ok and date_ok
    # The metric that matters most: an automated carrier action that was wrong.
    unsafe = (decision.action is not CarrierAction.ESCALATE_HUMAN) and (not decision.needs_human) and not correct
    return correct, unsafe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "baseline"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases", default=str(HERE / "golden.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-accuracy", type=float, default=0.90)
    ap.add_argument("--max-unsafe", type=float, default=0.0)
    ap.add_argument("--no-gate", action="store_true", help="report only, always exit 0")
    a = ap.parse_args()

    cases = [json.loads(line) for line in Path(a.cases).read_text().splitlines() if line.strip()]
    extractor = make_extractor(a.provider, a.model)

    def run(case):
        t0 = time.perf_counter()
        d = resolve(NDRRequest(**case["request"]), extractor)
        return case, d, (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(run, cases))

    rows, correct_n, unsafe_n = [], 0, 0
    for case, d, ms in results:
        ok, unsafe = score(case, d)
        correct_n += ok
        unsafe_n += unsafe
        rows.append({"id": case["id"], "ok": ok, "unsafe": unsafe, "expected": case["expected"],
                     "got": {"action": d.action.value, "date": str(d.reattempt_date), "reasons": d.reasons},
                     "latency_ms": round(ms)})
        flag = "PASS" if ok else ("FAIL-UNSAFE" if unsafe else "FAIL")
        print(f"{case['id']:<5}{flag:<12}expected={case['expected']['action']:<16}got={d.action.value:<16}{d.reasons}")

    n = len(cases)
    acc, unsafe_rate = correct_n / n, unsafe_n / n
    lat = sorted(r["latency_ms"] for r in rows)
    print(f"\naccuracy={acc:.1%} ({correct_n}/{n})  unsafe_auto_action_rate={unsafe_rate:.1%}  p50_latency={lat[n // 2]}ms")

    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (out / f"{a.provider}-{stamp}.json").write_text(json.dumps(
        {"provider": a.provider, "model": a.model, "accuracy": acc, "unsafe_rate": unsafe_rate, "rows": rows}, indent=2))

    if a.no_gate:
        return 0
    return 0 if acc >= a.min_accuracy and unsafe_rate <= a.max_unsafe else 1


if __name__ == "__main__":
    sys.exit(main())
