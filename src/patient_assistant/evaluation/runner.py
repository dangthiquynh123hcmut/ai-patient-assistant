"""CLI chạy bộ đánh giá.

pa-eval                               # LLM offline, toàn bộ dataset
pa-eval --llm bedrock --suite booking # Claude trên Bedrock (tốn phí), chỉ suite booking
pa-eval --fail-on-hard-zero           # exit 1 nếu false-success / slot hallucination / double-booking > 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from patient_assistant.config import Settings
from patient_assistant.container import build_llm
from patient_assistant.evaluation.cases import load_cases
from patient_assistant.evaluation.harness import run_cases
from patient_assistant.evaluation.metrics import compute
from patient_assistant.evaluation.report import hard_zero_violations, to_json, to_markdown


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pa-eval", description="Đánh giá AI Patient Assistant trên dataset JSONL")
    p.add_argument("--llm", choices=["offline", "bedrock"], default="offline")
    p.add_argument("--dataset", type=Path, nargs="*", help="File JSONL (mặc định: dataset đóng gói kèm)")
    p.add_argument("--suite", choices=["report", "booking", "conversation"], action="append")
    p.add_argument("--case", action="append", help="Chỉ chạy case có id này (lặp được)")
    p.add_argument("--out", type=Path, default=Path("eval_reports"))
    p.add_argument("--concurrency", type=int, default=4, help="Số case chạy song song")
    p.add_argument("--fail-on-hard-zero", action="store_true")
    return p


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    if args.suite:
        cases = [c for c in cases if c.suite in args.suite]
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    if not cases:
        print("Không có case nào khớp bộ lọc.", file=sys.stderr)
        return 2
    settings = Settings(llm_provider=args.llm)
    observations = await run_cases(cases, lambda: build_llm(settings), concurrency=args.concurrency)
    report = compute(cases, observations)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "run": stamp,
        "llm": args.llm,
        "model": settings.llm_model_id if args.llm == "bedrock" else "offline-rules",
        "cases": len(cases),
    }
    out_dir = args.out / f"{stamp}-{args.llm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(to_markdown(report, meta), "utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(to_json(report, observations, meta), ensure_ascii=False, indent=2), "utf-8"
    )

    print(
        f"Case đạt: {report.case_pass_rate.numerator}/{report.case_pass_rate.denominator} · "
        f"false-success={report.false_success.numerator} · slot-hallucination={report.slot_hallucination.numerator}"
        f" · double-booking={report.double_booking.numerator}"
    )
    for v in report.verdicts:
        if not v.passed:
            print(f"  ✗ {v.case_id}[{v.session}]: {'; '.join(v.failures)}")
    print(f"Báo cáo: {out_dir / 'report.md'}")
    if args.fail_on_hard_zero and hard_zero_violations(report):
        return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async(_parser().parse_args())))


if __name__ == "__main__":
    main()
