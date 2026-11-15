"""CLI chạy bộ đánh giá.

pa-eval                               # Claude trên Bedrock (tốn phí), toàn bộ dataset
pa-eval --suite booking               # chỉ suite booking
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
from patient_assistant.container import build_llm, require_llm
from patient_assistant.evaluation.cases import load_cases
from patient_assistant.evaluation.harness import run_cases
from patient_assistant.evaluation.metrics import compute
from patient_assistant.evaluation.report import hard_zero_violations, to_json, to_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pa-eval", description="Đánh giá AI Patient Assistant trên dataset JSONL")
    parser.add_argument("--dataset", type=Path, nargs="*", help="File JSONL (mặc định: dataset đóng gói kèm)")
    parser.add_argument("--suite", choices=["report", "booking", "conversation", "agent"], action="append")
    parser.add_argument(
        "--agent-mode",
        choices=["workflow", "booking", "full"],
        help="Chạy các case không tự khai báo agent_mode ở chế độ này. Mặc định workflow — KHÔNG theo PA_AGENT_MODE "
        "trong .env, vì các case đó viết cho luồng workflow",
    )
    parser.add_argument("--case", action="append", help="Chỉ chạy case có id này (lặp được)")
    parser.add_argument("--out", type=Path, default=Path("eval_reports"))
    parser.add_argument("--concurrency", type=int, default=4, help="Số case chạy song song")
    parser.add_argument("--fail-on-hard-zero", action="store_true")
    return parser


async def main_async(arguments: argparse.Namespace) -> int:
    cases = load_cases(arguments.dataset)
    if arguments.suite:
        cases = [case for case in cases if case.suite in arguments.suite]
    if arguments.case:
        cases = [case for case in cases if case.id in arguments.case]
    if not cases:
        print("Không có case nào khớp bộ lọc.", file=sys.stderr)
        return 2
    settings = Settings(agent_mode=arguments.agent_mode or "workflow")
    await require_llm(build_llm(settings), settings)
    observations = await run_cases(cases, lambda: build_llm(settings), settings, concurrency=arguments.concurrency)
    report = compute(cases, observations)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "run": stamp,
        "model": settings.llm_model_id,
        "agent_mode": settings.effective_agent_mode,
        "cases": len(cases),
    }
    output_directory = arguments.out / stamp
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "report.md").write_text(to_markdown(report, meta), "utf-8")
    (output_directory / "report.json").write_text(
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
    print(f"Báo cáo: {output_directory / 'report.md'}")
    if arguments.fail_on_hard_zero and hard_zero_violations(report):
        return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async(_parser().parse_args())))


if __name__ == "__main__":
    main()
