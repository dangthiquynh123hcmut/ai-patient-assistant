"""Kiểm tra kết nối tới Claude trên Amazon Bedrock bằng IAM, từng bước, kèm chẩn đoán khi lỗi.

    python -m patient_assistant.interface.cli.check_llm
    python -m patient_assistant.interface.cli.check_llm --region ap-southeast-1 --profile default

Đọc cấu hình từ .env (PA_AWS_REGION, PA_AWS_PROFILE, PA_LLM_MODEL_ID). Mỗi lần chạy gọi Claude 3 lần
với đầu vào rất ngắn (chi phí không đáng kể).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

import anthropic

from patient_assistant.application.ports.llm import LLMError, ReportPrompt
from patient_assistant.config import Settings
from patient_assistant.domain.report.validators import ReportValidationContext, parse_report, validate_report
from patient_assistant.infrastructure.llm.bedrock import AnthropicBedrockLLM

OK, FAIL = "✅", "❌"

_HINTS = {
    "auth": "Credential AWS không hợp lệ hoặc đã hết hạn. Chạy `aws sts get-caller-identity`; nếu dùng SSO thì "
    "`aws sso login --profile <tên>`.",
    "permission": "IAM chưa cho phép gọi Claude: cần quyền `bedrock:InvokeModel` (ID dạng `global.…-v1`) hoặc "
    "`bedrock-mantle:CreateInference` (ID dạng `anthropic.claude-opus-5-5`) trên model này, và "
    "model phải được bật ở AWS Console > Amazon Bedrock > Model access (đúng region).",
    "not_found": "Không thấy model ở region này. Kiểm tra PA_LLM_MODEL_ID (dạng `global.anthropic.claude-opus-4-6-v1` "
    "hoặc `anthropic.claude-opus-5-5`) và "
    "PA_AWS_REGION; thử region khác như us-east-1 hoặc us-west-2.",
    "rate": "Bị giới hạn tốc độ / quota. Đợi một lúc rồi chạy lại, hoặc xin tăng quota trong AWS.",
    "connection": "Không kết nối được tới endpoint Bedrock của region (bedrock-runtime / bedrock-mantle). "
    "Kiểm tra mạng, proxy, firewall công ty.",
}


def _explain(exc: BaseException) -> str:
    cause = exc.__cause__ or exc
    if isinstance(cause, anthropic.AuthenticationError):
        key = "auth"
    elif isinstance(cause, anthropic.PermissionDeniedError):
        key = "permission"
    elif isinstance(cause, anthropic.NotFoundError):
        key = "not_found"
    elif isinstance(cause, anthropic.RateLimitError):
        key = "rate"
    elif isinstance(cause, anthropic.APIConnectionError):
        key = "connection"
    else:
        return f"{type(cause).__name__}: {cause}"
    return f"{type(cause).__name__}: {cause}\n   → {_HINTS[key]}"


def _identity(region: str | None, profile: str | None) -> tuple[bool, str]:
    try:
        import boto3

        session = boto3.Session(profile_name=profile, region_name=region)
        who = session.client("sts").get_caller_identity()
        return True, f"IAM: {who['Arn']} (account {who['Account']}), region={session.region_name}"
    except Exception as exc:  # noqa: BLE001 - chỉ để báo lỗi cho người dùng
        network = any(word in type(exc).__name__ for word in ("Connect", "Proxy", "Timeout", "Endpoint"))
        hint = _HINTS["connection"].replace("endpoint Bedrock", "endpoint STS") if network else _HINTS["auth"]
        return False, f"Không xác định được danh tính IAM: {type(exc).__name__}: {exc}\n   → {hint}"


async def run(region: str | None, profile: str | None, model: str) -> int:
    print(
        f"Model: {model} · region: {region or '(theo AWS_REGION / ~/.aws/config)'} · profile: {profile or 'mặc định'}"
    )
    ok, message = _identity(region, profile)
    print(f"{OK if ok else FAIL} 1. {message}")
    if not ok:
        return 1

    llm = AnthropicBedrockLLM.from_region(region, model, aws_profile=profile)
    try:
        text, tokens_in, tokens_out = await llm.ping()
        print(f"{OK} 2. Gọi Claude thành công: “{text}” ({tokens_in} in / {tokens_out} out tokens)")
    except LLMError as exc:
        print(f"{FAIL} 2. Gọi Claude thất bại — {_explain(exc)}")
        return 1

    sentence = "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối"
    now = datetime.now().astimezone()
    try:
        raw = await llm.extract_report(ReportPrompt(sources=[sentence], now=now))
    except LLMError as exc:
        print(f"{FAIL} 3. Structured output thất bại — {_explain(exc)}")
        return 1
    draft, schema_issues = parse_report(raw)
    issues = schema_issues or validate_report(draft, ReportValidationContext(sources=[sentence], now=now))  # type: ignore[arg-type]
    if draft is None or issues:
        print(f"{FAIL} 3. Claude trả về nhưng validator từ chối: {[i.code for i in issues]} — đầu ra: {raw}")
        return 1
    print(
        f"{OK} 3. Trích xuất báo cáo qua validator: đau={draft.pain_level}, tập={draft.exercise_days}, "
        f"triệu chứng={draft.symptom_change}"
    )

    try:
        intents = await llm.classify_intent("dạo này cái gối hơi nhức, muốn hẹn bác sĩ xem lại")
    except LLMError as exc:
        print(f"{FAIL} 4. Phân loại intent thất bại — {_explain(exc)}")
        return 1
    print(f"{OK} 4. Phân loại câu nói lỏng: {[i.value for i in intents]}")
    print("\nSẵn sàng: khởi động server.")
    return 0


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Kiểm tra kết nối Claude trên Amazon Bedrock (IAM)")
    parser.add_argument("--region", default=settings.aws_region)
    parser.add_argument("--profile", default=settings.aws_profile)
    parser.add_argument("--model", default=settings.llm_model_id)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.region, args.profile, args.model)))


if __name__ == "__main__":
    main()
