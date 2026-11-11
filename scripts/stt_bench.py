"""So sánh model faster-whisper trên audio của chính bạn: độ trễ, tốc độ so với thời lượng audio, RAM, WER.

Chạy trong image backend (đã có faster-whisper khi BACKEND_EXTRAS=stt). Mỗi cấu hình chạy trong một tiến trình
con riêng để số RAM không lẫn nhau. Model tải về được giữ trong volume stt-models.

Thư mục mẫu:
    stt_samples/
        dat_lich_1.wav        # .wav .mp3 .m4a .webm .ogg .flac đều được
        dat_lich_1.txt        # (tùy chọn) câu đúng để tính WER; không có thì chỉ đo thời gian

Chạy (PowerShell, trong thư mục ai-patient-assistant):
    docker compose run --rm --no-deps -v ${PWD}/scripts:/bench:ro -v ${PWD}/stt_samples:/samples:ro `
        -v ${PWD}/stt_bench_result:/out backend python /bench/stt_bench.py /samples --out /out `
        --models small medium large-v3-turbo --beam 1 5 --language en

Lưu ý: WER so khớp từng từ, nên "6" và "six" bị tính là khác nhau — viết đáp án giống cách whisper viết số.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import resource
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}
LOW_CONFIDENCE = 0.6  # cùng ngưỡng PA_STT_CONFIDENCE_THRESHOLD mặc định


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB


# ---------------------------------------------------------------- tiến trình con: một cấu hình


def run_config(cfg: dict) -> dict:
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(cfg["model"], device="cpu", compute_type=cfg["compute_type"], cpu_threads=cfg["threads"])
    load_s = time.perf_counter() - t0
    files = [Path(f) for f in cfg["files"]]

    def once(path: Path) -> dict:
        start = time.perf_counter()
        segments, info = model.transcribe(
            str(path),
            language=None if cfg["language"] == "auto" else cfg["language"],
            beam_size=cfg["beam"],
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=cfg["prompt"] or None,
        )
        texts, probs = [], []
        for seg in segments:  # generator: giải mã thật sự diễn ra ở đây
            texts.append(seg.text.strip())
            probs += [w.probability for w in seg.words or []]
        elapsed = time.perf_counter() - start
        return {
            "file": path.name,
            "seconds": round(elapsed, 2),
            "audio_seconds": round(info.duration, 2),
            "rtf": round(elapsed / info.duration, 2) if info.duration else None,
            "language": info.language,
            "text": " ".join(texts).strip(),
            "low_conf_words": sum(p < LOW_CONFIDENCE for p in probs),
            "words": len(probs),
        }

    once(files[0])  # khởi động (warm-up), không tính
    rows = [once(f) for f in files]
    return {"load_seconds": round(load_s, 1), "peak_rss_mb": round(_rss_mb()), "rows": rows}


# ---------------------------------------------------------------- tiến trình chính


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark faster-whisper trên audio local")
    p.add_argument("samples", type=Path, help="Thư mục chứa audio (+ file .txt cùng tên làm đáp án)")
    p.add_argument("--models", nargs="+", default=["small", "medium", "large-v3-turbo"])
    p.add_argument("--language", default="en", help="en | vi | auto (để whisper tự nhận)")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam", type=int, nargs="+", default=[5], help="Ví dụ: --beam 1 5")
    p.add_argument("--threads", type=int, nargs="+", default=[0], help="0 = mặc định của CTranslate2")
    p.add_argument("--prompt", default="", help="initial_prompt (hệ thống đang dùng danh mục khoa/bác sĩ)")
    p.add_argument("--out", type=Path, default=Path("stt_bench_result"))
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._worker:
        print(
            json.dumps(
                run_config(json.loads(args._worker)), default=lambda o: o.item() if hasattr(o, "item") else str(o)
            )
        )
        return

    files = sorted(f for f in args.samples.iterdir() if f.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"Không thấy file audio trong {args.samples}")
    refs = {
        f.name: f.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        for f in files
        if f.with_suffix(".txt").exists()
    }
    args.out.mkdir(parents=True, exist_ok=True)

    summary, detail = [], []
    for model in args.models:
        for beam in args.beam:
            for threads in args.threads:
                cfg = {
                    "model": model,
                    "compute_type": args.compute_type,
                    "beam": beam,
                    "threads": threads,
                    "language": args.language,
                    "prompt": args.prompt,
                    "files": [str(f) for f in files],
                }
                label = f"{model} beam={beam} threads={threads or 'auto'}"
                print(f"▶ {label} ...", flush=True)
                proc = subprocess.run(
                    [sys.executable, __file__, str(args.samples), "--_worker", json.dumps(cfg)],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    print(f"  lỗi (có thể thiếu RAM — exit {proc.returncode}):\n{proc.stderr[-800:]}")
                    summary.append({"config": label, "error": f"exit {proc.returncode}"})
                    continue
                res = json.loads(proc.stdout.strip().splitlines()[-1])
                rows = res["rows"]
                for r in rows:
                    r["config"] = label
                    r["wer"] = round(wer(refs[r["file"]], r["text"]), 3) if r["file"] in refs else None
                    detail.append(r)
                wers = [r["wer"] for r in rows if r["wer"] is not None]
                summary.append(
                    {
                        "config": label,
                        "load_s": res["load_seconds"],
                        "peak_ram_mb": res["peak_rss_mb"],
                        "avg_s": round(statistics.mean(r["seconds"] for r in rows), 2),
                        "max_s": max(r["seconds"] for r in rows),
                        "avg_rtf": round(statistics.mean(r["rtf"] for r in rows if r["rtf"]), 2),
                        "avg_wer": round(statistics.mean(wers), 3) if wers else None,
                        "low_conf_%": round(
                            100 * sum(r["low_conf_words"] for r in rows) / max(1, sum(r["words"] for r in rows)), 1
                        ),
                    }
                )
                print(f"  xong: {summary[-1]}", flush=True)

    with (args.out / "detail.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "config",
                "file",
                "audio_seconds",
                "seconds",
                "rtf",
                "wer",
                "language",
                "low_conf_words",
                "words",
                "text",
            ],
        )
        w.writeheader()
        w.writerows(detail)
    cols = ["config", "load_s", "peak_ram_mb", "avg_s", "max_s", "avg_rtf", "avg_wer", "low_conf_%"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(s.get(c, s.get("error", ""))) for c in cols) + " |" for s in summary]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nChi tiết từng file: {args.out / 'detail.csv'}")


if __name__ == "__main__":
    main()
