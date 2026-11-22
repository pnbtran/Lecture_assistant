"""
bench.py — so sánh nhanh nhiều cấu hình Whisper trên vài đoạn ngắn của MỘT bài giảng,
để chọn cấu hình chép được nhiều chữ đúng nhất trước khi chạy cả bài.

Ví dụ:
  .venv\\Scripts\\python.exe bench.py "D:\\research_assistant\\data\\lectures\\bai.m4a"
  .venv\\Scripts\\python.exe bench.py bai.m4a --clips 0,60 --len 10 --with-large

Kết quả: bảng trên màn hình + thư mục <LECTURE_TOOL_OUT>\\bench\\<tên bài>\\ (mỗi cấu hình 1 file .txt).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lecture_tool as lt  # noqa: E402

SR = 16000


def preemph(wav):
    """Lọc nhấn cao tần + chuẩn hoá mức: giúp tiếng 'ù/đục' (micro xa) rõ phụ âm hơn."""
    import numpy as np
    y = np.empty_like(wav)
    y[0] = wav[0]
    y[1:] = wav[1:] - 0.95 * wav[:-1]
    p99 = float(np.percentile(np.abs(y[::8]), 99)) or 1e-4
    return np.clip(y * min(30.0, 0.3 / p99), -1, 1).astype("float32")


def normalize(wav):
    return lt.normalize_audio(wav)[0]


LOWCONF = {"no_speech_threshold": None, "log_prob_threshold": None}

# tên -> (model, tiền xử lý, chạy theo lô?, tham số transcribe)
CONFIGS = {
    "1_hien_tai":        ("large-v3-turbo", None, False,
                          {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 700}}),
    "2_vad_thap":        ("large-v3-turbo", None, False,
                          {"vad_filter": True, "vad_parameters": {"threshold": 0.15, "min_silence_duration_ms": 1500,
                                                                  "speech_pad_ms": 600}, **LOWCONF}),
    "3_khong_vad":       ("large-v3-turbo", None, False,
                          {"vad_filter": False, "word_timestamps": True,
                           "hallucination_silence_threshold": 2.0, **LOWCONF}),
    "4_loc_cao_tan":     ("large-v3-turbo", preemph, False,
                          {"vad_filter": True, "vad_parameters": {"threshold": 0.25, "min_silence_duration_ms": 1500,
                                                                  "speech_pad_ms": 600}, **LOWCONF}),
    "5_theo_lo":         ("large-v3-turbo", None, True,
                          {"vad_filter": True, "vad_parameters": {"threshold": 0.25, "min_silence_duration_ms": 1500,
                                                                  "speech_pad_ms": 600}, **LOWCONF}),
    "6_khong_prompt":    ("large-v3-turbo", None, False,
                          {"vad_filter": True, "vad_parameters": {"threshold": 0.25, "min_silence_duration_ms": 1500,
                                                                  "speech_pad_ms": 600}, "initial_prompt": None, **LOWCONF}),
}
_BVAD = {"vad_filter": True, "vad_parameters": {"threshold": 0.25, "min_silence_duration_ms": 1500,
                                                "speech_pad_ms": 600}, **LOWCONF}
CONFIGS["9_theo_lo_loc"] = ("large-v3-turbo", preemph, True, dict(_BVAD))

LARGE = {
    "7_large_v3":        ("large-v3", None, False,
                          {"vad_filter": True, "vad_parameters": {"threshold": 0.25, "min_silence_duration_ms": 1500,
                                                                  "speech_pad_ms": 600}, **LOWCONF}),
    "8_large_v3_khong_vad": ("large-v3", None, False,
                          {"vad_filter": False, "word_timestamps": True,
                           "hallucination_silence_threshold": 2.0, **LOWCONF}),
    "10_large_theo_lo":  ("large-v3", None, True, dict(_BVAD)),
    "11_large_theo_lo_loc": ("large-v3", preemph, True, dict(_BVAD)),
    "12_large_lo_loc_prompt_ngan": ("large-v3", preemph, True,
                          {**_BVAD, "initial_prompt": "Лекция по строительной механике. Говорит преподаватель."}),
    "13_large_lo_loc_khong_prompt": ("large-v3", preemph, True, {**_BVAD, "initial_prompt": None}),
}


def load(name, device, model_dir):
    lt.setup_cuda_dlls()
    from faster_whisper import WhisperModel
    ct = "float16" if device == "cuda" else "int8"
    return WhisperModel(name, device=device, compute_type=ct, download_root=model_dir)


def run_one(model, audio, batched, kw):
    base = {"language": "ru", "beam_size": 5, "condition_on_previous_text": False,
            "initial_prompt": lt.DEFAULT_PROMPT}
    base.update(kw)
    t0 = time.time()
    if batched:
        from faster_whisper import BatchedInferencePipeline
        base.pop("word_timestamps", None)
        base.pop("hallucination_silence_threshold", None)
        base.pop("condition_on_previous_text", None)
        segs, _ = BatchedInferencePipeline(model=model).transcribe(audio, batch_size=8, **base)
    else:
        segs, _ = model.transcribe(audio, **base)
    out = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()} for s in segs]
    out = lt.clean_segments(out)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="So sánh cấu hình Whisper trên vài đoạn của một bài giảng")
    ap.add_argument("audio")
    ap.add_argument("--clips", default="0,60", help="phút bắt đầu các đoạn thử, cách nhau dấu phẩy (mặc định 0,60)")
    ap.add_argument("--len", type=float, default=10, help="độ dài mỗi đoạn (phút), mặc định 10")
    ap.add_argument("--with-large", action="store_true", help="thử thêm model large-v3 (tải thêm ~3 GB lần đầu)")
    ap.add_argument("--only", default="", help="chỉ chạy các cấu hình có tên chứa chuỗi này, vd 3_,5_")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model-dir", default=os.environ.get("LECTURE_TOOL_MODEL_DIR"))
    ap.add_argument("--out", default=os.environ.get("LECTURE_TOOL_OUT", "lecture_out"))
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Không tìm thấy file: {audio_path}")
    outdir = Path(args.out) / "bench" / audio_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        try:
            lt.setup_cuda_dlls()
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    from faster_whisper import decode_audio
    print(f"Đọc audio: {audio_path.name}")
    wav = decode_audio(str(audio_path), sampling_rate=SR)
    clips = []
    for m in [float(x) for x in args.clips.split(",") if x.strip()]:
        a, b = int(m * 60 * SR), int((m + args.len) * 60 * SR)
        if a < len(wav):
            clips.append((f"phut{int(m):02d}", wav[a:min(b, len(wav))]))

    cfgs = dict(CONFIGS)
    if args.with_large:
        cfgs.update(LARGE)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        cfgs = {n: c for n, c in cfgs.items() if any(k in n for k in keys)}

    results = []
    model, model_name = None, None
    for name, (mname, pre, batched, kw) in cfgs.items():
        if mname != model_name:
            model = None
            gc.collect()
            print(f"\nNạp model {mname} ({device})...")
            model, model_name = load(mname, device, args.model_dir), mname
        for cname, clip in clips:
            audio = pre(clip) if pre else clip
            try:
                segs, dt = run_one(model, audio, batched, kw)
            except Exception as e:
                print(f"  {name:22s} {cname}: LỖI {e}")
                results.append({"cfg": name, "clip": cname, "error": str(e)})
                continue
            chars = sum(len(s["text"]) for s in segs)
            mins = len(clip) / SR / 60
            r = {"cfg": name, "clip": cname, "chars_per_min": round(chars / mins), "segments": len(segs),
                 "seconds": round(dt, 1)}
            results.append(r)
            print(f"  {name:22s} {cname}: {r['chars_per_min']:5d} ký tự/phút  {len(segs):4d} đoạn  {dt:6.1f} s")
            (outdir / f"{cname}__{name}.txt").write_text(
                "\n".join(f"[{lt.fmt_ts(s['start'])}] {s['text']}" for s in segs) + "\n", "utf-8")

    (outdir / "bench_result.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    print("\nBẢNG TỔNG HỢP (ký tự/phút; cao hơn = chép được nhiều hơn, nhưng cần đọc lại để chắc là chữ đúng):")
    names = list(dict.fromkeys(r["cfg"] for r in results))
    cl = [c for c, _ in clips]
    print(f"{'cấu hình':24s}" + "".join(f"{c:>10s}" for c in cl))
    for n in names:
        row = {r["clip"]: r.get("chars_per_min", "lỗi") for r in results if r["cfg"] == n}
        print(f"{n:24s}" + "".join(f"{str(row.get(c, '-')):>10s}" for c in cl))
    print(f"\nĐã lưu các bản chép thử vào: {outdir}")


if __name__ == "__main__":
    main()
