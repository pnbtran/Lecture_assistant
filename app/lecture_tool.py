#!/usr/bin/env python3
"""
lecture_tool.py — Audio bài giảng tiếng Nga  ->  văn bản + tóm tắt + từ khóa.

Quy trình (mỗi bước lưu file riêng, chạy lại sẽ bỏ qua bước đã xong):
  1. Phiên âm  : faster-whisper (chạy offline trên máy bạn), có VAD, ghi dần ra đĩa
                 -> nếu bị ngắt giữa chừng thì chạy lại sẽ làm tiếp từ chỗ dở.
  2. Từ khóa   : pymorphy3 (đưa về nguyên dạng) + wordfreq (loại từ thông dụng), chạy offline.
  3. Tóm tắt   : map-reduce qua LLM (Claude / Gemini / Ollama) -> outline theo mốc thời gian,
                 thuật ngữ + giải thích, thông báo của giảng viên, câu hỏi ôn tập.
  4. Báo cáo   : <tên>.report.md

Ví dụ:
  python lecture_tool.py bai1.mp3 --subject "Строительная механика" --lang vi
  python lecture_tool.py bai1.mp3 --llm none            # chỉ phiên âm + từ khóa
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

for _st in (sys.stdout, sys.stderr):          # tránh lỗi mã hóa (cp1251/cp1258) khi --help hoặc chuyển hướng ra file
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from functools import lru_cache
from pathlib import Path
from string import Template

# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #

def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, file=sys.stderr, flush=True)


def fmt_ts(sec: float, ms: bool = False) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    if ms:
        base += f",{int((sec - int(sec)) * 1000):03d}"
    return base


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # dòng cuối bị cắt dở khi ngắt đột ngột
    return out


# --------------------------------------------------------------------------- #
# 1. Phiên âm
# --------------------------------------------------------------------------- #

DEFAULT_PROMPT = (
    "Лекция по строительной механике и расчёту строительных конструкций. "
    "Сопротивление материалов, метод конечных элементов, железобетон, балка, рама, "
    "арка, жёсткость, изгибающий момент, поперечная сила, продольная сила, "
    "статически определимая система, степень свободы, собственная частота, ЛИРА, SCAD."
)

HALLUCINATION_RE = re.compile(
    r"(субтитры\s+(сделал|подогнал|создавал|делал)|dimatorzok|редактор субтитров|"
    r"корректор\s+[а-я]\.|продолжение следует\.{0,3}$)",
    re.IGNORECASE,
)

_MODEL_CACHE: dict = {}


def setup_cuda_dlls() -> None:
    """Windows: faster-whisper cần cuBLAS + cuDNN. Nếu đã `pip install nvidia-cublas-cu12
    nvidia-cudnn-cu12` thì thêm thư mục bin của chúng vào PATH để CTranslate2 tìm thấy."""
    if sys.platform != "win32":
        return
    import site
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for r in roots:
        nv = Path(r) / "nvidia"
        if not nv.is_dir():
            continue
        for b in nv.glob("*/bin"):
            if b.is_dir() and str(b) not in os.environ.get("PATH", ""):
                os.add_dll_directory(str(b))
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


def check_gpu(args) -> None:
    """python lecture_tool.py --check-gpu : thử chạy Whisper nhỏ trên GPU để biết môi trường đã ổn chưa."""
    setup_cuda_dlls()
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    log(f"Số GPU CUDA mà CTranslate2 thấy: {n}")
    if n == 0:
        log("-> Không dùng được GPU. Cập nhật driver NVIDIA (RTX 50 cần driver dòng 570 trở lên) "
            "và kiểm tra đã cài nvidia-cublas-cu12 / nvidia-cudnn-cu12. Tạm thời có thể chạy --device cpu.")
        return
    log(f"Kiểu tính toán GPU hỗ trợ: {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        m = WhisperModel("tiny", device="cuda", compute_type="float16", download_root=args.model_dir)
        segs, _ = m.transcribe(np.zeros(16000 * 5, dtype=np.float32), language="ru")
        list(segs)
        log("OK: Whisper chạy được trên GPU (float16).")
    except Exception as e:
        log(f"LỖI: {type(e).__name__}: {e}")
        log("Gợi ý: (1) cập nhật driver NVIDIA; (2) pip install -U \"nvidia-cublas-cu12>=12.8\" "
            "\"nvidia-cudnn-cu12>=9.11\"; (3) giữ compute type = float16 (đừng dùng int8_float16 trên RTX 50); "
            "(4) nếu vẫn lỗi, chạy --device cpu.")


def pick_device(args) -> tuple[str, str]:
    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = args.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def get_model(args):
    device, compute = pick_device(args)
    key = (args.model, device, compute)
    if key not in _MODEL_CACHE:
        setup_cuda_dlls()
        from faster_whisper import WhisperModel
        log(f"[1/4] Nạp model Whisper '{args.model}' ({device}, {compute}) "
            f"— lần đầu sẽ tải về, có thể vài GB...")
        _MODEL_CACHE[key] = WhisperModel(
            args.model, device=device, compute_type=compute,
            cpu_threads=args.threads, download_root=args.model_dir,
        )
    return _MODEL_CACHE[key], device


def _leak_re(prompt: str | None):
    """Regex bắt chuỗi >=3 mục liên tiếp chép lại từ câu gợi ý (Whisper hay 'đọc lại' prompt)."""
    if not prompt:
        return None
    items = [x.strip() for x in re.split(r"[.,;]", prompt) if len(x.strip()) >= 3]
    if len(items) < 3:
        return None
    alt = "|".join(re.escape(x) for x in sorted(items, key=len, reverse=True))
    return re.compile(rf"(?:(?:{alt})[\s.,;]*){{3,}}", re.IGNORECASE)


def clean_segments(segs: list[dict], prompt: str | None = None) -> list[dict]:
    """Bỏ câu ảo giác quen thuộc của Whisper, đoạn lặp liên tiếp và phần chép lại câu gợi ý."""
    leak = _leak_re(prompt)
    out: list[dict] = []
    repeat = 0
    for s in segs:
        text = s["text"].strip()
        if leak is not None:
            text = leak.sub(" ", text).strip()
            text = re.sub(r"\s{2,}", " ", text)
        if not text or HALLUCINATION_RE.search(text):
            continue
        if out and text.lower() == out[-1]["text"].lower():
            repeat += 1
            if repeat >= 2:      # cho phép lặp 1 lần, từ lần thứ 3 trở đi coi là vòng lặp ảo giác
                continue
        else:
            repeat = 0
        out.append({"start": float(s["start"]), "end": float(s["end"]), "text": text})
    return out


def normalize_audio(wav):
    """Khuếch đại âm thanh quá nhỏ (ghi âm lớp học xa micro) lên mức hợp lý, không làm méo."""
    import numpy as np
    if wav.size == 0:
        return wav, 1.0
    p99 = float(np.percentile(np.abs(wav[::8]), 99))
    if p99 <= 1e-4 or p99 >= 0.5:
        return wav, 1.0
    gain = min(10.0, 0.5 / p99)
    return np.clip(wav * gain, -1.0, 1.0).astype("float32"), gain


def preemphasis(wav):
    """Nhấn cao tần + chuẩn hoá mức: làm rõ phụ âm khi tiếng bị 'đục' (ghi từ chỗ ngồi, xa bảng)."""
    import numpy as np
    if wav.size < 2:
        return wav
    y = np.empty_like(wav)
    y[0] = wav[0]
    y[1:] = wav[1:] - 0.95 * wav[:-1]
    p99 = float(np.percentile(np.abs(y[::8]), 99)) or 1e-4
    return np.clip(y * min(30.0, 0.3 / p99), -1, 1).astype("float32")


def build_prompt(args) -> str | None:
    """Mặc định KHÔNG dùng câu gợi ý: thử nghiệm cho thấy Whisper hay chép lại nó vào bản phiên âm."""
    prompt = args.prompt if args.prompt else None
    if prompt == "default":
        prompt = DEFAULT_PROMPT
    if prompt and args.terms:
        prompt = (prompt + " " + ", ".join(read_terms(Path(args.terms))[:40])).strip()
    return prompt


def transcribe(audio: Path, outdir: Path, args) -> tuple[list[dict], dict]:
    stem = audio.stem
    seg_path = outdir / f"{stem}.segments.jsonl"
    meta_path = outdir / f"{stem}.meta.json"

    if args.force:
        seg_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    if meta_path.exists():
        log("[1/4] Đã có bản phiên âm — bỏ qua (dùng --force để làm lại).")
        return clean_segments(read_jsonl(seg_path), build_prompt(args)), json.loads(meta_path.read_text("utf-8"))

    from faster_whisper import decode_audio

    model, device = get_model(args)
    log(f"      Đọc audio: {audio.name}")
    wav = decode_audio(str(audio), sampling_rate=16000)
    total = len(wav) / 16000.0
    import numpy as np
    p99 = float(np.percentile(np.abs(wav[::8]), 99)) if len(wav) else 0.0
    log(f"      Mức âm lượng (p99): {p99:.3f}")
    if getattr(args, "normalize", False):
        wav, gain = normalize_audio(wav)
        log(f"      Đã khuếch đại x{gain:.1f} (--normalize).")
    if not getattr(args, "no_preemph", False):
        wav = preemphasis(wav)
        log("      Đã lọc nhấn cao tần (tắt bằng --no-preemph).")

    done = read_jsonl(seg_path)
    offset = done[-1]["end"] if done else 0.0
    if offset > 0:
        log(f"      Làm tiếp từ {fmt_ts(offset)} (đã có {len(done)} đoạn).")
    sub = wav[int(offset * 16000):]

    beam = args.beam_size or (5 if device == "cuda" else 2)
    prompt = build_prompt(args)

    kw = dict(
        language="ru",
        beam_size=beam,
        condition_on_previous_text=False,   # tránh vòng lặp ảo giác trên bài giảng dài
        initial_prompt=prompt or None,
    )
    if getattr(args, "no_vad", False):
        kw["vad_filter"] = False
    else:
        kw["vad_filter"] = True
        kw["vad_parameters"] = {"threshold": getattr(args, "vad_threshold", 0.25),
                                "min_silence_duration_ms": 1500, "speech_pad_ms": 600}
    if not getattr(args, "strict_conf", False):    # giữ cả đoạn "kém tự tin" (hay mất chữ ở lớp học ồn/xa mic)
        kw["no_speech_threshold"] = None
        kw["log_prob_threshold"] = None
    bs = getattr(args, "batch_size", 0)
    if bs and not getattr(args, "no_vad", False):
        # Chạy theo lô: chia audio thành các đoạn lời nói <=30 giây và phiên âm độc lập từng đoạn.
        # Cách này ít bị "nhảy cóc" bỏ qua cả phút lời nói như chế độ tuần tự trên bài giảng dài.
        from faster_whisper import BatchedInferencePipeline
        pipe = BatchedInferencePipeline(model=model)
        kw["batch_size"] = bs
        kw.pop("condition_on_previous_text", None)
        log(f"      Chế độ theo lô (batch_size={bs}).")
        iterator, _info = pipe.transcribe(sub, **kw)
    else:
        iterator, _info = model.transcribe(sub, **kw)

    t0 = time.time()
    with open(seg_path, "a", encoding="utf-8") as f:
        for s in iterator:
            rec = {"start": round(s.start + offset, 2), "end": round(s.end + offset, 2),
                   "text": s.text.strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = max(time.time() - t0, 1e-6)
            speed = (rec["end"] - offset) / elapsed
            log(f"\r      {100 * rec['end'] / max(total, 1):5.1f}%  "
                f"{fmt_ts(rec['end'])} / {fmt_ts(total)}   tốc độ x{speed:.2f}   ", end="")
    log("")
    all_segs = read_jsonl(seg_path)
    nchars = sum(len(x["text"]) for x in all_segs)
    per_min = nchars / max(total / 60.0, 1e-6)
    log(f"      Mật độ chữ: {per_min:.0f} ký tự/phút ({nchars} ký tự, {len(all_segs)} đoạn).")
    if per_min < 250:
        log("      ! Bản phiên âm khá THƯA. Nếu nghe lại thấy thầy nói liên tục, kiểm tra cách ghi âm "
            "(điện thoại hướng về bảng, xa laptop) hoặc thử --vad-threshold 0.15.")

    meta = {
        "audio": audio.name, "duration": round(total, 1), "model": args.model,
        "device": device, "language": "ru",
        "transcribed_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return clean_segments(read_jsonl(seg_path), prompt), meta


def make_paragraphs(segs: list[dict], gap: float = 1.8, max_chars: int = 700) -> list[dict]:
    paras: list[dict] = []
    cur: dict | None = None
    for s in segs:
        if cur is not None:
            pause = s["start"] - cur["end"]
            ends_sentence = cur["text"].rstrip().endswith((".", "?", "!", "…"))
            if pause > gap or (len(cur["text"]) > max_chars and ends_sentence):
                paras.append(cur)
                cur = None
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
        else:
            cur["text"] += " " + s["text"]
            cur["end"] = s["end"]
    if cur is not None:
        paras.append(cur)
    return paras


def write_transcripts(segs, paras, outdir: Path, stem: str) -> None:
    lines = [f"[{fmt_ts(p['start'])}] {p['text']}" for p in paras]
    (outdir / f"{stem}.transcript.txt").write_text("\n\n".join(lines) + "\n", "utf-8")
    srt = []
    for i, s in enumerate(segs, 1):
        srt.append(f"{i}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{s['text']}\n")
    (outdir / f"{stem}.srt").write_text("\n".join(srt), "utf-8")
    (outdir / f"{stem}.transcript.json").write_text(
        json.dumps(segs, ensure_ascii=False, indent=1), "utf-8")


# --------------------------------------------------------------------------- #
# 2. Từ khóa (offline)
# --------------------------------------------------------------------------- #

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
CHUNK_SPLIT = re.compile(r"[.,;:!?()\[\]«»\"…—–]+|\s-\s")
PHRASE_RE = re.compile(r"A{0,2}N(?:A?N){0,2}")

STOP_LEMMAS = {
    "раз", "вопрос", "случай", "вещь", "часть", "образ", "пример", "способ", "смысл",
    "сторона", "год", "день", "человек", "жизнь", "слово", "дело", "место", "вид", "тип",
    "число", "идея", "ответ", "урок", "лекция", "занятие", "минута", "секунда", "сегодня",
    "этот", "такой", "данный", "следующий", "другой", "новый", "большой", "маленький",
    "целый", "некоторый", "каждый", "любой", "первый", "второй", "третий", "тот", "весь",
    "свой", "наш", "ваш", "который", "какой", "самый", "нужный", "разный", "различный",
    "определённый", "известный", "правильный", "общий", "последний", "предыдущий",
}


@dataclass(frozen=True)
class Tok:
    surface: str
    lemma: str
    code: str            # 'N' danh từ / 'A' tính từ-phân từ
    case: str | None
    latin: bool
    spec: float          # độ "chuyên ngành" 0..1 (từ càng hiếm trong tiếng Nga phổ thông càng cao)


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=1)
def _zipf():
    try:
        from wordfreq import zipf_frequency
        return lambda w: zipf_frequency(w, "ru")
    except Exception:
        return None


def _spec(lemma: str) -> tuple[float, float]:
    z = _zipf()
    if z is None:
        return 0.6, 3.0
    zf = z(lemma)
    return min(1.0, max(0.08, (7.0 - zf) / 4.2)), zf


_VOTES: Counter = Counter()


def _build_votes(segs: list[dict]) -> None:
    """Từ điển có lemma trùng (vd. «эпюра» / «эпюр»). Mỗi dạng từ trong văn bản bỏ 1 phiếu cho
    mọi lemma khả dĩ; lemma đúng thường nhận được phiếu từ tất cả các dạng -> thắng."""
    _VOTES.clear()
    wc = Counter(w.lower() for s in segs for w in WORD_RE.findall(s["text"])
                 if not re.match(r"[A-Za-z]", w))
    for w, c in wc.items():
        parses = _morph().parse(w)
        for lem in {q.normal_form for q in parses if q.tag.POS == parses[0].tag.POS}:
            _VOTES[lem] += c * (2 if lem == w else 1)   # dạng gốc xuất hiện nguyên văn -> cộng thêm
    analyze.cache_clear()


def _best_parse(lw: str):
    parses = _morph().parse(lw)
    top = parses[0]
    same_pos = [q for q in parses if q.tag.POS == top.tag.POS]
    if len(same_pos) > 1:
        z = _zipf()
        return max(same_pos, key=lambda q: (_VOTES.get(q.normal_form, 0),
                                           z(q.normal_form) if z else 0.0, q.score))
    return top


@lru_cache(maxsize=None)
def analyze(word: str) -> Tok | None:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", word):
        if len(word) < 3:
            return None
        return Tok(word, word.upper() if word.isupper() else word.lower(), "N", None, True, 1.0)
    if word.isupper() and 3 <= len(word) <= 6:           # аббревиатура: МКЭ, ЖБК, СНиП...
        return Tok(word, word, "N", None, True, 1.0)
    lw = word.lower()
    p = _best_parse(lw)
    pos = p.tag.POS
    if pos == "NOUN":
        code = "N"
    elif pos in ("ADJF", "PRTF"):
        code = "A"
    else:
        return None
    if any(g in p.tag for g in ("Name", "Surn", "Patr", "Geox")):
        return None
    lemma = p.normal_form
    if lemma in STOP_LEMMAS or len(lemma) < 3:
        return None
    spec, zf = _spec(lemma)
    if code == "A" and zf > 5.6:
        return None
    return Tok(lw, lemma, code, p.tag.case, False, spec)


def _inflect_adj(lemma: str, gender: str | None) -> str:
    try:
        f = _morph().parse(lemma)[0].inflect({"nomn", gender or "masc"})
        return f.word if f else lemma
    except Exception:
        return lemma


def _display(surfaces: tuple[str, ...]) -> str:
    toks = [analyze(w) for w in surfaces]
    if len(toks) == 1:
        return toks[0].lemma
    codes = "".join(t.code for t in toks)
    k = codes.index("N")
    head = toks[k]
    gender = None if head.latin else _morph().parse(head.lemma)[0].tag.gender
    out = []
    for i, t in enumerate(toks):
        if i < k:
            out.append(_inflect_adj(t.lemma, gender))
        elif i == k:
            out.append(head.lemma)
        else:
            out.append(t.surface if t.latin else t.surface.lower())
    return " ".join(out)


def read_terms(path: Path) -> list[str]:
    terms = []
    for line in path.read_text("utf-8").splitlines():
        line = line.split("#")[0].split("\t")[0].strip()
        if line:
            terms.append(line)
    return terms


def extract_keywords(segs: list[dict], top: int = 30) -> list[dict]:
    _build_votes(segs)
    counts: Counter = Counter()
    first: dict = {}
    surf: dict = defaultdict(Counter)

    for s in segs:
        for chunk in CHUNK_SPLIT.split(s["text"]):
            toks = [analyze(w) for w in WORD_RE.findall(chunk)]
            L = len(toks)
            for i in range(L):
                if toks[i] is None:
                    continue
                for n in range(1, 7):
                    if i + n > L or toks[i + n - 1] is None:
                        break
                    seq = toks[i:i + n]
                    codes = "".join(t.code for t in seq)
                    if n == 1:
                        if codes != "N":
                            continue
                    else:
                        if not PHRASE_RE.fullmatch(codes):
                            continue
                        nouns = [k for k, c in enumerate(codes) if c == "N"]
                        ok = True
                        for k2 in nouns[1:]:      # danh từ sau danh từ đầu phải ở sở hữu cách
                            if not (seq[k2].case == "gent" or seq[k2].latin):
                                ok = False
                            if codes[k2 - 1] == "A" and seq[k2 - 1].case != "gent":
                                ok = False
                        if not ok:
                            continue
                        if seq[0].latin and seq[0].lemma.islower():
                            continue
                    key = tuple(t.lemma for t in seq)
                    counts[key] += 1
                    first.setdefault(key, s["start"])
                    surf[key][tuple(t.surface if t.latin else t.surface.lower() for t in seq)] += 1

    def min_count(key) -> int:
        if len(key) == 1:
            return 2 if (key[0].isupper() or re.match(r"[A-Za-z]", key[0])) else 3
        return 2

    cands = {k: c for k, c in counts.items() if c >= min_count(k)}

    # Bỏ cụm ngắn nếu hầu như luôn nằm trong một cụm dài hơn (vd. "элемент" trong "конечный элемент")
    sub_max: dict = defaultdict(int)
    for key, c in cands.items():
        n = len(key)
        for a in range(n):
            for b in range(a + 1, n + 1):
                if b - a < n:
                    sub = key[a:b]
                    if c > sub_max[sub]:
                        sub_max[sub] = c

    scored = []
    for key, c in cands.items():
        if sub_max.get(key, 0) >= 0.6 * c:
            continue
        spec = sum(analyze(w).spec for w in surf[key].most_common(1)[0][0]) / len(key)
        score = c * spec * (1 + 0.35 * (len(key) - 1))
        scored.append((score, key, c))
    scored.sort(reverse=True)

    out = []
    for score, key, c in scored[:top]:
        best = surf[key].most_common(1)[0][0]
        out.append({"term": _display(best), "count": c, "first": first[key]})
    return out


def count_glossary(segs: list[dict], terms: list[str]) -> list[dict]:
    """Đếm số lần các thuật ngữ trong danh sách riêng của bạn xuất hiện (so khớp theo nguyên dạng)."""
    def lemmas(text: str) -> list[str]:
        res = []
        for w in WORD_RE.findall(text):
            if re.match(r"[A-Za-z]", w):
                res.append(w.lower())
            else:
                res.append(_best_parse(w.lower()).normal_form)
        return res

    seg_lem = [(s["start"], " " + " ".join(lemmas(s["text"])) + " ") for s in segs]
    rows = []
    for t in terms:
        needle = " " + " ".join(lemmas(t)) + " "
        cnt, first = 0, None
        for start, text in seg_lem:
            c = text.count(needle)
            if c:
                cnt += c
                first = start if first is None else first
        rows.append({"term": t, "count": cnt, "first": first})
    return rows


# --------------------------------------------------------------------------- #
# 3. Tóm tắt bằng LLM (map-reduce)
# --------------------------------------------------------------------------- #

class LLM:
    def __init__(self, kind: str, model: str | None):
        self.kind = kind
        defaults = {"claude": "claude-sonnet-5-5", "gemini": "gemini-2.5-flash",
                    "ollama": "qwen3.5:9b"}
        self.model = model or defaults[kind]

    def __call__(self, prompt: str, max_tokens: int = 4096) -> str:
        last = None
        for wait in (0, 5, 20):
            if wait:
                time.sleep(wait)
            try:
                return self._call(prompt, max_tokens)
            except Exception as e:          # mạng / quota / rate limit
                last = e
                log(f"      (lỗi gọi LLM: {type(e).__name__}: {str(e)[:120]} — thử lại)")
        raise RuntimeError(f"Gọi LLM thất bại: {last}")

    def _call(self, prompt: str, max_tokens: int) -> str:
        if self.kind == "claude":
            import anthropic
            client = anthropic.Anthropic()          # đọc ANTHROPIC_API_KEY
            msg = client.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        import requests
        if self.kind == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model}:generateContent")
            r = requests.post(
                url, headers={"x-goog-api-key": key or ""}, timeout=300,
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens,
                                           "responseMimeType": "application/json"}})
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        if self.kind == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not host.startswith("http"):
                host = "http://" + host
            r = requests.post(
                f"{host}/api/generate", timeout=1800,
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json",
                      "think": False,   # tắt "suy nghĩ" của Qwen3.x: nhanh hơn nhiều, JSON gọn hơn
                      "options": {"num_ctx": int(os.environ.get("LECTURE_TOOL_NUM_CTX", "8192")),
                                  "temperature": 0.2}})
            r.raise_for_status()
            return r.json()["response"]
        raise ValueError(self.kind)


def parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


LANG_NAME = {"vi": "Vietnamese", "ru": "Russian"}
LANG_RULE = {
    "vi": ("OUTPUT LANGUAGE: VIETNAMESE. Every title, bullet, overview sentence and explanation MUST be "
           "written in Vietnamese, even though the transcript is Russian. Russian (Cyrillic) is allowed ONLY "
           "for key technical terms, followed by the Vietnamese equivalent in parentheses at first use, "
           "e.g. «жёсткость (độ cứng)». In \"terms\": \"term\" = the Russian term only (no parentheses), "
           "\"explain\" = a Vietnamese explanation."),
    "ru": "Write in clear, simple Russian (B1-B2 level), short sentences.",
}

MAP_PROMPT = Template("""\
You are an assistant for a Vietnamese master's student of civil engineering studying at a Russian university.
Below is an automatic speech-recognition transcript (Whisper) of part $START–$END of a lecture in Russian.
Subject: $SUBJECT.
The transcript may contain recognition errors, especially in technical terms, names and formulas: silently
correct the obvious ones, but never invent content that is not in the text.
$LANG_RULE

Return ONLY valid JSON, no commentary, with exactly these keys:
{
  "title": "short heading for this part",
  "points": ["3-7 concise bullets: definitions, formulas, conditions, numbers, step-by-step logic"],
  "terms": [{"term": "Russian term as used by the lecturer", "explain": "one-sentence explanation"}],
  "admin": "announcements to the WHOLE group about FUTURE tests, deadlines, homework, grading or software, each starting with the [hh:mm:ss] marker where it is said; empty string if none"
}
"admin" must NOT include: a test or task that the students are doing during this very class, hints for
solving it, or private conversations with one student. If unsure, leave "admin" empty.
The transcript lines start with [hh:mm:ss] time markers.
"terms": at most 5 SPECIALISED technical terms of this part (skip everyday words like работа, сила, интеграл).

TRANSCRIPT:
$TEXT

FINAL REMINDER: $LANG_RULE
""")

REDUCE_PROMPT = Template("""\
You are an assistant for a Vietnamese master's student of civil engineering studying at a Russian university.
Below are structured notes of consecutive parts of ONE lecture in Russian. Subject: $SUBJECT.
$LANG_RULE

Return ONLY valid JSON with exactly these keys:
{
  "overview": "one paragraph (5-8 sentences): what the lecture is about and how it develops",
  "takeaways": ["5-10 most important things to remember"],
  "review_questions": ["5-8 self-check questions in Russian, as an examiner would ask"]
}
For Vietnamese output, add a short Vietnamese translation in parentheses after each review question.

NOTES:
$NOTES
""")


def _cyr_ratio(d: dict) -> float:
    """Tỉ lệ chữ Kirin trong tiêu đề + các ý (để phát hiện model 'quên' viết tiếng Việt)."""
    txt = " ".join([str(d.get("title", ""))] + [str(x) for x in d.get("points", [])])
    txt = re.sub(r"«[^»]*»|\([^)]*\)|\*\*[^*]*\*\*", " ", txt)   # bỏ thuật ngữ được trích
    cyr = len(re.findall(r"[А-Яа-яЁё]", txt))
    lat = len(re.findall(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]", txt))
    return cyr / max(cyr + lat, 1)


def _term_key(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", name)


def make_chunks(paras: list[dict], max_sec: float, max_chars: int) -> list[dict]:
    chunks, cur = [], None
    for p in paras:
        if cur and (p["end"] - cur["start"] > max_sec or len(cur["text"]) + len(p["text"]) > max_chars):
            chunks.append(cur)
            cur = None
        line = f"[{fmt_ts(p['start'])}] {p['text']}"
        if cur is None:
            cur = {"start": p["start"], "end": p["end"], "text": line}
        else:
            cur["text"] += "\n" + line
            cur["end"] = p["end"]
    if cur:
        chunks.append(cur)
    return chunks


def summarize(paras: list[dict], llm: LLM, args, cache: Path) -> dict:
    if cache.exists() and not args.force:
        data = json.loads(cache.read_text("utf-8"))
        if data.get("overview") is not None and data.get("lang") == args.lang:
            log("[3/4] Đã có bản tóm tắt — bỏ qua (dùng --force để làm lại).")
            return data
    else:
        data = {}

    subject = args.subject or "a university lecture on civil engineering / structural mechanics"
    chunks = make_chunks(paras, args.chunk_min * 60, args.chunk_chars)
    done = data.get("sections", []) if data.get("lang") == args.lang else []
    sections = []
    log(f"[3/4] Tóm tắt bằng {llm.kind} ({llm.model}) — {len(chunks)} phần.")
    for i, ch in enumerate(chunks):
        if i < len(done) and abs(done[i]["start"] - ch["start"]) < 1:
            sections.append(done[i])
            continue
        log(f"      phần {i + 1}/{len(chunks)}  [{fmt_ts(ch['start'])}–{fmt_ts(ch['end'])}]")
        t0 = time.time()
        prompt = MAP_PROMPT.safe_substitute(
            START=fmt_ts(ch["start"]), END=fmt_ts(ch["end"]), SUBJECT=subject,
            LANG_RULE=LANG_RULE[args.lang], TEXT=ch["text"])
        raw = llm(prompt)
        d = parse_json(raw)
        if args.lang == "vi" and isinstance(d, dict) and _cyr_ratio(d) > 0.5:
            log("      (bản tóm tắt phần này lại viết bằng tiếng Nga — yêu cầu viết lại bằng tiếng Việt)")
            raw2 = llm(prompt + "\n\nREMINDER: your previous answer was in Russian. Answer again, "
                       "in VIETNAMESE (Russian only for the technical terms).")
            d2 = parse_json(raw2)
            if isinstance(d2, dict) and _cyr_ratio(d2) < _cyr_ratio(d):
                d = d2
        log(f"        ({time.time() - t0:.0f} s)")
        if not isinstance(d, dict):
            d = {"title": f"Phần {i + 1}", "points": [raw.strip()[:1500]], "terms": [], "admin": ""}
        d.update(start=ch["start"], end=ch["end"])
        sections.append(d)
        cache.write_text(json.dumps({"lang": args.lang, "sections": sections},
                                    ensure_ascii=False, indent=1), "utf-8")

    notes = "\n\n".join(
        f"## {s.get('title', '')} [{fmt_ts(s['start'])}–{fmt_ts(s['end'])}]\n" +
        "\n".join(f"- {p}" for p in s.get("points", []))
        for s in sections)
    log("      tổng hợp toàn bài...")
    raw = llm(REDUCE_PROMPT.safe_substitute(
        SUBJECT=subject, LANG_RULE=LANG_RULE[args.lang], NOTES=notes))
    r = parse_json(raw)
    if not isinstance(r, dict):
        r = {"overview": raw.strip(), "takeaways": [], "review_questions": []}
    result = {"lang": args.lang, "sections": sections,
              "overview": r.get("overview", ""), "takeaways": r.get("takeaways", []),
              "review_questions": r.get("review_questions", [])}
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    return result


# --------------------------------------------------------------------------- #
# 4. Báo cáo Markdown
# --------------------------------------------------------------------------- #

H = {
    "vi": dict(title="Ghi chú bài giảng", src="Nguồn", dur="Thời lượng", model="Model phiên âm",
               overview="Tóm tắt tổng quan", take="Ý chính cần nhớ", admin="Thông báo của giảng viên",
               outline="Nội dung theo thời gian", kw="Từ khóa tự động (theo nguyên dạng)",
               term="Từ khóa", cnt="Số lần", at="Xuất hiện lần đầu", terms="Thuật ngữ và giải thích",
               q="Câu hỏi ôn tập", gloss="Thuật ngữ trong danh sách của bạn", nollm=
               "_Chưa có tóm tắt (chạy với --llm claude|gemini|ollama để tạo)._"),
    "ru": dict(title="Конспект лекции", src="Источник", dur="Длительность", model="Модель распознавания",
               overview="Краткое содержание", take="Главное", admin="Объявления преподавателя",
               outline="Содержание по времени", kw="Ключевые слова (автоматически)",
               term="Термин", cnt="Раз", at="Впервые", terms="Термины и пояснения",
               q="Вопросы для повторения", gloss="Термины из вашего списка", nollm=
               "_Краткое содержание не создано (запустите с --llm claude|gemini|ollama)._"),
}


def write_report(outdir: Path, stem: str, meta: dict, kws: list[dict], summ: dict | None,
                 gloss: list[dict] | None, lang: str) -> Path:
    h = H[lang]
    L = [f"# {h['title']}: {stem}", "",
         f"- {h['src']}: `{meta.get('audio', stem)}`",
         f"- {h['dur']}: {fmt_ts(meta.get('duration', 0))}",
         f"- {h['model']}: {meta.get('model', '?')}", ""]

    if summ:
        L += [f"## {h['overview']}", "", summ["overview"], ""]
        if summ["takeaways"]:
            L += [f"## {h['take']}", ""] + [f"- {t}" for t in summ["takeaways"]] + [""]
        admin = [(s, s.get("admin", "").strip()) for s in summ["sections"]]
        admin = [(s, a) for s, a in admin if a]
        if admin:
            L += [f"## {h['admin']}", ""] + [
                f"- {a}" if re.match(r"\[\d\d:\d\d:\d\d\]", a) else f"- [{fmt_ts(s['start'])}] {a}"
                for s, a in admin] + [""]
        L += [f"## {h['outline']}", ""]
        for s in summ["sections"]:
            L += [f"### [{fmt_ts(s['start'])} – {fmt_ts(s['end'])}] {s.get('title', '')}", ""]
            L += [f"- {p}" for p in s.get("points", [])] + [""]
    else:
        L += [h["nollm"], ""]

    L += [f"## {h['kw']}", "", f"| {h['term']} | {h['cnt']} | {h['at']} |", "|---|---|---|"]
    L += [f"| {k['term']} | {k['count']} | {fmt_ts(k['first'])} |" for k in kws] + [""]

    if gloss:
        L += [f"## {h['gloss']}", "", f"| {h['term']} | {h['cnt']} | {h['at']} |", "|---|---|---|"]
        L += [f"| {g['term']} | {g['count']} | {fmt_ts(g['first']) if g['first'] is not None else '—'} |"
              for g in gloss] + [""]

    if summ:
        seen, rows = set(), []
        for s in summ["sections"]:
            for t in s.get("terms", []):
                if not isinstance(t, dict):
                    continue
                name = re.sub(r"\s*\(.*?\)\s*", " ", str(t.get("term", ""))).strip()
                key = _term_key(name)
                if name and key not in seen:
                    seen.add(key)
                    rows.append(f"- **{name}** — {t.get('explain', '')}")
        if rows:
            L += [f"## {h['terms']}", ""] + rows + [""]
        if summ["review_questions"]:
            L += [f"## {h['q']}", ""] + [f"{i}. {q}" for i, q in enumerate(summ["review_questions"], 1)] + [""]

    path = outdir / f"{stem}.report.md"
    path.write_text("\n".join(L), "utf-8")
    return path


# --------------------------------------------------------------------------- #
# Chạy chính
# --------------------------------------------------------------------------- #

def choose_llm(args) -> LLM | None:
    kind = args.llm
    if kind == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            kind = "claude"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            kind = "gemini"
        else:
            log("[!] Không thấy ANTHROPIC_API_KEY / GEMINI_API_KEY -> bỏ qua bước tóm tắt "
                "(dùng --llm ollama nếu có Ollama chạy sẵn).")
            return None
    if kind == "none":
        return None
    return LLM(kind, args.llm_model)


def lectures_root(audio: Path) -> Path | None:
    """Thư mục 'lectures' chứa file (nếu có) — để biết file nằm trong thư mục môn nào."""
    for par in audio.resolve().parents:
        if par.name.lower() == "lectures":
            return par
    return None


def subject_folder(audio: Path) -> str:
    """lectures/<môn>/bai.m4a -> '<môn>'; file nằm thẳng trong lectures -> ''."""
    root = lectures_root(audio)
    if not root:
        return ""
    rel = audio.resolve().relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def subject_name(folder: str, data_dir: Path | None) -> str:
    """Tên môn tiếng Nga cho prompt: tra data/subjects.json, không có thì dùng chính tên thư mục."""
    if not folder:
        return ""
    if data_dir:
        f = data_dir / "subjects.json"
        if f.exists():
            try:
                m = json.loads(f.read_text("utf-8"))
                if folder in m:
                    return str(m[folder])
            except Exception:
                pass
    return folder


def process(audio: Path, args) -> None:
    folder = subject_folder(audio)
    outdir = Path(args.out) / folder / audio.stem if folder else Path(args.out) / audio.stem
    if folder:                          # file nằm trong thư mục môn: tên môn lấy từ thư mục (ưu tiên hơn --subject)
        root = lectures_root(audio)
        args = argparse.Namespace(**{**vars(args), "subject": subject_name(folder, root.parent if root else None)})
    outdir.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    log(f"\n=== {audio.name}  ->  {outdir}")

    segs, meta = transcribe(audio, outdir, args)
    if not segs:
        log("Không nhận được lời nói nào trong file audio.")
        return
    paras = make_paragraphs(segs)
    write_transcripts(segs, paras, outdir, stem)
    log(f"[2/4] Từ khóa ({len(segs)} đoạn, {sum(len(s['text']) for s in segs)} ký tự)...")
    kws = extract_keywords(segs, args.keywords)
    gloss = count_glossary(segs, read_terms(Path(args.terms))) if args.terms else None

    summ = None
    llm = choose_llm(args)
    if llm and llm.kind == "ollama" and _MODEL_CACHE:
        _MODEL_CACHE.clear()            # nhả ~3 GB VRAM của Whisper để Qwen chạy trọn trên GPU
        import gc
        gc.collect()
    if llm:
        try:
            summ = summarize(paras, llm, args, outdir / f"{stem}.summary.json")
        except Exception as e:
            log(f"[!] Bước tóm tắt lỗi: {e}\n    Phiên âm + từ khóa vẫn được lưu; chạy lại để làm tiếp.")
    log("[4/4] Viết báo cáo...")
    rpt = write_report(outdir, stem, meta, kws, summ, gloss, args.lang)
    log(f"Xong: {rpt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio bài giảng (tiếng Nga) -> phiên âm, tóm tắt, từ khóa")
    ap.add_argument("audio", nargs="*", help="file audio/video (mp3, m4a, wav, mp4, ...)")
    ap.add_argument("--out", default=os.environ.get("LECTURE_TOOL_OUT", "lecture_out"),
                    help="thư mục kết quả (mặc định: biến LECTURE_TOOL_OUT hoặc ./lecture_out)")
    ap.add_argument("--subject", default="", help='tên môn, vd "Строительная механика"')
    ap.add_argument("--lang", choices=["vi", "ru"], default="vi", help="ngôn ngữ của bản tóm tắt")
    ap.add_argument("--model", default="large-v3",
                    help="Whisper: large-v3 (mặc định, chính xác nhất) | large-v3-turbo (nhanh hơn ~2x, sai nhiều hơn)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--compute-type", default="auto", help="auto | int8 | float16 | ...")
    ap.add_argument("--beam-size", type=int, default=None, help="mặc định: 5 (GPU) / 2 (CPU)")
    ap.add_argument("--normalize", action="store_true", help="khuếch đại audio quá nhỏ trước khi phiên âm")
    ap.add_argument("--no-vad", action="store_true", help="tắt bộ lọc VAD (không cắt khoảng lặng)")
    ap.add_argument("--vad-threshold", type=float, default=0.25,
                    help="ngưỡng VAD, thấp hơn = giữ nhiều lời hơn. Mặc định 0.25")
    ap.add_argument("--no-preemph", action="store_true", help="tắt lọc nhấn cao tần (mặc định bật)")
    ap.add_argument("--strict-conf", action="store_true",
                    help="bỏ các đoạn Whisper kém tự tin (mặc định: giữ lại để không mất chữ)")
    ap.add_argument("--keep-low-conf", action="store_true", help=argparse.SUPPRESS)  # giữ tương thích, nay là mặc định
    ap.add_argument("--batch-size", type=int, default=8,
                    help="phiên âm theo lô (mặc định 8, ít mất chữ hơn). 0 = chế độ tuần tự cũ")
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU (0 = tự động)")
    ap.add_argument("--model-dir", default=os.environ.get("LECTURE_TOOL_MODEL_DIR"),
                    help="thư mục lưu model Whisper (mặc định: biến LECTURE_TOOL_MODEL_DIR)")
    ap.add_argument("--prompt", default=None,
                    help="câu gợi ý cho Whisper (mặc định: KHÔNG dùng vì dễ bị chép lại; 'default' = gợi ý ngành xây dựng)")
    ap.add_argument("--terms", default=None,
                    help="file .txt danh sách thuật ngữ của bạn (mỗi dòng 1 thuật ngữ tiếng Nga)")
    ap.add_argument("--keywords", type=int, default=30, help="số từ khóa tự động")
    ap.add_argument("--llm", choices=["auto", "none", "claude", "gemini", "ollama"], default="auto")
    ap.add_argument("--llm-model", default=None, help="tên model LLM (mặc định theo từng nhà cung cấp)")
    ap.add_argument("--chunk-min", type=int, default=10, help="mỗi phần tóm tắt dài ~ bao nhiêu phút")
    ap.add_argument("--chunk-chars", type=int, default=12000, help="giới hạn ký tự mỗi phần")
    ap.add_argument("--force", action="store_true", help="làm lại từ đầu, bỏ qua kết quả đã lưu")
    ap.add_argument("--check-gpu", action="store_true", help="kiểm tra Whisper có chạy được trên GPU không")
    args = ap.parse_args()

    if args.check_gpu:
        check_gpu(args)
        return
    if not args.audio:
        ap.error("cần ít nhất một file audio (hoặc dùng --check-gpu)")

    exts = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".mov"}
    files: list[Path] = []
    for a in args.audio:
        p = Path(a)
        if p.is_dir():                       # cả thư mục (kể cả thư mục con theo môn)
            files += sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts)
        elif any(ch in a for ch in "*?["):   # PowerShell không tự mở rộng ký tự đại diện
            files += sorted(x for x in Path(p.parent).glob(p.name) if x.suffix.lower() in exts)
        elif p.exists():
            files.append(p)
        else:
            log(f"Không tìm thấy file: {a}")
    if len(files) > 1:
        log(f"Tìm thấy {len(files)} file — bài nào đã xong sẽ được bỏ qua.")
    for i, p in enumerate(files, 1):
        if len(files) > 1:
            log(f"\n##### Bài {i}/{len(files)}")
        try:
            process(p, args)
        except KeyboardInterrupt:
            raise
        except Exception as e:               # một bài lỗi không làm dừng cả loạt
            log(f"[!] Lỗi khi xử lý {p.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
