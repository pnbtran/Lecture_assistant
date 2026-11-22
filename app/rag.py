"""
rag.py — trợ lý hỏi đáp trên tài liệu của bạn, trả lời KÈM NGUỒN, chạy hoàn toàn trên máy (Ollama).

  index   : đọc bài giảng đã phiên âm (lecture_out), PDF (papers), bảng/ghi chú (notes) -> chỉ mục
  ask     : hỏi một câu, nhận câu trả lời có trích dẫn [1], [2]... và danh sách nguồn
  chat    : hỏi liên tục nhiều câu
  sources : xem chỉ mục đang có những tài liệu nào

Ví dụ (PowerShell, trong D:\\research_assistant\\lecture_tool):
  .venv\\Scripts\\python.exe rag.py index
  .venv\\Scripts\\python.exe rag.py ask "Điều kiện biên tự nhiên là gì?"
  .venv\\Scripts\\python.exe rag.py chat

Cách hoạt động: tài liệu được cắt thành đoạn nhỏ có địa chỉ (bài + mốc thời gian, PDF + trang,
bảng + dòng). Câu hỏi được tìm theo 2 cách (nghĩa — model bge-m3 hiểu cả tiếng Việt lẫn tiếng Nga;
và từ khoá — BM25), ghép kết quả, rồi Qwen trả lời CHỈ dựa trên các đoạn tìm được. Mọi trích dẫn
được kiểm tra lại bằng code: số [n] không tồn tại sẽ bị gỡ và cảnh báo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lecture_tool as lt  # noqa: E402

INDEX_VERSION = 1
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen3.5:9b")
VERIFY_MODEL = os.environ.get("RAG_VERIFY_MODEL", "gemma3:4b")
NUM_CTX = int(os.environ.get("LECTURE_TOOL_NUM_CTX", "8192"))


def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, flush=True)


# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #

def default_data_dir() -> Path:
    out = os.environ.get("LECTURE_TOOL_OUT")
    if out:
        return Path(out).parent                      # D:\research_assistant\data
    return Path(__file__).resolve().parent.parent / "data"


def default_index_dir(data: Path) -> Path:
    return data.parent / "index"                     # D:\research_assistant\index


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

def _host() -> str:
    h = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return h if h.startswith("http") else "http://" + h


def ollama_embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    import requests
    r = requests.post(f"{_host()}/api/embed", timeout=600,
                      json={"model": model, "input": texts, "keep_alive": "3m"})
    if r.status_code == 404:
        raise RuntimeError(f"Ollama chưa có model '{model}'. Chạy: ollama pull {model}")
    r.raise_for_status()
    return r.json()["embeddings"]


def ollama_generate(prompt: str, model: str = CHAT_MODEL, json_mode: bool = False,
                    temperature: float = 0.2, max_tokens: int = 1500) -> str:
    import requests
    body = {"model": model, "prompt": prompt, "stream": False, "think": False,
            "options": {"num_ctx": NUM_CTX, "temperature": temperature, "num_predict": max_tokens}}
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{_host()}/api/generate", timeout=900, json=body)
    if r.status_code == 404:
        raise RuntimeError(f"Ollama chưa có model '{model}'. Chạy: ollama pull {model}")
    r.raise_for_status()
    return r.json()["response"]


def check_ollama() -> None:
    import requests
    try:
        requests.get(f"{_host()}/api/version", timeout=5).raise_for_status()
    except Exception:
        sys.exit("Không kết nối được Ollama. Mở app Ollama (biểu tượng con lạc đà) hoặc chạy 'ollama serve' rồi thử lại.")


# --------------------------------------------------------------------------- #
# Ngày của tài liệu (đọc từ tên file: 16-10-..., 16.10-..., 2026-10-16_...)
# --------------------------------------------------------------------------- #

def doc_date(name: str):
    from datetime import date
    today = date.today()
    explicit_year = True
    m = re.search(r"(20\d\d)[-._](\d{1,2})[-._](\d{1,2})", name)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"\D{0,3}(\d{1,2})[-._](\d{1,2})(?:[-._](20\d\d|\d\d))?(?!\d)", name)
        if not m:
            return None
        d, mo = int(m.group(1)), int(m.group(2))
        if m.group(3):
            y = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
        else:
            y, explicit_year = today.year, False
    try:
        dt = date(y, mo, d)
    except ValueError:
        return None
    if not explicit_year and dt > today:      # năm không ghi và ngày ở tương lai -> năm ngoái
        dt = date(y - 1, mo, d)
    return dt


def fmt_date(dt) -> str:
    return dt.strftime("%d.%m.%Y") if dt else ""


# --------------------------------------------------------------------------- #
# 1. Cắt tài liệu thành đoạn có địa chỉ
# --------------------------------------------------------------------------- #

def chunks_from_lecture(seg_path: Path, max_chars: int = 900, max_sec: float = 120) -> list[dict]:
    """Cửa sổ ~1-2 phút lời giảng; đoạn sau gối lên đoạn trước 1 câu để không cắt đứt ý."""
    stem = seg_path.name[: -len(".segments.jsonl")]
    segs = lt.clean_segments(lt.read_jsonl(seg_path))
    out, cur = [], []
    for s in segs:
        if cur and (sum(len(x["text"]) for x in cur) + len(s["text"]) > max_chars
                    or s["end"] - cur[0]["start"] > max_sec):
            out.append(cur)
            # gối 1 câu sang đoạn sau (chỉ khi đoạn trước có >1 câu, tránh 2 đoạn trùng mốc thời gian)
            cur = cur[-1:] if len(cur) > 1 and len(cur[-1]["text"]) < max_chars // 3 else []
        cur.append(s)
    if cur:
        out.append(cur)
    res = []
    for g in out:
        res.append({"kind": "lecture", "doc": stem, "start": g[0]["start"], "end": g[-1]["end"],
                    "text": " ".join(x["text"] for x in g)})
    return res


def chunks_from_summary(sum_path: Path) -> list[dict]:
    """Bản tóm tắt (do Qwen viết) — đánh dấu riêng vì có thể sai, chỉ dùng như 'bản đồ'."""
    stem = sum_path.name[: -len(".summary.json")]
    try:
        d = json.loads(sum_path.read_text("utf-8"))
    except Exception:
        return []
    res = []
    for s in d.get("sections", []):
        pts = "\n".join(f"- {p}" for p in s.get("points", []))
        res.append({"kind": "summary", "doc": stem, "start": s.get("start"), "end": s.get("end"),
                    "text": f"{s.get('title', '')}\n{pts}"})
    for s in d.get("sections", []):
        if str(s.get("admin") or "").strip():
            res.append({"kind": "summary", "doc": stem, "start": s.get("start"), "end": s.get("end"),
                        "text": "Thông báo của giảng viên: " + str(s["admin"])})
    return res


def _split_text(text: str, max_chars: int = 1200) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > max_chars:                        # đoạn quá dài: cắt theo câu
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 1 > max_chars and cur:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        out.append(cur)
    return out


def chunks_from_pdf(path: Path) -> tuple[list[dict], int]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    res, empty = [], 0
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"-\n(\w)", r"\1", text)           # nối từ bị ngắt dòng bằng gạch nối
        if len(text.strip()) < 40:
            empty += 1
            continue
        for part in _split_text(text):
            res.append({"kind": "pdf", "doc": path.name, "page": i, "text": part})
    return res, empty


def chunks_from_table(path: Path) -> list[dict]:
    """Mỗi dòng của bảng -> 1 đoạn 'Tiêu đề cột: giá trị; ...' kèm tên sheet + số dòng."""
    rows: list[tuple[str, int, list]] = []
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        for ws in wb.worksheets:
            for r_i, row in enumerate(ws.iter_rows(values_only=True), 1):
                rows.append((ws.title, r_i, list(row)))
    else:
        import csv
        with open(path, encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except Exception:
                dialect = csv.excel
            for r_i, row in enumerate(csv.reader(f, dialect), 1):
                rows.append(("csv", r_i, row))
    res, header, cur_sheet = [], None, None
    for sheet, r_i, row in rows:
        vals = ["" if v is None else str(v).strip() for v in row]
        if not any(vals):
            continue
        if sheet != cur_sheet:
            cur_sheet, header = sheet, None
        if header is None:                                # dòng không trống đầu tiên = tiêu đề
            header = [v or f"cột {j + 1}" for j, v in enumerate(vals)]
            continue
        parts = [f"{header[j] if j < len(header) else f'cột {j + 1}'}: {v}" for j, v in enumerate(vals) if v]
        text = "; ".join(parts)
        if len(text) > 2500:
            text = text[:2500] + "…"
        res.append({"kind": "table", "doc": path.name, "sheet": sheet, "row": r_i, "text": text})
    return res


def chunks_from_note(path: Path) -> list[dict]:
    text = path.read_text("utf-8", errors="replace")
    return [{"kind": "note", "doc": path.name, "part": i, "text": t}
            for i, t in enumerate(_split_text(text), 1)]


SKIP_DIR = re.compile(r"^(bench|test_.*|_.*|\..*)$", re.IGNORECASE)


def subject_of(file_key: str) -> str:
    """Môn = thư mục con cấp 1: lecture_out/<môn>/<bài>/..., papers/<môn>/x.pdf, notes/<môn>/x.xlsx."""
    parts = Path(file_key.replace("\\", "/")).parts
    if not parts:
        return ""
    if parts[0] == "lecture_out":
        return parts[1] if len(parts) >= 4 else ""
    return parts[1] if len(parts) >= 3 else ""


def discover(data: Path) -> list[tuple[Path, str]]:
    """Liệt kê mọi tài liệu cần đưa vào chỉ mục: (đường dẫn, loại)."""
    found: list[tuple[Path, str]] = []
    lo = data / "lecture_out"
    if lo.is_dir():
        for p in sorted(lo.rglob("*.segments.jsonl")):
            if any(SKIP_DIR.match(x) for x in p.relative_to(lo).parts[:-1]):
                continue                     # bỏ thư mục thử nghiệm: bench, test_A, _cu...
            if p.with_name(p.name.replace(".segments.jsonl", ".meta.json")).exists():
                found.append((p, "lecture"))
                sp = p.with_name(p.name.replace(".segments.jsonl", ".summary.json"))
                if sp.exists():
                    found.append((sp, "summary"))
    for sub in ("papers", "notes"):
        d = data / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            ext = p.suffix.lower()
            if not p.is_file() or p.name.startswith(("~$", ".")):
                continue
            if any(SKIP_DIR.match(x) for x in p.relative_to(d).parts[:-1]):
                continue
            if ext == ".pdf":
                found.append((p, "pdf"))
            elif ext in (".xlsx", ".xlsm", ".csv"):
                found.append((p, "table"))
            elif ext in (".md", ".txt"):
                found.append((p, "note"))
    return found


def make_chunks_for(path: Path, kind: str) -> tuple[list[dict], str]:
    if kind == "lecture":
        return chunks_from_lecture(path), ""
    if kind == "summary":
        return chunks_from_summary(path), ""
    if kind == "pdf":
        ch, empty = chunks_from_pdf(path)
        warn = (f"{empty} trang không có chữ (có thể là bản scan, cần OCR)" if empty else "")
        return ch, warn
    if kind == "table":
        return chunks_from_table(path), ""
    return chunks_from_note(path), ""


# --------------------------------------------------------------------------- #
# 2. Chỉ mục (lưu dạng file thường: chunks.jsonl + emb.npy + manifest.json)
# --------------------------------------------------------------------------- #

def file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def load_index(idx: Path):
    import numpy as np
    man_p, ch_p, emb_p = idx / "manifest.json", idx / "chunks.jsonl", idx / "emb.npy"
    if not (man_p.exists() and ch_p.exists() and emb_p.exists()):
        return None
    man = json.loads(man_p.read_text("utf-8"))
    if man.get("version") != INDEX_VERSION:
        return None
    chunks = lt.read_jsonl(ch_p)
    emb = np.load(emb_p)
    if len(chunks) != len(emb):
        return None
    return man, chunks, emb


def cmd_index(args) -> None:
    import numpy as np
    data, idx = Path(args.data), Path(args.index)
    idx.mkdir(parents=True, exist_ok=True)
    check_ollama()
    old = None if args.rebuild else load_index(idx)
    old_files = old[0]["files"] if old else {}
    old_by_file: dict[str, list[int]] = {}
    if old:
        for i, c in enumerate(old[1]):
            old_by_file.setdefault(c["file"], []).append(i)

    found = discover(data)
    if not found:
        sys.exit(f"Không tìm thấy tài liệu nào trong {data} (lecture_out, papers, notes).")
    chunks, embs, files, todo = [], [], {}, []
    for p, kind in found:
        key = str(p.relative_to(data))
        sig = file_sig(p)
        files[key] = {"sig": sig, "kind": kind}
        if old and old_files.get(key, {}).get("sig") == sig and key in old_by_file:
            for i in old_by_file[key]:
                chunks.append(old[1][i])
                embs.append(old[2][i])
        else:
            todo.append((p, kind, key))

    log(f"Tài liệu: {len(found)}  |  giữ nguyên: {len(found) - len(todo)}  |  cần xử lý: {len(todo)}")
    for n, (p, kind, key) in enumerate(todo, 1):
        new, warn = make_chunks_for(p, kind)
        log(f"  [{n}/{len(todo)}] {key}  ->  {len(new)} đoạn" + (f"   (!) {warn}" if warn else ""))
        if not new:
            continue
        vecs = []
        for b in range(0, len(new), 16):
            batch = new[b:b + 16]
            vecs += ollama_embed([passage_text(c) for c in batch])
            log(f"\r        embed {min(b + 16, len(new))}/{len(new)}", end="")
        log("")
        for c, v in zip(new, vecs):
            c["file"] = key
            c["id"] = hashlib.sha1(f"{key}|{c['text'][:200]}|{len(chunks)}".encode()).hexdigest()[:12]
            chunks.append(c)
            embs.append(np.asarray(v, dtype=np.float32))

    if not chunks:
        sys.exit("Không có đoạn văn bản nào để đưa vào chỉ mục.")
    E = np.vstack(embs).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    np.save(idx / "emb.npy", E)
    with open(idx / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (idx / "manifest.json").write_text(json.dumps(
        {"version": INDEX_VERSION, "embed_model": EMBED_MODEL, "data": str(data), "files": files,
         "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), "utf-8")
    kinds = Counter(c["kind"] for c in chunks)
    log(f"Xong: {len(chunks)} đoạn ({', '.join(f'{k}: {v}' for k, v in kinds.items())}) -> {idx}")


def passage_text(c: dict) -> str:
    """Văn bản đem đi embed: thêm tên tài liệu để câu hỏi kiểu 'bài 7 nói gì' cũng tìm được."""
    return f"{c['doc']}\n{c['text']}"


# --------------------------------------------------------------------------- #
# 3. Tìm kiếm: nghĩa (bge-m3) + từ khoá (BM25), ghép bằng Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #

TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ0-9]+|[А-Яа-яЁё]+")


def _norm_tok(w: str) -> str:
    w = w.lower().replace("ё", "е")
    if re.match(r"[а-я]", w):
        try:
            p = lt._morph().parse(w)
            if p:
                return p[0].normal_form.replace("ё", "е")
        except Exception:
            pass
        return w[:6] if len(w) > 6 else w            # dự phòng: cắt gốc thô
    return w


def tokenize(text: str) -> list[str]:
    return [_norm_tok(w) for w in TOKEN_RE.findall(text) if len(w) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


_BM25_CACHE: dict[str, BM25] = {}


def get_bm25(idx: Path, chunks: list[dict]) -> BM25:
    key = str(idx)
    if key not in _BM25_CACHE:
        cache = idx / "bm25_tokens.json"
        toks = None
        if cache.exists():
            try:
                d = json.loads(cache.read_text("utf-8"))
                if d.get("n") == len(chunks) and d.get("ids") == [c.get("id") for c in chunks]:
                    toks = d["toks"]
            except Exception:
                toks = None
        if toks is None:
            toks = [tokenize(passage_text(c)) for c in chunks]
            cache.write_text(json.dumps({"n": len(chunks), "ids": [c.get("id") for c in chunks],
                                        "toks": toks}, ensure_ascii=False), "utf-8")
        _BM25_CACHE[key] = BM25(toks)
    return _BM25_CACHE[key]


REWRITE_PROMPT = """\
Convert the user's question into search queries for a database of RUSSIAN university lectures on
structural mechanics / civil engineering (transcripts), research papers and notes.
Return ONLY JSON: {"ru": "the question translated into natural Russian, using standard Russian technical terms",
"keywords": ["2-6 Russian terms that are EXPLICITLY in the question or are its direct translation"]}
Do NOT add related topics, theories or terms that the question does not mention.
Question: """


def expand_query(q: str, use_llm: bool = True) -> list[str]:
    queries = [q]
    if not use_llm:
        return queries
    try:
        d = lt.parse_json(ollama_generate(REWRITE_PROMPT + q, json_mode=True, max_tokens=300))
        if isinstance(d, dict):
            if d.get("ru"):
                queries.append(str(d["ru"]))
            kws = d.get("keywords") or []
            if kws:
                queries.append(" ".join(str(k) for k in kws))
    except Exception as e:
        log(f"(không dịch được câu hỏi sang tiếng Nga: {e} — tìm bằng câu gốc)")
    return queries


def search(idx: Path, question: str, k: int = 8, kinds: set[str] | None = None,
           doc_filter: str | None = None, use_llm_rewrite: bool = True,
           subject: str | None = None) -> tuple[list[dict], list[str]]:
    import numpy as np
    loaded = load_index(idx)
    if not loaded:
        sys.exit(f"Chưa có chỉ mục ở {idx}. Chạy trước: rag.py index")
    man, chunks, E = loaded
    queries = expand_query(question, use_llm_rewrite)
    allowed = np.ones(len(chunks), dtype=bool)
    for i, c in enumerate(chunks):
        if kinds and c["kind"] not in kinds:
            allowed[i] = False
        if doc_filter and doc_filter.lower() not in c["doc"].lower():
            allowed[i] = False
        if subject is not None and subject_of(c.get("file", "")).lower() != subject.lower():
            allowed[i] = False
    if not allowed.any():
        return [], queries

    rrf = np.zeros(len(chunks))
    qv = np.asarray(ollama_embed(queries), dtype=np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9
    dense = E @ qv.T                                         # (n_chunks, n_queries)
    bm = get_bm25(idx, chunks)
    rank_lists = [dense[:, j] for j in range(dense.shape[1])]
    for qq in queries:
        rank_lists.append(np.asarray(bm.scores(tokenize(qq))))
    for sc in rank_lists:
        sc = np.where(allowed, sc, -np.inf)
        order = np.argsort(-sc)[:60]
        for r, i in enumerate(order):
            if np.isfinite(sc[i]) and sc[i] > 0:
                rrf[i] += 1.0 / (60 + r)
    rrf[~allowed] = 0
    is_sum = np.array([c["kind"] == "summary" for c in chunks])
    rrf[is_sum] *= 0.6                                    # ưu tiên lời giảng gốc hơn bản tóm tắt máy
    best = [i for i in np.argsort(-rrf) if rrf[i] > 0]
    # bản tóm tắt chỉ là 'bản đồ': tối đa 2 đoạn; mỗi đoạn tóm tắt kéo theo đoạn LỜI GIẢNG GỐC khớp nhất
    # trong cùng khoảng thời gian, để trích dẫn được đúng phút thầy nói (không chỉ đầu phần tóm tắt)
    sim = dense.max(axis=1)
    picked, n_sum = [], 0
    for i in best:
        if i in picked:
            continue
        c = chunks[i]
        if c["kind"] == "summary":
            if n_sum >= 2:
                continue
            n_sum += 1
            picked.append(i)
            if c.get("start") is not None and c.get("end") is not None:
                cand = [j for j, x in enumerate(chunks)
                        if x["kind"] == "lecture" and x["doc"] == c["doc"] and allowed[j]
                        and c["start"] - 1 <= x["start"] <= c["end"] + 1 and j not in picked]
                if cand:
                    picked.append(max(cand, key=lambda j: sim[j]))
        else:
            picked.append(i)
        if len(picked) >= k:
            break
    out = []
    for rank, i in enumerate(picked[:k + 1]):
        c = dict(chunks[i])
        c["score"] = float(rrf[i])
        c["sim"] = float(sim[i])
        c["subject"] = subject_of(c.get("file", ""))
        # 3 đoạn bài giảng đứng đầu: nối thêm đoạn LIỀN SAU (ý của thầy hay bị cắt đôi giữa 2 đoạn)
        if c["kind"] == "lecture" and rank < 4 and i + 1 < len(chunks):
            nx = chunks[i + 1]
            if nx["kind"] == "lecture" and nx["doc"] == c["doc"] and nx["start"] > c["start"] and (i + 1) not in picked:
                extra = nx["text"]
                tail = c["text"][-200:]
                cut = extra.find(tail[-80:]) if len(tail) >= 80 else -1      # bỏ phần gối đầu trùng lặp
                if cut >= 0:
                    extra = extra[cut + 80:]
                c["text"] = c["text"] + " " + extra.strip()
                c["end"] = nx["end"]
        out.append(c)
    return out, queries


# --------------------------------------------------------------------------- #
# 4. Trả lời có trích dẫn + kiểm tra trích dẫn
# --------------------------------------------------------------------------- #

def cite_label(c: dict, with_date: bool = False) -> str:
    k = c["kind"]
    if with_date:
        dt = doc_date(c["doc"])
        base = cite_label(c)
        return base + (f"; ngày {fmt_date(dt)}" if dt else "")
    if k == "lecture":
        return f"{c['doc']} · {lt.fmt_ts(c['start'])}"
    if k == "summary":
        st = lt.fmt_ts(c["start"]) if c.get("start") is not None else ""
        return f"{c['doc']} · {st} (bản tóm tắt tự động)"
    if k == "pdf":
        return f"{c['doc']} · tr. {c['page']}"
    if k == "table":
        return f"{c['doc']} · {c['sheet']}, dòng {c['row']}"
    return f"{c['doc']} · phần {c.get('part', 1)}"


ANSWER_PROMPT = """\
You are a research assistant for a Vietnamese master's student of civil engineering at a Russian university.
Today is {today}. Answer the QUESTION using ONLY the numbered SOURCES below. Each source header shows
its document and, if known, the date of the lecture — use dates to interpret words like "next", "last", "recent". Lecture sources are automatic speech-recognition
transcripts in Russian and may contain recognition errors; interpret obvious errors sensibly
(e.g. "функционал огранжа" = "функционал Лагранжа"), but never invent facts that are not supported by the sources.

Rules:
- Answer in {lang}. Keep key Russian technical terms in Russian with a {lang} gloss in parentheses at first use.
- Every sentence that uses a source must contain its number in square brackets BEFORE the final period,
  e.g. "Mô men tại bản lề bằng không [2]." or "... [1][3]."
- Sources may describe DIFFERENT events (e.g. a test written during one class vs. a test announced for the
  next class, or different lectures). Never merge them into one story: say which lecture/date each fact is from,
  and only include what actually answers the question.
- Only use numbers that exist in SOURCES.
- For yes/no or true/false questions, start with the verdict and cite the sources that justify it in that
  same sentence, e.g. "Sai: lực dọc được xác định theo quy tắc góc tù [2][3]."
- Plain text only: no Markdown (**bold**) and no LaTeX ($...$); write symbols directly, e.g. N, Q, M, EA.
- If the sources do not contain the answer, say so clearly in one sentence and do not guess.
- Sources marked "(bản tóm tắt tự động)" are machine summaries: prefer the original lecture/paper sources.
- Do not add facts from a source that is about a different topic than the question.
- Be concise: 3-10 sentences, or a short list if the question asks for steps/conditions.

SOURCES:
{sources}

QUESTION: {question}

ANSWER ({lang}, with [n] citations):"""


def build_sources(hits: list[dict], max_chars: int = 1900) -> str:
    out = []
    for i, c in enumerate(hits, 1):
        t = c["text"].strip()
        if len(t) > max_chars:
            t = t[:max_chars] + "…"
        out.append(f"[{i}] ({cite_label(c, with_date=True)})\n{t}")
    return "\n\n".join(out)


CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def check_citations(answer: str, n: int) -> tuple[str, list[int], list[int]]:
    """Gỡ trích dẫn không tồn tại; trả về (câu trả lời đã sửa, các số hợp lệ đã dùng, các số bị gỡ)."""
    used, bad = [], []

    def fix(m):
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1))]
        if 0 in nums:                    # vd khoảng [0, 1] trong công thức — không phải trích dẫn
            return m.group(0)
        ok = [x for x in nums if 1 <= x <= n]
        bad.extend(x for x in nums if not 1 <= x <= n)
        for x in ok:
            if x not in used:
                used.append(x)
        return "".join(f"[{x}]" for x in ok)

    fixed = CITE_RE.sub(fix, answer)
    return fixed, used, bad


# --------------------------------------------------------------------------- #
# 5. Kiểm tra chéo: model thứ hai (khác họ) đối chiếu từng câu với đúng nguồn nó trích
# --------------------------------------------------------------------------- #

VERIFY_PROMPT = """\
You are a strict fact-checker. For each CLAIM below, decide whether the SOURCES cited by that claim
support it. Sources are Russian lecture transcripts produced by speech recognition (they contain recognition
errors — judge the meaning, not the spelling), research papers or table rows. Claims may be in Vietnamese.

Verdicts:
- "supported": the cited sources clearly state or directly imply the claim.
- "partial": part of the claim is supported, but some detail (number, condition, name) is not in the sources.
- "unsupported": the cited sources do not say this, say something different, or are about another topic.
- "off_topic": the sources do support it, but it is about a DIFFERENT event/situation than the QUESTION asks
  (e.g. the question asks about the NEXT test, the claim describes a test written during an earlier class).
Use ONLY the cited sources; your own knowledge does not count as support.

QUESTION the answer was written for: {question}

Return ONLY JSON: {{"results": [{{"id": 1, "verdict": "supported|partial|unsupported|off_topic", "note": "max 15 words, in Vietnamese"}}]}}

SOURCES:
{sources}

CLAIMS:
{claims}
"""

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
CITE_AFTER_PUNCT = re.compile(r"([.!?…])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)")


def normalize_citations(ans: str) -> str:
    """'… câu. [1][3] Câu sau' -> '… câu [1][3]. Câu sau' (trích dẫn thuộc về câu đứng TRƯỚC nó)."""
    def mv(m):
        cites = re.sub(r"\s+", "", m.group(2))
        return f" {cites}{m.group(1)} "
    return re.sub(r"[ \t]{2,}", " ", CITE_AFTER_PUNCT.sub(mv, ans)).strip()
NOT_FOUND_RE = re.compile(r"((không|chưa) (tìm thấy|có thông tin|đề cập|nêu|nói|cung cấp)|không chứa|"
                          r"нет (информации|сведений)|не содерж|не найден|не упомина)",
                          re.IGNORECASE)


def split_claims(ans: str) -> list[dict]:
    """Tách câu trả lời thành các câu; mỗi câu kèm số nguồn nó trích."""
    out = []
    for part in SENT_SPLIT.split(ans):
        t = part.strip()
        if len(re.sub(CITE_RE, "", t).strip(" -*•.:")) < 8:
            continue
        cites = []
        for m in CITE_RE.finditer(t):
            for x in re.split(r"\s*[,;]\s*", m.group(1)):
                if x.isdigit() and int(x) not in cites:
                    cites.append(int(x))
        out.append({"text": t, "cites": cites})
    return out


def verify(claims: list[dict], hits: list[dict], model: str = VERIFY_MODEL, question: str = "") -> list[dict]:
    """Gắn verdict cho từng câu. Câu không trích nguồn -> 'no_citation' (kiểm bằng code, không cần model)."""
    to_check = [c for c in claims if c["cites"]]
    for c in claims:
        if not c["cites"]:
            c["verdict"] = "not_found" if NOT_FOUND_RE.search(c["text"]) else "no_citation"
    if not to_check:
        return claims
    used = sorted({n for c in to_check for n in c["cites"] if 1 <= n <= len(hits)})
    src = "\n\n".join(f"[{n}] ({cite_label(hits[n - 1], with_date=True)}) {hits[n - 1]['text'][:1900]}" for n in used)
    cl = "\n".join(f"{i}. (cites {', '.join(f'[{n}]' for n in c['cites'])}) {re.sub(CITE_RE, '', c['text']).strip()}"
                   for i, c in enumerate(to_check, 1))
    try:
        raw = ollama_generate(VERIFY_PROMPT.format(sources=src, claims=cl, question=question), model=model,
                              json_mode=True, temperature=0.0, max_tokens=800)
        d = lt.parse_json(raw)
        res = d.get("results", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
        by_id = {int(r.get("id", 0)): r for r in res if isinstance(r, dict) and str(r.get("id", "")).isdigit()}
    except Exception as e:
        log(f"(không chạy được bước kiểm tra chéo: {type(e).__name__}: {e})")
        by_id = {}
    for i, c in enumerate(to_check, 1):
        r = by_id.get(i)
        v = str(r.get("verdict", "")).lower() if r else ""
        c["verdict"] = v if v in ("supported", "partial", "unsupported", "off_topic") else "unchecked"
        c["note"] = str(r.get("note", "")) if r else ""
    return claims


def answer(idx: Path, question: str, k: int = 8, lang: str = "vi", kinds: set[str] | None = None,
           doc_filter: str | None = None, show_context: bool = False, rewrite: bool = True,
           check: bool = True, strict: bool = False, second: bool = False, subject: str | None = None) -> dict:
    t0 = time.time()
    hits, queries = search(idx, question, k=k, kinds=kinds, doc_filter=doc_filter, use_llm_rewrite=rewrite,
                           subject=subject)
    if not hits:
        return {"answer": "Không tìm thấy đoạn tài liệu nào phù hợp.", "sources": [], "hits": [], "queries": queries}
    lang_name = {"vi": "Vietnamese", "ru": "Russian"}[lang]
    from datetime import date
    prompt = ANSWER_PROMPT.format(lang=lang_name, sources=build_sources(hits), question=question,
                                  today=fmt_date(date.today()))
    raw = ollama_generate(prompt, max_tokens=1200).strip()
    fixed, used, bad = check_citations(normalize_citations(raw), len(hits))
    res = {"answer": fixed, "used": used, "bad": bad, "hits": hits, "queries": queries}
    if check:
        t1 = time.time()
        claims = verify(split_claims(fixed), hits, question=question)
        res["claims"] = claims
        res["verify_seconds"] = round(time.time() - t1, 1)
        if strict:                                         # bỏ hẳn các câu nguồn không đỡ
            keep = [c["text"] for c in claims if c["verdict"] not in ("unsupported", "no_citation", "off_topic")]
            res["answer"] = " ".join(keep) if keep else "Không có câu nào trong câu trả lời được nguồn xác nhận."
        else:
            out = fixed
            for c in claims:
                if c["verdict"] in ("unsupported", "no_citation", "off_topic"):
                    out = out.replace(c["text"], c["text"] + " ⚠ ", 1)
                elif c["verdict"] == "partial":
                    out = out.replace(c["text"], c["text"] + " (?)", 1)
            res["answer"] = re.sub(r"[ \t]{2,}", " ", out).strip()
    if second:                                             # ý kiến thứ hai: model khác trả lời độc lập
        raw2 = ollama_generate(prompt, model=VERIFY_MODEL, max_tokens=900).strip()
        res["second"], _, _ = check_citations(raw2, len(hits))
    res["seconds"] = round(time.time() - t0, 1)
    if show_context:
        res["prompt"] = prompt
    return res


VERDICT_VI = {"supported": "✓ có nguồn xác nhận", "partial": "(?) chỉ đúng một phần",
              "unsupported": "⚠ nguồn KHÔNG nói điều này", "no_citation": "⚠ không trích nguồn",
              "off_topic": "⚠ đúng nguồn nhưng nói về việc khác với câu hỏi",
              "not_found": "ℹ báo không tìm thấy", "unchecked": "· chưa kiểm được"}


def print_answer(res: dict, verbose: bool = False) -> None:
    log("")
    log(res["answer"])
    claims = res.get("claims")
    if claims:
        cnt = Counter(c["verdict"] for c in claims)
        ok = cnt.get("supported", 0)
        log(f"\nKiểm tra chéo ({VERIFY_MODEL}): {ok}/{len(claims)} câu được nguồn xác nhận"
            + (f", {cnt['partial']} đúng một phần" if cnt.get("partial") else "")
            + (f", {cnt.get('unsupported', 0) + cnt.get('no_citation', 0) + cnt.get('off_topic', 0)} câu ⚠"
               if cnt.get("unsupported") or cnt.get("no_citation") or cnt.get("off_topic") else ""))
        for i, c in enumerate(claims, 1):
            if c["verdict"] not in ("supported", "not_found") or verbose:
                short = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(CITE_RE, "", c["text"])).strip()
                short = short[:110] + ("…" if len(short) > 110 else "")
                note = f" — {c['note']}" if c.get("note") else ""
                log(f"  {VERDICT_VI.get(c['verdict'], c['verdict'])}: «{short}»{note}")
    hits, used = res.get("hits", []), res.get("used", [])
    if hits:
        log("\nNguồn:")
        for i in used:
            c = hits[i - 1]
            snippet = re.sub(r"\s+", " ", c["text"])[:160]
            log(f"  [{i}] {cite_label(c)}")
            log(f"       «{snippet}…»")
        if not used:
            log("  (!) Câu trả lời KHÔNG trích dẫn nguồn nào — đừng tin nếu chưa tự kiểm tra.")
            log("      Các đoạn tìm được gần nhất:")
            for i, c in enumerate(hits[:3], 1):
                log(f"  [{i}] {cite_label(c)}")
        if res.get("bad"):
            log(f"  (!) Đã gỡ trích dẫn không tồn tại: {sorted(set(res['bad']))}")
        unused = [i for i in range(1, len(hits) + 1) if i not in used]
        if verbose and unused:
            log("  Các đoạn tìm được nhưng không dùng: " + "; ".join(f"[{i}] {cite_label(hits[i - 1])}" for i in unused))
    if res.get("second"):
        log(f"\nÝ kiến thứ hai ({VERIFY_MODEL}, trả lời độc lập trên cùng nguồn):\n{res['second']}")
    if verbose:
        log(f"\n(tìm với: {res.get('queries')}; tổng {res.get('seconds')} s, kiểm tra chéo {res.get('verify_seconds', 0)} s)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cmd_ask(args) -> None:
    check_ollama()
    kinds = set(args.only.split(",")) if args.only else None
    res = answer(Path(args.index), args.question, k=args.k, lang=args.lang, kinds=kinds,
                 doc_filter=args.doc, rewrite=not args.no_rewrite, check=not args.no_check,
                 strict=args.strict, second=args.second, subject=args.mon)
    print_answer(res, args.verbose)
    if args.save:
        log_path = Path(args.index) / "history.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": args.question,
                                "answer": res["answer"],
                                "checks": [{"text": c["text"], "verdict": c["verdict"], "note": c.get("note", "")}
                                           for c in res.get("claims", [])],
                                "sources": [cite_label(res["hits"][i - 1]) for i in res.get("used", [])]},
                               ensure_ascii=False) + "\n")


def cmd_chat(args) -> None:
    check_ollama()
    kinds = set(args.only.split(",")) if args.only else None
    log("Hỏi đáp trên tài liệu của bạn. Gõ câu hỏi rồi Enter; gõ 'q' để thoát.")
    log("Mẹo: thêm  @tên  để chỉ tìm trong một tài liệu, vd: @лек7 điều kiện biên tự nhiên là gì?")
    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit", "thoat", "thoát"):
            break
        if not q:
            continue
        doc = args.doc
        m = re.match(r"@(\S+)\s+(.*)", q)
        if m:
            doc, q = m.group(1), m.group(2)
        try:
            res = answer(Path(args.index), q, k=args.k, lang=args.lang, kinds=kinds, doc_filter=doc,
                         rewrite=not args.no_rewrite, check=not args.no_check, strict=args.strict,
                         second=args.second, subject=args.mon)
            print_answer(res, args.verbose)
        except Exception as e:
            log(f"[!] Lỗi: {type(e).__name__}: {e}")


def cmd_sources(args) -> None:
    loaded = load_index(Path(args.index))
    if not loaded:
        sys.exit("Chưa có chỉ mục. Chạy: rag.py index")
    man, chunks, _ = loaded
    cnt = Counter((subject_of(c.get("file", "")) or "(chung)", c["kind"], c["doc"]) for c in chunks)
    log(f"Chỉ mục tạo lúc {man.get('built_at')} — {len(chunks)} đoạn, model nhúng {man.get('embed_model')}")
    cur = None
    for (subj, kind, doc), n in sorted(cnt.items()):
        if subj != cur:
            log(f"\n[{subj}]")
            cur = subj
        log(f"  {kind:8s} {n:5d}  {doc}")


def main() -> None:
    data = default_data_dir()
    ap = argparse.ArgumentParser(description="Hỏi đáp có trích dẫn trên bài giảng, PDF và ghi chú của bạn")
    ap.add_argument("--data", default=str(data), help=f"thư mục dữ liệu (mặc định {data})")
    ap.add_argument("--index", default=None, help="thư mục chỉ mục (mặc định <data>\\..\\index)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="tạo/cập nhật chỉ mục (chỉ xử lý tài liệu mới hoặc đã đổi)")
    p.add_argument("--rebuild", action="store_true", help="làm lại toàn bộ chỉ mục")

    for name in ("ask", "chat"):
        p = sub.add_parser(name, help="hỏi một câu" if name == "ask" else "hỏi liên tục")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--save", action="store_true", help="lưu câu hỏi + trả lời vào index\\history.jsonl")
        p.add_argument("-k", type=int, default=8, help="số đoạn tài liệu đưa cho model (mặc định 8)")
        p.add_argument("--lang", choices=["vi", "ru"], default="vi")
        p.add_argument("--only", default=None, help="chỉ tìm trong loại: lecture,summary,pdf,table,note")
        p.add_argument("--doc", default=None, help="chỉ tìm trong tài liệu có tên chứa chuỗi này, vd лек7")
        p.add_argument("--mon", default=None, help="chỉ tìm trong một môn (tên thư mục môn), vd стромех")
        p.add_argument("--no-rewrite", action="store_true", help="không dịch câu hỏi sang tiếng Nga trước khi tìm")
        p.add_argument("-v", "--verbose", action="store_true", help="in thêm chi tiết tìm kiếm")
        p.add_argument("--no-check", action="store_true", help="bỏ bước kiểm tra chéo (nhanh hơn ~5-15 s)")
        p.add_argument("--strict", action="store_true", help="tự bỏ các câu mà nguồn không xác nhận")
        p.add_argument("--second", action="store_true",
                       help=f"thêm ý kiến thứ hai: {VERIFY_MODEL} trả lời độc lập để so sánh")

    sub.add_parser("sources", help="xem chỉ mục đang có tài liệu nào")
    args = ap.parse_args()
    if args.index is None:
        args.index = str(default_index_dir(Path(args.data)))
    {"index": cmd_index, "ask": cmd_ask, "chat": cmd_chat, "sources": cmd_sources}[args.cmd](args)


if __name__ == "__main__":
    main()
