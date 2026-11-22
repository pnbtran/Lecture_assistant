"""
app.py — giao diện web cho trợ lý bài giảng (Lecture Assistant) (chạy trên máy bạn, chỉ mở ở 127.0.0.1).

  .venv\\Scripts\\python.exe app.py          -> mở trình duyệt ở http://127.0.0.1:8765

Không cần cài thêm thư viện: dùng http.server có sẵn của Python.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lecture_tool as lt  # noqa: E402
import rag  # noqa: E402

HOST, PORT = "127.0.0.1", int(os.environ.get("RA_PORT", "8765"))
DATA = rag.default_data_dir()
INDEX = rag.default_index_dir(DATA)
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mp4", ".webm"}

LLM_LOCK = threading.Lock()          # Ollama + 8 GB VRAM: mỗi lúc chỉ chạy 1 việc
JOB = {"running": False, "name": "", "log": [], "started": 0.0, "ok": None}


# --------------------------------------------------------------------------- #
# Tìm file gốc cho trích dẫn
# --------------------------------------------------------------------------- #

def audio_for(doc: str) -> Path | None:
    """doc = tên bài (stem) -> file audio trong data\\lectures (đọc meta.json nếu có)."""
    lo = DATA / "lecture_out"
    meta = next(iter(lo.rglob(f"{doc}.meta.json")), None) if lo.is_dir() else None
    name = None
    if meta:
        try:
            name = json.loads(meta.read_text("utf-8")).get("audio")
        except Exception:
            name = None
    lec = DATA / "lectures"
    if not lec.is_dir():
        return None
    for p in lec.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and (p.name == name or p.stem == doc):
            return p
    return None


def safe_under(base: Path, rel: str) -> Path | None:
    try:
        p = (base / rel).resolve()
        p.relative_to(base.resolve())
        return p if p.is_file() else None
    except Exception:
        return None


def source_payload(c: dict, n: int) -> dict:
    d = {"n": n, "kind": c["kind"], "doc": c["doc"], "label": rag.cite_label(c), "subject": c.get("subject", ""),
         "text": re.sub(r"\s+", " ", c["text"]).strip()[:1200]}
    dt = rag.doc_date(c["doc"])
    d["date"] = rag.fmt_date(dt) if dt else ""
    if c["kind"] in ("lecture", "summary") and c.get("start") is not None:
        d["start"] = float(c["start"])
        a = audio_for(c["doc"])
        if a:
            d["audio"] = "/media/lectures/" + urllib.parse.quote(str(a.relative_to(DATA / "lectures")).replace("\\", "/"))
    if c["kind"] in ("pdf", "table", "note"):
        rel = c.get("file", "")
        if rel:
            url = "/media/data/" + urllib.parse.quote(rel.replace("\\", "/"))
            if c["kind"] == "pdf":
                url += f"#page={c.get('page', 1)}"
            d["open"] = url
    return d


# --------------------------------------------------------------------------- #
# Việc chạy nền: cập nhật chỉ mục / xử lý bài giảng mới
# --------------------------------------------------------------------------- #

class _Tee:
    def __init__(self, orig):
        self.orig = orig

    def write(self, s):
        try:
            self.orig.write(s)
        except Exception:
            pass
        for part in s.replace("\r", "\n").split("\n"):
            if part.strip():
                JOB["log"].append(part.rstrip())
                del JOB["log"][:-200]
        return len(s)

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


def run_job(name: str, fn) -> bool:
    if JOB["running"]:
        return False

    def work():
        JOB.update(running=True, name=name, log=[], started=time.time(), ok=None)
        old_out = sys.stdout
        sys.stdout = _Tee(old_out)
        try:
            with LLM_LOCK:
                fn()
            JOB["ok"] = True
        except SystemExit as e:
            JOB["log"].append(f"Dừng: {e}")
            JOB["ok"] = False
        except Exception as e:
            JOB["log"].append(f"Lỗi: {type(e).__name__}: {e}")
            JOB["ok"] = False
        finally:
            sys.stdout = old_out
            JOB["running"] = False
            rag._BM25_CACHE.clear()

    threading.Thread(target=work, daemon=True).start()
    return True


def job_index():
    import argparse
    rag.cmd_index(argparse.Namespace(data=str(DATA), index=str(INDEX), rebuild=False))


def job_lectures(subject: str, lang: str):
    import argparse
    lec = DATA / "lectures"
    exts = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".mov"}
    files = sorted(p for p in lec.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    a = argparse.Namespace(
        out=str(DATA / "lecture_out"), subject=subject, lang=lang, model="large-v3", device="auto",
        compute_type="auto", beam_size=None, threads=0, model_dir=os.environ.get("LECTURE_TOOL_MODEL_DIR"),
        prompt=None, terms=None, keywords=30, llm="ollama", llm_model=None, chunk_min=10, chunk_chars=12000,
        force=False, batch_size=8, normalize=False, no_vad=False, vad_threshold=0.25, no_preemph=False,
        strict_conf=False)
    print(f"Tìm thấy {len(files)} file audio — bài đã xong sẽ được bỏ qua.")
    for i, p in enumerate(files, 1):
        print(f"##### Bài {i}/{len(files)}: {p.name}")
        try:
            lt.process(p, a)
        except Exception as e:
            print(f"[!] Lỗi với {p.name}: {type(e).__name__}: {e}")
    lt._MODEL_CACHE.clear()          # trả VRAM cho Ollama
    job_index()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class H(BaseHTTPRequestHandler):
    server_version = "LectureAssistant/1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, p: Path):
        """Gửi file, hỗ trợ Range để trình duyệt tua audio tới đúng phút."""
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix.lower() == ".m4a":
            ctype = "audio/mp4"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                code = 206
        length = max(0, end - start + 1)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                buf = f.read(min(1 << 16, left))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(buf)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(u.path)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/status":
            loaded = rag.load_index(INDEX)
            docs = {}
            if loaded:
                for c in loaded[1]:
                    key = (rag.subject_of(c.get("file", "")), c["kind"], c["doc"])
                    docs[key] = docs.get(key, 0) + 1
            return self._json({
                "index": bool(loaded), "built_at": loaded[0].get("built_at") if loaded else None,
                "chunks": len(loaded[1]) if loaded else 0,
                "docs": [{"subject": sj, "kind": k, "doc": d, "n": n,
                          "date": rag.fmt_date(rag.doc_date(d)) if rag.doc_date(d) else ""}
                         for (sj, k, d), n in sorted(docs.items())],
                "job": {k: JOB[k] for k in ("running", "name", "ok")} | {"log": JOB["log"][-25:]},
                "data": str(DATA), "models": {"chat": rag.CHAT_MODEL, "verify": rag.VERIFY_MODEL},
            })
        if path.startswith("/media/lectures/"):
            p = safe_under(DATA / "lectures", path[len("/media/lectures/"):])
            return self._file(p) if p else self._send(404, b"not found", "text/plain")
        if path.startswith("/media/data/"):
            p = safe_under(DATA, path[len("/media/data/"):])
            return self._file(p) if p else self._send(404, b"not found", "text/plain")
        if path == "/api/history":
            hp = INDEX / "history.jsonl"
            items = lt.read_jsonl(hp)[-50:] if hp.exists() else []
            return self._json(list(reversed(items)))
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/api/ask":
            return self._ask(body)
        if u.path == "/api/index":
            ok = run_job("Cập nhật chỉ mục", job_index)
            return self._json({"started": ok, "busy": not ok})
        if u.path == "/api/process":
            subject = str(body.get("subject") or "Строительная механика")
            lang = "ru" if body.get("lang") == "ru" else "vi"
            ok = run_job("Phiên âm + tóm tắt bài mới", lambda: job_lectures(subject, lang))
            return self._json({"started": ok, "busy": not ok})
        self._send(404, b"not found", "text/plain")

    def _ask(self, body: dict):
        q = str(body.get("question") or "").strip()
        if not q:
            return self._json({"error": "Nhập câu hỏi trước đã."}, 400)
        if JOB["running"]:
            return self._json({"error": f"Đang bận: {JOB['name']}. Đợi xong rồi hỏi lại."}, 409)
        if not rag.load_index(INDEX):
            return self._json({"error": "Chưa có chỉ mục. Bấm “Cập nhật chỉ mục” trước."}, 409)
        kinds = set(body["only"]) if body.get("only") else None
        subj = body.get("subject")
        subj = None if subj in (None, "__all__") else ("" if subj == "(chung)" else str(subj))
        try:
            with LLM_LOCK:
                res = rag.answer(INDEX, q, k=int(body.get("k") or 8), lang="ru" if body.get("lang") == "ru" else "vi",
                                 kinds=kinds, doc_filter=(body.get("doc") or None), check=not body.get("no_check"),
                                 strict=bool(body.get("strict")), second=bool(body.get("second")), subject=subj)
        except SystemExit as e:
            return self._json({"error": str(e)}, 500)
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            if "Connection" in type(e).__name__ or "connect" in msg.lower():
                msg = "Không kết nối được Ollama. Mở app Ollama rồi thử lại."
            return self._json({"error": f"{type(e).__name__}: {msg}"}, 500)
        hits = res.get("hits", [])
        out = {
            "answer": res.get("answer", ""), "used": res.get("used", []), "bad": sorted(set(res.get("bad", []))),
            "claims": [{"text": c["text"], "verdict": c["verdict"], "note": c.get("note", "")}
                       for c in res.get("claims", [])],
            "sources": [source_payload(c, i) for i, c in enumerate(hits, 1)],
            "second": res.get("second"), "seconds": res.get("seconds"), "queries": res.get("queries"),
        }
        if body.get("save", True):
            with open(INDEX / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": q,
                                    "answer": out["answer"],
                                    "sources": [s["label"] for s in out["sources"] if s["n"] in out["used"]],
                                    "checks": out["claims"]}, ensure_ascii=False) + "\n")
        self._json(out)


# --------------------------------------------------------------------------- #
# Trang web (một file, không tải gì từ Internet)
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trợ lý bài giảng</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--text:#1f1e1c;--muted:#6b6963;--line:#e4e1da;--accent:#2f5fb3;--accent-bg:#e8eefa;
--ok:#2e6b1f;--ok-bg:#eaf3de;--warn:#8a5a00;--warn-bg:#fdf1d8;--bad:#9b2c2c;--bad-bg:#fbe9e9;--radius:10px}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--card:#252523;--text:#ecebe7;--muted:#a8a69f;--line:#3a3936;
--accent:#8fb0ee;--accent-bg:#26324a;--ok:#a8d58a;--ok-bg:#24341c;--warn:#f0c674;--warn-bg:#3d3018;--bad:#f2a3a3;--bad-bg:#432222}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px;max-width:1240px;margin:0 auto;padding:20px}
@media (max-width:900px){.wrap{grid-template-columns:minmax(0,1fr)}}
h1{font-size:20px;font-weight:600;margin:0 0 4px}h2{font-size:15px;font-weight:600;margin:0 0 10px}
.sub{color:var(--muted);font-size:13px;margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
textarea{width:100%;min-height:72px;resize:vertical;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:var(--radius);padding:10px 12px}
textarea:focus,input:focus,select:focus{outline:2px solid var(--accent);outline-offset:0}
input,select{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:var(--radius);padding:6px 8px}
button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:var(--radius);padding:7px 14px;cursor:pointer}
button:hover{background:var(--bg)}button.primary{background:var(--text);color:var(--card);border-color:var(--text)}
button.primary:hover{opacity:.9}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.opts{font-size:13px;color:var(--muted);margin-top:10px}
.opts label{display:inline-flex;gap:5px;align-items:center;margin-right:14px}
.answer{font-size:16px;line-height:1.75;white-space:pre-wrap}
.cite{display:inline-block;min-width:22px;padding:0 5px;margin:0 1px;border-radius:6px;background:var(--accent-bg);color:var(--accent);font-size:12px;font-weight:600;text-align:center;cursor:pointer;text-decoration:none;vertical-align:1px}
.s-unsupported,.s-no_citation,.s-off_topic{background:var(--bad-bg);border-bottom:2px solid var(--bad)}
.s-partial{background:var(--warn-bg);border-bottom:2px dotted var(--warn)}
.check{font-size:13px;margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.check .item{margin:4px 0}.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:999px;margin-right:6px}
.t-ok{background:var(--ok-bg);color:var(--ok)}.t-warn{background:var(--warn-bg);color:var(--warn)}.t-bad{background:var(--bad-bg);color:var(--bad)}
.src{border:1px solid var(--line);border-radius:var(--radius);padding:10px 12px;margin:8px 0;font-size:14px}
.src.hl{outline:2px solid var(--accent)}.src.unused{opacity:.65}
.src .h{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px}
.src .lbl{font-weight:600}.src .txt{color:var(--muted);font-size:13px;max-height:4.8em;overflow:hidden;cursor:pointer}
.src .txt.open{max-height:none}.kind{font-size:11px;padding:1px 7px;border-radius:999px;background:var(--bg);border:1px solid var(--line);color:var(--muted)}
.player{position:sticky;top:0;z-index:5;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-bottom:16px;display:none}
.player audio{width:100%}.player .now{font-size:13px;color:var(--muted);margin-bottom:4px}
.err{color:var(--bad);font-size:14px;margin-top:8px}.muted{color:var(--muted);font-size:13px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.doclist{font-size:13px;max-height:260px;overflow:auto}.doclist div{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px dashed var(--line)}
.log{font:12px/1.5 ui-monospace,Consolas,monospace;background:var(--bg);border-radius:8px;padding:8px;max-height:200px;overflow:auto;white-space:pre-wrap;margin-top:8px}
.hist div{font-size:13px;padding:5px 0;border-bottom:1px dashed var(--line);cursor:pointer}.hist div:hover{color:var(--accent)}
.second{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;font-size:14px;white-space:pre-wrap}
</style></head><body>
<div class="wrap">
<main>
  <h1>Trợ lý bài giảng</h1>
  <p class="sub">Hỏi bằng tiếng Việt hoặc tiếng Nga. Câu trả lời chỉ dựa trên bài giảng, bài báo và ghi chú của bạn — bấm vào số [n] để xem nguồn, bấm “Nghe” để phát đúng đoạn thầy nói.</p>
  <div class="player" id="player"><div class="now" id="now"></div><audio id="audio" controls preload="none"></audio></div>
  <div class="card">
    <textarea id="q" placeholder="Điều kiện biên tự nhiên là gì?"></textarea>
    <div class="row" style="margin-top:10px">
      <button class="primary" id="ask">Hỏi</button>
      <span id="busy" class="muted"></span>
    </div>
    <div class="opts">
      <label>Môn <select id="subj"><option value="__all__">tất cả các môn</option></select></label>
      <label>Chỉ tìm trong <select id="doc"><option value="">tất cả tài liệu</option></select></label>
      <label>Trả lời bằng <select id="lang"><option value="vi">tiếng Việt</option><option value="ru">tiếng Nga</option></select></label>
      <label><input type="checkbox" id="check" checked> kiểm tra chéo</label>
      <label><input type="checkbox" id="strict"> bỏ câu không có nguồn</label>
      <label><input type="checkbox" id="second"> ý kiến thứ hai</label>
    </div>
    <div class="err" id="err"></div>
  </div>
  <div id="result"></div>
</main>
<aside>
  <div class="card">
    <h2>Tài liệu</h2>
    <div id="docinfo" class="muted">Đang tải…</div>
    <div class="doclist" id="docs"></div>
    <div class="row" style="margin-top:12px">
      <button id="btnIndex">Cập nhật chỉ mục</button>
      <button id="btnProc">Xử lý bài giảng mới</button>
    </div>
    <div class="muted" style="margin-top:6px">Mỗi môn một thư mục con: <code>data\lectures\&lt;môn&gt;</code>, <code>data\papers\&lt;môn&gt;</code>, <code>data\notes\&lt;môn&gt;</code>.</div>
    <div class="log" id="log" style="display:none"></div>
  </div>
  <div class="card"><h2>Câu hỏi gần đây</h2><div class="hist" id="hist"><span class="muted">Chưa có.</span></div></div>
</aside>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=t=>{t=Math.max(0,Math.floor(t||0));const h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;return[h,m,s].map(x=>String(x).padStart(2,"0")).join(":")};
const KIND={lecture:"bài giảng",summary:"tóm tắt máy",pdf:"PDF",table:"bảng",note:"ghi chú"};
const VERD={supported:["t-ok","có nguồn xác nhận"],partial:["t-warn","đúng một phần"],unsupported:["t-bad","nguồn không nói điều này"],
 no_citation:["t-bad","không trích nguồn"],off_topic:["t-bad","đúng nguồn nhưng lạc đề"],not_found:["t-ok","báo không tìm thấy"],unchecked:["t-warn","chưa kiểm được"]};
let lastJobRunning=false;

function play(url,start,label){
  const a=$("audio"),p=$("player");p.style.display="block";
  $("now").textContent="Đang phát: "+label;
  const go=()=>{try{a.currentTime=Math.max(0,start-3)}catch(e){};a.play().catch(()=>{})};
  if(a.dataset.src!==url){a.src=url;a.dataset.src=url;a.addEventListener("loadedmetadata",go,{once:true});a.load()}else go();
  window.scrollTo({top:0,behavior:"smooth"});
}

function md(h){
  return h.replace(/\*\*([^*\n]+)\*\*/g,"<b>$1</b>").replace(/\$\$([^$]+)\$\$/g,"<i>$1</i>")
          .replace(/\$([^$\n]{1,80})\$/g,"<i>$1</i>").replace(/\\(?:text|mathrm)\{([^}]*)\}/g,"$1")
          .replace(/^[-*] /gm,"• ");
}
function render(r){
  let html=esc((r.answer||"").replace(/\s*⚠/g,""));
  for(const c of (r.claims||[])){
    if(["unsupported","no_citation","off_topic","partial"].includes(c.verdict)){
      const t=esc(c.text);const i=html.indexOf(t);
      if(i>=0)html=html.slice(0,i)+`<span class="s-${c.verdict}" title="${esc(VERD[c.verdict][1]+(c.note?": "+c.note:""))}">`+t+"</span>"+html.slice(i+t.length);
    }
  }
  html=html.replace(/\[(\d+)\]/g,(m,n)=>`<a class="cite" data-n="${n}">${n}</a>`);
  html=md(html);
  let s=`<div class="card"><div class="answer">${html}</div>`;
  if(r.claims&&r.claims.length){
    const ok=r.claims.filter(c=>c.verdict==="supported").length;
    s+=`<div class="check"><b>Kiểm tra chéo:</b> ${ok}/${r.claims.length} câu được nguồn xác nhận`;
    for(const c of r.claims){if(c.verdict==="supported"||c.verdict==="not_found")continue;const v=VERD[c.verdict]||["t-warn",c.verdict];
      s+=`<div class="item"><span class="tag ${v[0]}">${v[1]}</span>${esc(c.text.replace(/\s*\[\d+\]/g,"").slice(0,140))}${c.note?` <span class="muted">— ${esc(c.note)}</span>`:""}</div>`}
    s+="</div>";
  }
  if(r.bad&&r.bad.length)s+=`<div class="muted" style="margin-top:6px">Đã gỡ trích dẫn không tồn tại: ${r.bad.join(", ")}</div>`;
  if(r.second)s+=`<div class="second"><b>Ý kiến thứ hai</b> (model khác, cùng nguồn):\n${md(esc(r.second)).replace(/\[(\d+)\]/g,(m,n)=>`<a class="cite" data-n="${n}">${n}</a>`)}</div>`;
  s+=`<div class="muted" style="margin-top:10px">${r.seconds} s</div></div><div class="card"><h2>Nguồn</h2>`;
  const used=new Set(r.used||[]);
  const order=[...r.sources].sort((a,b)=>(used.has(b.n)-used.has(a.n))||a.n-b.n);
  for(const x of order){
    let act="";
    if(x.audio&&x.start!=null)act+=`<button data-play="${esc(x.audio)}" data-start="${x.start}" data-label="${esc(x.label)}">Nghe từ ${fmt(x.start)}</button>`;
    if(x.open)act+=`<a href="${esc(x.open)}" target="_blank"><button>${x.kind==="pdf"?"Mở trang":"Mở file"}</button></a>`;
    s+=`<div class="src ${used.has(x.n)?"":"unused"}" id="src${x.n}"><div class="h"><a class="cite" data-n="${x.n}">${x.n}</a>
      <span class="lbl">${esc(x.label)}</span><span class="kind">${KIND[x.kind]||x.kind}</span>${x.subject?`<span class="kind">${esc(x.subject)}</span>`:""}${x.date?`<span class="muted">${esc(x.date)}</span>`:""}
      ${used.has(x.n)?"":`<span class="muted">không dùng</span>`}<span style="flex:1"></span>${act}</div>
      <div class="txt" title="Bấm để xem đủ">${esc(x.text)}</div></div>`;
  }
  s+="</div>";$("result").innerHTML=s;
}

document.addEventListener("click",e=>{
  const c=e.target.closest(".cite");
  if(c){const el=$("src"+c.dataset.n);if(el){document.querySelectorAll(".src.hl").forEach(x=>x.classList.remove("hl"));el.classList.add("hl");el.scrollIntoView({behavior:"smooth",block:"center"})}return}
  const b=e.target.closest("[data-play]");if(b){play(b.dataset.play,parseFloat(b.dataset.start),b.dataset.label);return}
  const t=e.target.closest(".src .txt");if(t){t.classList.toggle("open");return}
  const h=e.target.closest(".hist [data-q]");if(h){$("q").value=h.dataset.q;$("q").focus()}
});

async function ask(){
  const q=$("q").value.trim();$("err").textContent="";
  if(!q){$("err").textContent="Nhập câu hỏi trước đã.";return}
  $("ask").disabled=true;const t0=Date.now();
  const tick=setInterval(()=>{$("busy").innerHTML=`<span class="spin"></span> Đang tìm và trả lời… ${Math.round((Date.now()-t0)/1000)} s`},500);
  try{
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,doc:$("doc").value,subject:$("subj").value,
      lang:$("lang").value,no_check:!$("check").checked,strict:$("strict").checked,second:$("second").checked})});
    const j=await r.json();if(!r.ok||j.error){$("err").textContent=j.error||("Lỗi "+r.status)}else{render(j);loadHist()}
  }catch(e){$("err").textContent="Không kết nối được tới app. Cửa sổ đen (app.py) còn mở không?"}
  finally{clearInterval(tick);$("busy").textContent="";$("ask").disabled=false}
}
$("ask").onclick=ask;$("q").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.ctrlKey||!e.shiftKey)){e.preventDefault();ask()}});

async function status(){
  try{
    const j=await (await fetch("/api/status")).json();
    $("docinfo").textContent=j.index?`${j.chunks} đoạn · cập nhật ${j.built_at}`:"Chưa có chỉ mục — bấm “Cập nhật chỉ mục”.";
    const seen=new Set(),sel=$("doc"),cur=sel.value,ss=$("subj"),scur=ss.value;
    const subjects=[...new Set(j.docs.map(d=>d.subject||"(chung)"))];
    ss.innerHTML='<option value="__all__">tất cả các môn</option>'+subjects.map(x=>`<option>${esc(x)}</option>`).join("");
    ss.value=subjects.includes(scur)?scur:"__all__";
    const inSubj=d=>ss.value==="__all__"||(d.subject||"(chung)")===ss.value;
    let lastS=null;
    $("docs").innerHTML=j.docs.filter(d=>d.kind!=="summary").map(d=>{const sj=d.subject||"(chung)";const head=sj!==lastS?`<div style="border:0;font-weight:600;padding-top:8px;color:var(--text)">${esc(sj)}</div>`:"";lastS=sj;
      return head+`<div><span>${esc(d.doc)}</span><span class="muted">${KIND[d.kind]||d.kind}${d.date?" · "+esc(d.date):""}</span></div>`}).join("");
    sel.innerHTML='<option value="">tất cả tài liệu</option>'+j.docs.filter(d=>inSubj(d)&&!seen.has(d.doc)&&seen.add(d.doc)).map(d=>`<option>${esc(d.doc)}</option>`).join("");
    sel.value=[...sel.options].some(o=>o.value===cur)?cur:"";
    const L=$("log"),run=j.job.running;
    if(run||j.job.log.length){L.style.display="block";L.textContent=(run?"⏳ ":(j.job.ok?"✓ ":"✗ "))+j.job.name+"\n"+j.job.log.join("\n");L.scrollTop=L.scrollHeight}
    $("btnIndex").disabled=$("btnProc").disabled=run;
    if(lastJobRunning&&!run)loadHist();lastJobRunning=run;
  }catch(e){$("docinfo").textContent="Mất kết nối với app."}
}
async function job(url,body){await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});status()}
$("btnIndex").onclick=()=>job("/api/index");
$("subj").onchange=()=>status();
$("btnProc").onclick=()=>{const s=prompt("Bài nằm trong thư mục môn sẽ tự lấy tên môn theo thư mục.\nTên môn (tiếng Nga) cho các bài nằm thẳng trong data\\lectures:","Строительная механика");if(s!==null)job("/api/process",{subject:s,lang:$("lang").value})};
async function loadHist(){try{const h=await (await fetch("/api/history")).json();
  $("hist").innerHTML=h.length?h.slice(0,15).map(x=>`<div data-q="${esc(x.question)}" title="${esc(x.time)}">${esc(x.question)}</div>`).join(""):'<span class="muted">Chưa có.</span>'}catch(e){}}
status();loadHist();setInterval(status,3000);
</script></body></html>
"""


def main():
    srv = ThreadingHTTPServer((HOST, PORT), H)
    url = f"http://{HOST}:{PORT}/"
    print(f"Trợ lý bài giảng đang chạy: {url}")
    print(f"Dữ liệu: {DATA}   |   Chỉ mục: {INDEX}")
    print("Đóng cửa sổ này (hoặc Ctrl+C) để tắt.")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
