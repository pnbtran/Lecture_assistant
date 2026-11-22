# Trợ lý bài giảng (Lecture Assistant) — ghi âm bài giảng tiếng Nga → văn bản, tóm tắt, hỏi đáp có trích dẫn

Mọi thứ chạy **trên laptop của bạn** (Whisper + Ollama), lưu trên ổ D, không gửi dữ liệu ra ngoài.

```
audio bài giảng ──► lecture_tool.py ──► bản chép + tóm tắt + từ khóa (lecture_out)
PDF / Excel / ghi chú ───────────────┐
                                     ├─► rag.py index ──► chỉ mục (index)
bản chép + tóm tắt ──────────────────┘
câu hỏi ──► app.py (web) hoặc rag.py ask ──► câu trả lời có nguồn [n] + kiểm tra chéo
```

---

## 1. Dùng hằng ngày

### Mở trợ lý
Bấm đúp **`D:\research_assistant\lecture_assistant\start_assistant.bat`** (nên tạo lối tắt ra Desktop).
Trình duyệt mở `http://127.0.0.1:8765`. Đóng cửa sổ đen để tắt. Trang chỉ mở được trên chính máy này.

### Chia theo môn
Mỗi môn **một thư mục con** (tên tùy bạn, nên ngắn), dùng cùng tên ở cả ba chỗ:
```
data\lectures\стромех\16-10-лек6-стромех.m4a     data\papers\стромех\Ivanov_2021.pdf
data\lectures\жбк\...                            data\notes\стромех\Tran_tong_quan.xlsx
```
- Kết quả tự vào `data\lecture_out\<môn>\<bài>\`; trên trang web chọn **Môn** để chỉ tìm trong môn đó.
- Tên môn đầy đủ (tiếng Nga) dùng cho bản tóm tắt: khai báo trong `data\subjects.json`, vd.
  `{"стромех": "Строительная механика", "жбк": "Железобетонные конструкции"}`. Không khai báo thì dùng chính tên thư mục.
- File để thẳng trong `data\lectures` (không có thư mục môn) thuộc nhóm **(chung)**.
- Thư mục tên `bench`, `test_…` hoặc bắt đầu bằng `_` không được đưa vào chỉ mục (dùng cho thử nghiệm / lưu trữ).

### Thêm bài giảng mới
1. Tải file ghi âm từ Google Drive, chép vào `D:\research_assistant\lecture_assistant\data\lectures\<môn>\`.
   Đặt tên **có ngày**: `23-10-лек8-стромех.m4a`, `23-10-механика-пр5.m4a` — trợ lý dùng ngày để hiểu “buổi tới”, “bài gần nhất”.
2. Trên trang web bấm **“Xử lý bài giảng mới”** → nhập tên môn (tiếng Nga). Mỗi bài 1h30 mất ~7 phút
   (phiên âm ~1,5 phút + tóm tắt ~5 phút), xong tự cập nhật chỉ mục.

### Thêm tài liệu môn học (PDF, Excel, ghi chú)
- PDF bài báo → `data\papers\`  ·  Excel/CSV (vd. `Tran_tong_quan.xlsx`) → `data\notes\`  ·  ghi chú `.md/.txt` → `data\notes\`
- Bấm **“Cập nhật chỉ mục”**. Chỉ tài liệu mới/đã sửa được xử lý lại.
- PDF dạng scan (ảnh, không có chữ) sẽ bị báo “trang không có chữ” — hiện chưa có OCR.

### Đọc câu trả lời
- Số **[n]** → bấm để xem đoạn nguồn. Nút **“Nghe từ hh:mm:ss”** phát audio từ ~3 giây trước chỗ thầy nói; **“Mở trang”** mở PDF đúng trang.
- **Kiểm tra chéo** (Gemma đối chiếu từng câu với đúng nguồn được trích):
  - không tô màu = có nguồn xác nhận
  - nền vàng = chỉ đúng một phần
  - nền đỏ = nguồn không nói điều này / không trích nguồn / đúng nguồn nhưng nói về việc khác
- Tùy chọn: *chỉ tìm trong* một tài liệu · trả lời tiếng Việt/tiếng Nga · *bỏ câu không có nguồn* · *ý kiến thứ hai* (Gemma trả lời độc lập để so).
- Mỗi câu hỏi mất ~30–40 s (bỏ kiểm tra chéo: ~20 s). Lịch sử lưu ở `index\history.jsonl`.

> **Kiểm tra chéo không bắt được mọi lỗi.** Nó bắt tốt câu *nguồn không nói*, nhưng có thể bỏ sót câu *ghép nhầm khái niệm giữa hai bài*.
> Với công thức, số liệu, hạn nộp, yêu cầu kiểm tra: luôn bấm “Nghe” để xác nhận.

---

### Phiên âm một file / một thư mục
Kéo thả file ghi âm (hoặc cả thư mục môn) lên **`phien_am.bat`** → phiên âm + tóm tắt tiếng Việt.

## 2. Dòng lệnh (PowerShell, trong `D:\research_assistant\lecture_assistant\app`)

`$env:RA_PYTHON` = Python của dự án (`runtime\python_env\Scripts\python.exe`), đặt sẵn khi sắp xếp thư mục.

Mỗi dòng là **một** lệnh — dán và Enter từng dòng.

```powershell
# phiên âm + tóm tắt mọi bài trong thư mục (bài đã xong tự bỏ qua)
& $env:RA_PYTHON lecture_tool.py D:\research_assistant\lecture_assistant\data\lectures --subject "Строительная механика" --lang vi --llm ollama

# chỉ phiên âm, chưa tóm tắt
& $env:RA_PYTHON lecture_tool.py D:\research_assistant\lecture_assistant\data\lectures --llm none

# chỉ mục + hỏi đáp
& $env:RA_PYTHON rag.py index
& $env:RA_PYTHON rag.py ask "Điều kiện biên tự nhiên là gì?" -v
& $env:RA_PYTHON rag.py ask "Как проверить равновесие узла?" --doc пр3 --lang ru
& $env:RA_PYTHON rag.py chat            # hỏi liên tục; "@лек7 câu hỏi" = chỉ tìm trong bài 7; q = thoát
& $env:RA_PYTHON rag.py sources         # chỉ mục đang có gì

# giao diện web (giống start_assistant.bat)
& $env:RA_PYTHON app.py
```

| Tùy chọn | Ý nghĩa |
|---|---|
| `lecture_tool.py … --force` | làm lại từ đầu cả phiên âm lẫn tóm tắt |
| `lecture_tool.py … --lang ru` | tóm tắt bằng tiếng Nga đơn giản |
| `rag.py ask … --strict` | tự bỏ các câu nguồn không xác nhận |
| `rag.py ask … --second` | thêm câu trả lời độc lập của Gemma |
| `rag.py ask … --no-check` | bỏ kiểm tra chéo (nhanh hơn) |
| `rag.py ask … --only pdf,table` | chỉ tìm trong PDF và bảng |
| `rag.py ask … --mon стромех` | chỉ tìm trong một môn |
| `rag.py index --rebuild` | xây lại toàn bộ chỉ mục |

Chỉ tóm tắt lại một bài (không phiên âm lại): xóa `<bài>.summary.json` trong `data\lecture_out\<bài>\` rồi chạy lại lệnh phiên âm + tóm tắt.

---

## 3. Thư mục trên ổ D

```
D:\research_assistant\
  runtime\                DÙNG CHUNG với trợ lý luận văn, không lên GitHub
    python_env\           môi trường Python (faster-whisper, pymorphy3, …)
    ollama\  ollama_models\   chương trình Ollama + model (qwen3.5:9b, gemma3:4b, bge-m3)
    whisper_models\       model Whisper (large-v3, large-v3-turbo)
    hf_cache\  pip_cache\  downloads\
  lecture_assistant\      ← DỰ ÁN NÀY (repo GitHub)
    start_assistant.bat  phien_am.bat  setup.ps1  requirements.txt  README.md  .gitignore
    app\                  code: lecture_tool.py, rag.py, app.py, bench.py
    data\lectures\<môn>\   audio gốc                 (không lên GitHub)
    data\lecture_out\<môn>\ kết quả: .report.md (đọc cái này), .transcript.txt, .srt, .summary.json
    data\papers\  data\notes\  PDF, Excel, ghi chú
    data\subjects.json     tên thư mục môn -> tên môn tiếng Nga (tùy chọn)
    index\                chỉ mục hỏi đáp + history.jsonl
  thesis_assistant\       trợ lý luận văn (repo riêng)
```

**Sao lưu**: `lecture_assistant\data\` và `lecture_assistant\index\` là dữ liệu của bạn. `runtime\` tải lại được (`setup.ps1`).

---

## 4. Cấu hình đã chọn (và vì sao)

| Thành phần | Cấu hình | Lý do (đo trên bài 16-10-лек6) |
|---|---|---|
| Phiên âm | Whisper `large-v3`, float16 trên RTX 5050, chạy **theo lô** | Ít mất chữ nhất: ~424 ký tự/phút so với ~210 của cấu hình ban đầu |
| Tiền xử lý | lọc nhấn cao tần | Ghi âm từ chỗ ngồi bị “đục”, mất phụ âm |
| Gợi ý thuật ngữ cho Whisper | **tắt** | Whisper hay chép lại nguyên câu gợi ý vào bản chép (`--prompt default` để bật lại) |
| VAD | ngưỡng 0,25, giữ đoạn kém tự tin | Không bỏ lời nói nhỏ |
| Tóm tắt | `qwen3.5:9b`, tắt “suy nghĩ”, ngữ cảnh 8k | Vừa 8 GB VRAM, ~25–35 s mỗi phần 10 phút |
| Tìm kiếm | `bge-m3` (nghĩa, hiểu cả Việt–Nga) + BM25 (từ khoá) | Câu hỏi tiếng Việt tìm được lời giảng tiếng Nga |
| Kiểm tra chéo | `gemma3:4b` (khác họ với Qwen) | Không “tự chấm bài mình” |

`bench.py` dùng để so sánh lại các cấu hình Whisper trên một bài mới:
`& $env:RA_PYTHON bench.py "D:\…\bai.m4a" --with-large`.

---

## 5. Ghi âm tốt hơn (quan trọng hơn mọi tinh chỉnh phần mềm)

- Điện thoại đặt trên bàn, **đầu có micro (cạnh dưới) hướng về bảng**, kê cao lên hộp bút; xa laptop; ngồi gần hơn khi có thể.
- App ghi âm Samsung: chất lượng **Cao 256 kbps 48 kHz**; bật **chặn cuộc gọi khi đang ghi âm**.
- **Chụp ảnh bảng** sau mỗi lần thầy viết xong — công thức trên bảng không có trong audio.
- Chuyển file qua Google Drive / cáp (không gửi qua Zalo/Telegram dạng tin nhắn thoại — bị nén).

---

## 6. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| “Không kết nối được Ollama” | Chạy lại `start_assistant.bat` (tự bật `runtime\ollama\ollama.exe serve`) |
| PowerShell đứng yên, tiêu đề có chữ **Select** | Bấm vào cửa sổ, nhấn **Esc**. Đừng bấm chuột vào cửa sổ khi đang chạy |
| `invalid choice: 'chat.\.venv…'` | Hai lệnh bị dán dính — dán từng dòng một |
| Bản phiên âm báo “khá THƯA” | Nghe lại đoạn đó: nếu là giờ làm bài/nghỉ thì bình thường; nếu thầy có giảng → kiểm tra cách đặt điện thoại, hoặc thử `--vad-threshold 0.15 --force` |
| Tóm tắt chậm bất thường (> 1 phút/phần) | Đóng YouTube/game để giải phóng GPU; `ollama ps` phải ghi `100% GPU` |
| Sửa code xong mà web không đổi | Đóng cửa sổ đen, mở lại `start_assistant.bat` |
| Máy ngủ giữa chừng | Settings → Power: “Make my device sleep after” = Never khi cắm sạc |

---

## 7. Giới hạn cần biết

- Bản chép là tự động: sai ở thuật ngữ, số liệu, công thức (vd. “огранжа” = Лагранжа). Tóm tắt và câu trả lời có thể kế thừa lỗi đó.
- Tóm tắt có thể **suy diễn sai “thông báo của giảng viên”** (bài kiểm tra đang làm trong tiết ≠ bài kiểm tra buổi sau) — đối chiếu bằng mốc thời gian.
- Model 9B có thể **ghép nhầm khái niệm giữa các bài**; kiểm tra chéo không phải lúc nào cũng bắt được.
- Không tách người nói (thầy / sinh viên). Không đọc được chữ trên bảng.

## 8. Lộ trình tiếp theo

1. Dùng thật 1–2 tuần, ghi lại câu trả lời sai (lịch sử tự lưu).
2. Thêm PDF bài báo và `Tran_tong_quan.xlsx`; OCR tiếng Nga nếu có PDF scan.
3. `eval.py`: bộ ~20 câu hỏi có đáp án + nguồn đúng → chấm tự động mỗi lần thay đổi.
4. Danh sách thuật ngữ theo môn để sửa lỗi nhận dạng; Qwen kiểm tra lần hai cho câu quan trọng; thử model lớn hơn khi nâng RAM lên 32 GB.
