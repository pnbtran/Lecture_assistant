# setup.ps1 - dung moi truong cho tro ly bai giang (va runtime dung chung voi tro ly luan van), MOI THU TREN O D:
# Bo cuc:  <Root>\runtime\ (Python, Ollama, model)  +  <Root>\lecture_assistant\ (repo nay)
# Chay (khong can Administrator):
#     powershell -ExecutionPolicy Bypass -File .\setup_G0.ps1
# Chay lai nhieu lan van an toan (buoc nao xong roi se bo qua).
param(
    [string]$Root = "D:\research_assistant",
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Step($m) { Write-Host ""; Write-Host "=== $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  !   $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 1. Thu muc
Step "1/6  Tao thu muc tren $Root"
$drive = Split-Path $Root -Qualifier
if (-not (Test-Path $drive)) { throw "Khong thay o $drive" }
$RT = Join-Path $Root "runtime"
$Proj = Join-Path $Root "lecture_assistant"
foreach ($d in @("runtime\ollama_models", "runtime\whisper_models", "runtime\hf_cache", "runtime\pip_cache",
                 "runtime\downloads", "lecture_assistant\data\lectures", "lecture_assistant\data\lecture_out",
                 "lecture_assistant\data\papers", "lecture_assistant\data\notes", "lecture_assistant\index")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}
Ok "Da co runtime\... va lecture_assistant\data\..."

# ---------------------------------------------------------------- 2. Bien moi truong
Step "2/6  Dat bien moi truong (User) de moi thu nam tren D:"
$vars = [ordered]@{
    RA_RUNTIME               = $RT
    RA_PYTHON                = (Join-Path $RT "python_env\Scripts\python.exe")
    RA_OLLAMA                = (Join-Path $RT "ollama\ollama.exe")
    OLLAMA_MODELS            = (Join-Path $RT "ollama_models")
    HF_HOME                  = (Join-Path $RT "hf_cache")
    PIP_CACHE_DIR            = (Join-Path $RT "pip_cache")
    LECTURE_TOOL_MODEL_DIR   = (Join-Path $RT "whisper_models")
    LECTURE_TOOL_OUT         = (Join-Path $Proj "data\lecture_out")
    OLLAMA_MAX_LOADED_MODELS = "1"      # chi nap 1 model mot luc (VRAM 8 GB)
    OLLAMA_KEEP_ALIVE        = "3m"     # nha model khoi VRAM sau 3 phut khong dung
    OLLAMA_FLASH_ATTENTION   = "1"
    OLLAMA_KV_CACHE_TYPE     = "q8_0"   # giam mot nua bo nho cho ngu canh
    OLLAMA_CONTEXT_LENGTH    = "8192"
}
foreach ($k in $vars.Keys) {
    [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")
    Set-Item -Path "Env:$k" -Value $vars[$k]
    Write-Host ("      {0} = {1}" -f $k, $vars[$k])
}
Ok "Da luu. (Cua so PowerShell/terminal MOI se tu nhan bien nay.)"

# ---------------------------------------------------------------- 3. Ollama
Step "3/6  Ollama"
$ollamaDir = Join-Path $RT "ollama"
$existing = Get-Command ollama -ErrorAction SilentlyContinue
if ($existing) {
    $ollama = $existing.Source
    Ok "Da co Ollama: $ollama"
} else {
    $setup = Join-Path $RT "downloads\OllamaSetup.exe"
    if (-not (Test-Path $setup)) {
        Write-Host "      Tai OllamaSetup.exe ..."
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $setup
    }
    Write-Host "      Cai vao $ollamaDir (cua so cai dat se hien ra, cu bam Install) ..."
    Start-Process -FilePath $setup -ArgumentList "/DIR=`"$ollamaDir`"" -Wait
    $ollama = Join-Path $ollamaDir "ollama.exe"
    if (-not (Test-Path $ollama)) { throw "Cai Ollama khong thanh cong: khong thay $ollama" }
    Ok "Da cai Ollama vao $ollamaDir"
}

# Khoi dong lai server trong phien nay de chac chan nhan bien moi truong vua dat
Get-Process | Where-Object { $_.ProcessName -like "ollama*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod "http://127.0.0.1:11434/api/version" | Out-Null; $ready = $true; break }
    catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { throw "Ollama server khong len. Thu mo ung dung Ollama tu Start menu roi chay lai script." }
$ver = (Invoke-RestMethod "http://127.0.0.1:11434/api/version").version
Ok "Ollama dang chay, phien ban $ver (model se luu o $($vars.OLLAMA_MODELS))"

# ---------------------------------------------------------------- 4. Keo model
if ($SkipModels) {
    Step "4/6  Bo qua keo model (-SkipModels)"
} else {
    Step "4/6  Keo model (tong khoang 10-11 GB, lan dau se lau)"
    foreach ($m in @("bge-m3", "qwen3.5:9b", "gemma3:4b")) {
        Write-Host "      ollama pull $m"
        & $ollama pull $m
        if ($LASTEXITCODE -ne 0) { Warn "Khong keo duoc $m (xem loi o tren)" } else { Ok "Da co $m" }
    }
}

# ---------------------------------------------------------------- 5. Do toc do
function Test-Model([string]$name, [hashtable]$extra) {
    $body = @{ model = $name; stream = $false
               prompt = "Кратко, в двух предложениях: что такое момент инерции сечения?" }
    if ($extra) { foreach ($k in $extra.Keys) { $body[$k] = $extra[$k] } }
    $json = $body | ConvertTo-Json -Depth 5
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post `
             -ContentType "application/json; charset=utf-8" `
             -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 900
        $tps = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)
        Ok ("{0}: {1} token/giay" -f $name, $tps)
        Write-Host ("      Tra loi mau: " + ($r.response.Trim() -replace "\s+", " ")) -ForegroundColor DarkGray
    } catch {
        Warn "Thu $name loi: $($_.Exception.Message)"
    }
}
if (-not $SkipModels) {
    Step "5/6  Do toc do va xem model co nam tren GPU khong"
    Test-Model "qwen3.5:9b" @{ think = $false }
    & $ollama ps
    Write-Host "      -> Cot PROCESSOR nen ghi '100% GPU'. Neu ghi 'xx%/yy% CPU/GPU' nghia la model khong vua VRAM." -ForegroundColor DarkGray
    Test-Model "gemma3:4b" $null
    & $ollama ps
}

# ---------------------------------------------------------------- 6. Python + lecture_tool
Step "6/6  Python (runtime\python_env) + thu vien"
$tool = Join-Path $Proj "app"
$venv = Join-Path $RT "python_env"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not (Test-Path (Join-Path $tool "lecture_tool.py"))) {
    Warn "Chua thay app\lecture_tool.py trong $Proj."
} elseif (-not $py) {
    Warn "Chua thay Python launcher 'py'. Cai Python 3.11-3.13 (python.org, tick 'py launcher') roi chay lai."
} else {
    Push-Location $tool
    try {
        if (-not (Test-Path $venv)) { & py -3 -m venv $venv }
        $venvPy = Join-Path $venv "Scripts\python.exe"
        & $venvPy -m pip install --upgrade pip
        & $venvPy -m pip install -r (Join-Path $Proj "requirements.txt")
        Write-Host ""
        Write-Host "      Kiem tra Whisper tren GPU (lan dau se tai model 'tiny' ~75 MB) ..."
        & $venvPy lecture_tool.py --check-gpu
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "XONG. Mo tro ly: $Proj\start_assistant.bat" -ForegroundColor Cyan
