# Build the NiveshRL desktop app: icon -> PyInstaller one-folder build -> seed data -> (optional) Setup.exe.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1            # app folder only
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Installer # also NiveshRL-Setup.exe (needs Inno Setup 6)
#
# Output: build\dist\NiveshRL\NiveshRL.exe (+ _internal\, seed\) and build\installer\NiveshRL-Setup.exe
param([switch]$Installer)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location $Root

Write-Host "== icon"
& $Py -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PySide6.QtWidgets import QApplication; a=QApplication([]); from niveshrl.desktop.tray import make_icon; ok=make_icon().pixmap(256,256).toImage().save(r'packaging\niveshrl.ico'); print('icon', ok)"

Write-Host "== C++ core"
& $Py -c "import niveshrl_core" 2>$null
if ($LASTEXITCODE -ne 0) { & $Py -m pip install ./cpp }

Write-Host "== PyInstaller"
& $Py -m PyInstaller packaging\niveshrl.spec --noconfirm --distpath build\dist --workpath build\work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "== seed data"
$Seed = "build\dist\NiveshRL\seed"
if (Test-Path $Seed) { Remove-Item -Recurse -Force $Seed }
New-Item -ItemType Directory -Force $Seed | Out-Null
foreach ($d in @("configs", "report\results")) { Copy-Item -Recurse $d (Join-Path $Seed $d) }
New-Item -ItemType Directory -Force (Join-Path $Seed "data") | Out-Null
foreach ($f in @("ind_nifty200list.csv", "nifty200_panel.parquet", "nifty200_context.parquet", "stocks.parquet", "context.parquet")) {
    if (Test-Path "data\$f") { Copy-Item "data\$f" (Join-Path $Seed "data\$f") }
}
foreach ($d in @("predictions", "daily", "constituents", "models")) { if (Test-Path "data\$d") { Copy-Item -Recurse "data\$d" (Join-Path $Seed "data\$d") } }
# intraday agent: the liquid universe and the replay's learning (warm start + the Replay tab); not the 5-minute bars
New-Item -ItemType Directory -Force (Join-Path $Seed "data\intraday\replay") | Out-Null
if (Test-Path "data\intraday\universe.csv") { Copy-Item "data\intraday\universe.csv" (Join-Path $Seed "data\intraday\universe.csv") }
New-Item -ItemType Directory -Force (Join-Path $Seed "data\intraday\dl") | Out-Null      # v2: the trained TCN
if (Test-Path "data\intraday\dl\tcn_latest.pkl") { Copy-Item "data\intraday\dl\tcn_latest.pkl" (Join-Path $Seed "data\intraday\dl\tcn_latest.pkl") }
foreach ($f in @("state.json", "trades.csv", "bandit.json", "scorer.pkl", "shadow.parquet")) {
    if (Test-Path "data\intraday\replay\$f") { Copy-Item "data\intraday\replay\$f" (Join-Path $Seed "data\intraday\replay\$f") }
}
# RL checkpoints used by the investor plan (small); skip the 32 MB SB3 runs and logs
foreach ($r in Get-ChildItem runs -Directory) {
    if ((Test-Path (Join-Path $r.FullName "best.pt")) -and (Test-Path (Join-Path $r.FullName "args.json")) -and -not $r.Name.StartsWith("sb3")) {
        Copy-Item -Recurse $r.FullName (Join-Path $Seed "runs\$($r.Name)")
    }
}
$size = (Get-ChildItem -Recurse build\dist\NiveshRL | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("App folder: build\dist\NiveshRL ({0:N0} MB)" -f $size)

if ($Installer) {
    $Iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
              "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Iscc) { throw "Inno Setup 6 not found. Install it (https://jrsoftware.org/isdl.php) and re-run with -Installer." }
    & $Iscc packaging\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
    Write-Host "Installer: build\installer\NiveshRL-Setup.exe"
}
