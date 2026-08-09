# start-seizureguard.ps1 - ensure one hidden monitor process per camera.
# Idempotent: safe to run from logon shortcut and the hourly watchdog task.
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

# Pose gate runs in its CUDA venv when available; monitor fail-opens without it.
$posePython = Join-Path $repo ".venv-pose\Scripts\python.exe"
if (Test-Path $posePython) { $env:SEIZUREGUARD_POSE_PYTHON = $posePython }

# Operational logs only (never event footage): drop logs older than 14 days.
Get-ChildItem $logDir -Filter "*.log" | Where-Object {
    $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Force -Confirm:$false -ErrorAction SilentlyContinue
Get-ChildItem $logDir -Filter "*.err" | Where-Object {
    $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Force -Confirm:$false -ErrorAction SilentlyContinue

$cams = @{
    "mi360" = "rtsp://127.0.0.1:8554/mi360_h264"
    "c700"  = "rtsp://127.0.0.1:8554/c700_h264"
}

foreach ($name in $cams.Keys) {
    $url = $cams[$name]
    # Only the PTZ camera's monitor honors the tracker moving-flag; giving it
    # to the other monitor would wrongly suppress motion during mi360 moves.
    if ($name -eq "mi360") {
        $env:SEIZUREGUARD_MOVING_FLAG = Join-Path $repo "data\tracker\mi360_moving.json"
    } else {
        Remove-Item Env:SEIZUREGUARD_MOVING_FLAG -ErrorAction SilentlyContinue
    }
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match [regex]::Escape($url) -and
                       $_.CommandLine -match "monitor.py" }
    if ($running) { continue }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Start-Process -WindowStyle Hidden -FilePath "python" `
        -ArgumentList "-u", (Join-Path $repo "src\monitor.py"),
            "--source", $url, "--name", $name `
        -WorkingDirectory $repo `
        -RedirectStandardOutput (Join-Path $logDir "$name`_$stamp.log") `
        -RedirectStandardError (Join-Path $logDir "$name`_$stamp.err")
}

# PTZ dog tracker: opt-in until the mi360 motor channel proves reliable
# (2026-08-09: command bursts wedged it - see FOLLOWUPS). Enable with
# setx SEIZUREGUARD_TRACK 1, then kill python processes once.
if ($env:SEIZUREGUARD_TRACK -eq "1" -and (Test-Path $posePython)) {
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "tracker.py" }
    if (-not $running) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        Start-Process -WindowStyle Hidden -FilePath $posePython `
            -ArgumentList "-u", (Join-Path $repo "src\tracker.py"),
                "--source", $cams["mi360"], "--cam", "mi360" `
            -WorkingDirectory $repo `
            -RedirectStandardOutput (Join-Path $logDir "tracker_$stamp.log") `
            -RedirectStandardError (Join-Path $logDir "tracker_$stamp.err")
    }
}
