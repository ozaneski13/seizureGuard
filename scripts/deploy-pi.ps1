# deploy-pi.ps1 - push current src/ + scripts/ to the Pi and restart monitors.
$repo = Split-Path -Parent $PSScriptRoot
scp -q (Join-Path $repo "src\monitor.py") (Join-Path $repo "src\alerts.py") `
    (Join-Path $repo "src\verify_event.py") (Join-Path $repo "src\pose_gate.py") `
    (Join-Path $repo "src\tracker.py") pi:~/seizureGuard/src/
scp -q (Join-Path $repo "scripts\sampling.py") `
    (Join-Path $repo "scripts\extract_event_from_video.py") pi:~/seizureGuard/scripts/
ssh pi 'sudo systemctl restart seizureguard-mi360 seizureguard-c700; sleep 8; systemctl is-active seizureguard-mi360 seizureguard-c700'
