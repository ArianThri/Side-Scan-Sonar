$projectRoot = Split-Path -Parent $PSScriptRoot

$env:SCTD_PROJECT_ROOT = $projectRoot
$env:TEMP = Join-Path $projectRoot ".tmp\runtime"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
$env:TORCH_HOME = Join-Path $projectRoot "models\cache\torch"
$env:HF_HOME = Join-Path $projectRoot "models\cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot ".config\ultralytics"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".config\matplotlib"
$env:XDG_CACHE_HOME = Join-Path $projectRoot ".cache"

@(
    $env:TEMP,
    $env:PIP_CACHE_DIR,
    $env:TORCH_HOME,
    $env:HF_HOME,
    $env:HF_HUB_CACHE,
    $env:YOLO_CONFIG_DIR,
    $env:MPLCONFIGDIR,
    $env:XDG_CACHE_HOME
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

& (Join-Path $projectRoot ".venv\Scripts\Activate.ps1")
Write-Host "SCTD environment active at $projectRoot"
