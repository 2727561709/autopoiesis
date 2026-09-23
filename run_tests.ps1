# 逐模块独立运行测试（保持"独立仓库、独立测试"原则）
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$srcs = (Get-ChildItem $root -Directory -Filter "cog-*" | ForEach-Object { Join-Path $_.FullName "src" }) -join ";"
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$env:PYTHONPATH;$srcs" } else { $env:PYTHONPATH = $srcs }
$failed = $false
foreach ($m in (Get-ChildItem $root -Directory -Filter "cog-*")) {
    if (-not (Test-Path (Join-Path $m.FullName "tests"))) { continue }
    Write-Host "`n=== $($m.Name) ===" -ForegroundColor Cyan
    python -m pytest $m.FullName -q
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
if ($failed) { exit 1 } else { Write-Host "`nAll modules passed." -ForegroundColor Green }
