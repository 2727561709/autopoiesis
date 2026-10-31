<#
RSI 三步流水线（按顺序，任一步失败即停）。

  01_verify_batching  证明快路径 == 逐条 generate（不过不许用任何数字）
  02_probe_v3         全面测量 v3：gap 是否从 v2 的 -1.0 转正
  03_baseline_A       把测量写进账本，作为对照组 A 的零点

用法：
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Ckpt runs\dpo-v3

单步跑（内存紧张时推荐）：
  powershell -ExecutionPolicy Bypass -File scripts\01_verify_batching.ps1
  ...
#>
param(
    [string]$Ckpt = 'runs/dpo-v3'
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$steps = @('01_verify_batching.ps1', '02_probe_v3.ps1', '03_baseline_A.ps1')

foreach ($s in $steps) {
    $path = Join-Path $here $s
    Write-Host ""
    Write-Host "############ $s ############" -ForegroundColor Cyan
    & $path -Ckpt $Ckpt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STOP: $s failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "ALL STEPS DONE — 对照组零点已就位，可以开始跑 L0 周期" `
    -ForegroundColor Green
exit 0
