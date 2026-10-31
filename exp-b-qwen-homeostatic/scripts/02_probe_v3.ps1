<#
RSI 第 1 步：给 v3 做全面测量（四个探针 + 剂量-反应曲线）。

这一步回答一个悬了一周的问题：v3 的 modulation_gap 是否从 v2 的 -1.0
转正（> 0.5 才算通过）。

分两批跑是为了内存紧张时也能进行：先跑全套（agreement/survival/
modulation），再单独跑 sweep——probe 的 --only 会**合并进同一个
probe_report.json**，不会覆盖前三个结果。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\02_probe_v3.ps1
  powershell -ExecutionPolicy Bypass -File scripts\02_probe_v3.ps1 -Ckpt runs\sft-lora-v2
#>
param(
    [string]$Ckpt = 'runs/dpo-v3'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:HF_HOME = 'D:\hf_cache'

$teacher = '../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt'
Write-Host "== probe: $Ckpt (full suite) ==" -ForegroundColor Cyan
& py -3.12 -m homeo_lm.probe --ckpt $Ckpt --teacher $teacher
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: full suite" -ForegroundColor Red; exit 1 }

Write-Host "== probe: $Ckpt (modulation sweep) ==" -ForegroundColor Cyan
& py -3.12 -m homeo_lm.probe --ckpt $Ckpt --teacher $teacher --only modulation-sweep
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: sweep" -ForegroundColor Red; exit 1 }

$reportPath = Join-Path $Ckpt 'probe_report.json'
if (-not (Test-Path $reportPath)) {
    Write-Host "FAIL: $reportPath not found" -ForegroundColor Red
    exit 1
}
$rep = Get-Content $reportPath -Raw | ConvertFrom-Json
$gap   = $rep.modulation.modulation_gap
$surv  = $rep.survival.survival_rate
$agree = $rep.agreement.agreement
$slope = $rep.'modulation-sweep'.slope
$mono  = $rep.'modulation-sweep'.monotone_increasing

Write-Host ""
Write-Host "=========== VERDICT: $Ckpt ===========" -ForegroundColor Cyan
Write-Host ("modulation_gap      : {0:N3}" -f $gap)
Write-Host ("survival_rate       : {0:N3}" -f $surv)
Write-Host ("agreement           : {0:N3}" -f $agree)
Write-Host ("sweep slope         : {0:N3}" -f $slope)
Write-Host ("monotone_increasing : {0}" -f $mono)

if ($gap -ge 0.5) {
    Write-Host "PASS: 调制已转正（v3 判据 gap > 0.5）" -ForegroundColor Green
    exit 0
}
if ($gap -lt 0) {
    Write-Host "FAIL: 仍是反向调制（v2 的失败模式：越饿越不去找食物）" `
        -ForegroundColor Red
    exit 1
}
Write-Host "PARTIAL: 调制为正但未过 0.5 — 记录数值，继续下一轮改进" `
    -ForegroundColor Yellow
exit 0
