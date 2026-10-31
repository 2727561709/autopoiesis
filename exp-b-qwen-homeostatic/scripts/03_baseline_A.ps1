<#
RSI 第 2 步：把 v3 的测量写进账本，作为**对照组 A 的零点**。

为什么要这一行：RSI 是否真的有效，判据是"同等算力下 D（L0+L1）是否优于
A（固定课程一直训）"。A 的零点不落账本，后面所有"改进了多少"都是空的。

这条记录 verdict = "baseline"（不带 change 时 rsi.py 自动标注），探针配置
哈希一并写入——将来谁改了探针，本轮所有采纳作废。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\03_baseline_A.ps1
  powershell -ExecutionPolicy Bypass -File scripts\03_baseline_A.ps1 -Tag baseline-A-v2 -Ckpt runs\sft-lora-v2
#>
param(
    [string]$Ckpt = 'runs/dpo-v3',
    [string]$Tag = 'baseline-A',
    [string]$Ledger = 'runs/rsi_ledger.jsonl'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:HF_HOME = 'D:\hf_cache'

$teacher = '../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt'
Write-Host "== baseline: $Ckpt -> $Ledger (tag $Tag) ==" -ForegroundColor Cyan

& py -3.12 -m homeo_lm.rsi --ckpt $Ckpt --teacher $teacher `
    --tag $Tag --out $Ledger
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "FAIL: rsi exited with code $code" -ForegroundColor Red
    exit 1
}

if (Test-Path $Ledger) {
    $n = (Get-Content $Ledger | Where-Object { $_.Trim() -ne '' }).Count
    Write-Host "ledger now has $n entries: $Ledger" -ForegroundColor Green
}
Write-Host "PASS: 对照组零点已记账" -ForegroundColor Green
exit 0
