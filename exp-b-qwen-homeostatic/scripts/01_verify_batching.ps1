<#
RSI 第 0 步（硬性门禁）：证明"批量首 token 快路径"测的是同一个策略。

为什么必须先跑这一步：探针提速约 200× 靠的是把"贪心解码一个动作词"
换成"5 个动作首 token 上取 argmax"。如果两者不等价，后面所有指标测的
就不是被测策略——数字再漂亮也没有意义。

不通过时脚本以非零码退出，且 02/03 由 run_all.ps1 拦住不许继续。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\01_verify_batching.ps1
  powershell -ExecutionPolicy Bypass -File scripts\01_verify_batching.ps1 -Ckpt runs\dpo-v3 -N 50
#>
param(
    [string]$Ckpt = 'runs/dpo-v3',
    [int]$N = 50
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:HF_HOME = 'D:\hf_cache'

$teacher = '../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt'
Write-Host "== verify batching: $Ckpt on $N states ==" -ForegroundColor Cyan

$out = & py -3.12 -m homeo_lm.probe --ckpt $Ckpt --teacher $teacher `
    --verify-batching $N 2>&1
$code = $LASTEXITCODE
$out | ForEach-Object { Write-Host $_ }

if ($code -ne 0) {
    Write-Host "FAIL: probe exited with code $code" -ForegroundColor Red
    exit 1
}
if (($out | Out-String) -match 'BATCHING OK') {
    Write-Host "PASS: 快路径 == 逐条 generate，可以使用" -ForegroundColor Green
    exit 0
}
Write-Host "FAIL: BATCHING MISMATCH — 禁止使用快路径，后续数字一律作废" `
    -ForegroundColor Red
exit 1
