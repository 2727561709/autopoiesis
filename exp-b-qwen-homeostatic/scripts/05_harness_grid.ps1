<#
RSI 第 4 步（终局）：跑 A–E 对照组网格，判定 RSI 是否真的有效。

这是唯一能回答下面这个问题的实验：
    指标涨了，是"自我改进有效"，还是"同等算力下闷头多训也会涨"？

所有组**从同一个起点**（默认 runs\sft-lora-v2）出发、用**同等算力预算**。
Harness 会先核对算力是否真相等——不等则整轮作废（那是有意义的保护，不是
bug：算力不等的比较等于没比）。

支持 -Resume：跳过已完成的格子，长跑中断不用重来。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\05_harness_grid.ps1
  powershell -ExecutionPolicy Bypass -File scripts\05_harness_grid.ps1 -Run
  powershell -ExecutionPolicy Bypass -File scripts\05_harness_grid.ps1 -Run -Groups ABCDE -Seeds 0,1,2 -Resume
#>
param(
    [string]$Groups = 'ABCDE',
    [string]$Seeds = '0,1,2',
    [string]$Start = 'runs/sft-lora-v2',
    [string]$Out = 'runs/harness_results.jsonl',
    [int]$Samples = 3,
    [int]$MaxTrials = 5,
    [double]$Noise = 0.02,
    [string]$Device = '',
    [switch]$Run,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:HF_HOME = 'D:\hf_cache'

$teacher = '../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt'
$mode = if ($Run) { 'REAL RUN' } else { 'DRY-RUN' }
Write-Host "== harness grid ($mode) groups=$Groups seeds=$Seeds ==" `
    -ForegroundColor Cyan

$cmdArgs = @(
    '-m', 'homeo_lm.harness',
    '--groups', $Groups,
    '--seeds', $Seeds,
    '--start', $Start,
    '--teacher', $teacher,
    '--out', $Out,
    '--samples', $Samples,
    '--max-trials', $MaxTrials,
    '--noise', $Noise
)
if ($Device) { $cmdArgs += @('--device', $Device) }
if ($Run) { $cmdArgs += '--run' }
if ($Resume) { $cmdArgs += '--resume' }

& py -3.12 @cmdArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "STOP: harness exited $code" -ForegroundColor Red
    exit $code
}
Write-Host "DONE: 结果在 $Out（看 verdict 字段：rsi_pays 或 no_evidence）" `
    -ForegroundColor Green
exit 0
