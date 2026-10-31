<#
RSI 第 3 步：跑一个 L0 自适应调参周期（提议 → 门控 → 试验 → 判定）。

默认 **dry-run**（只打印"下一步试什么"和训练命令）；加 -Run 才真跑训练。
真跑一次 = 重新生成偏好对 + DPO 训练 + 多次独立测量，请在时间/内存允许的
时候跑。

关键设计：所有试验都从**同一个 parent**（默认 runs\sft-lora-v2）出发，
否则"这个旋钮好不好"会和"起点不同"混在一起，无法归因。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\04_l0_cycle.ps1
  powershell -ExecutionPolicy Bypass -File scripts\04_l0_cycle.ps1 -Run -Samples 3
  powershell -ExecutionPolicy Bypass -File scripts\04_l0_cycle.ps1 -Run -MeasureParent
#>
param(
    [string]$Ckpt = 'runs/dpo-v3',
    [string]$Ledger = 'runs/rsi_ledger.jsonl',
    [int]$Samples = 3,
    [int]$MaxTrials = 5,
    [switch]$Run,
    [switch]$MeasureParent
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:HF_HOME = 'D:\hf_cache'

$teacher = '../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt'
$mode = if ($Run) { 'REAL RUN' } else { 'DRY-RUN' }
Write-Host "== L0 cycle ($mode) ledger=$Ledger samples=$Samples ==" `
    -ForegroundColor Cyan

$cmdArgs = @(
    '-m', 'homeo_lm.rsi', '--cycle',
    '--ckpt', $Ckpt,
    '--teacher', $teacher,
    '--out', $Ledger,
    '--samples', $Samples,
    '--max-trials', $MaxTrials
)
if ($Run) { $cmdArgs += '--run-trials' }
if ($MeasureParent) { $cmdArgs += '--measure-parent' }

& py -3.12 @cmdArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "STOP: L0 cycle exited $code" -ForegroundColor Red
    exit $code
}
Write-Host "DONE: 结果已记入 $Ledger（采纳≠上线，上线由 P4 金丝雀决定）" `
    -ForegroundColor Green
exit 0
