# 阶段 A 一键脚本：测试 -> 训练 -> 评测（需要 py -3.12 + torch）
# 用法: .\run_exp_a.ps1 [-Updates 500]
param(
    [int]$Updates = 500,
    [string]$Out = "runs/base"
)

$ErrorActionPreference = "Stop"
$py = "py"
$args = @("-3.12")

Push-Location "$PSScriptRoot\exp-a-homeostatic-rl"

Write-Host "== 1/3 安装依赖 ==" -ForegroundColor Cyan
& $py @args -m pip install -e ".[dev]" --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== 2/3 单元测试 ==" -ForegroundColor Cyan
& $py @args -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

Write-Host "== 3/3 训练 $Updates updates -> $Out ==" -ForegroundColor Cyan
& $py @args -m homeo_rl.train --updates $Updates --out $Out
if ($LASTEXITCODE -ne 0) { throw "train failed" }

Write-Host "== 行为探针 ==" -ForegroundColor Cyan
& $py @args -m homeo_rl.evaluate --ckpt "$Out/checkpoint.pt"

Pop-Location
