# 阶段 B 一键流水线：数据生成 -> QLoRA SFT -> 行为探针
# Stage-B one-shot pipeline: data gen -> QLoRA SFT -> behavioral probes
# 用法 / usage: .\run_exp_b.ps1 [-Episodes 200]
param(
    [int]$Episodes = 40,
    [int]$MaxSteps = 400,
    [string]$Teacher = "runs/curriculum/checkpoint.pt",
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct"
)

# 不用 Stop：py.exe 的无害 stderr 警告（如 bitsandbytes DLL）会误杀流水线，
# 出错判定改用 $LASTEXITCODE / Don't use Stop: harmless stderr warnings from
# py.exe (e.g. the bitsandbytes DLL notice) would kill the pipeline; use
# $LASTEXITCODE for error detection instead.
$ErrorActionPreference = "Continue"
$py = "py"
$args = @("-3.12")

# 国内镜像下载模型（缓存放 D 盘，C 盘紧张）/ China mirror for model
# download (cache on D:, C: is tight)
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\hf_cache"

Push-Location "$PSScriptRoot\exp-b-qwen-homeostatic"

if (-not (Test-Path $Teacher)) {
    # 从 exp-a 目录解析相对路径 / resolve relative to exp-a
    $Teacher = Join-Path "$PSScriptRoot\exp-a-homeostatic-rl" "runs\curriculum\checkpoint.pt"
}

Write-Host "== 1/3 生成教师轨迹数据 ==" -ForegroundColor Cyan
& $py @args -m homeo_lm.gen_data --ckpt $Teacher --episodes $Episodes --max-steps $MaxSteps --out data/sft.jsonl
if ($LASTEXITCODE -ne 0) { throw "gen_data failed" }

Write-Host "== 2/3 QLoRA SFT ==" -ForegroundColor Cyan
& $py @args -m homeo_lm.sft --data data/sft.jsonl --model $Model --out runs/sft-lora --resume
if ($LASTEXITCODE -ne 0) { throw "sft failed" }

Write-Host "== 3/3 行为探针 ==" -ForegroundColor Cyan
& $py @args -m homeo_lm.probe --ckpt runs/sft-lora --teacher $Teacher
if ($LASTEXITCODE -ne 0) { throw "probe failed" }

Pop-Location
Write-Host "阶段 B v1 完成 / Stage-B v1 done" -ForegroundColor Green
