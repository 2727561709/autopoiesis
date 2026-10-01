# 低显存 SFT + 探针（防碎片配置，RTX 3050 4GB）
# Low-VRAM SFT + probes (fragmentation-resistant config, RTX 3050 4GB)
$ErrorActionPreference = "Continue"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = "D:\hf_cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$log = "D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic\sft_run.log"

Push-Location D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic

"== SFT (1 epoch, batch 2, accum 16, expandable segments, resume) ==" | Out-File $log -Encoding utf8
py -3.12 -m homeo_lm.sft --data data/sft.jsonl --epochs 1 --batch-size 2 --accum 16 --resume --out runs/sft-lora *>> $log
"sft exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "sft failed" }

"== probes ==" | Out-File $log -Append -Encoding utf8
py -3.12 -m homeo_lm.probe --ckpt runs/sft-lora --teacher ..\exp-a-homeostatic-rl\runs\curriculum\checkpoint.pt *>> $log
"probe exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

Pop-Location
"ALL DONE" | Out-File $log -Append -Encoding utf8
