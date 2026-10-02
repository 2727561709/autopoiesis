# 阶段 B v2 流水线：200 局数据（新种子）→ 动作平衡 SFT（2 epochs）→ 三探针
# Stage-B v2 pipeline: 200-episode data (fresh seed) -> balanced SFT
# (2 epochs, collapse monitoring) -> three probes
# v1 教训：rest 占 50% 的类不平衡 + 1 epoch → 策略坍缩为单一动作
# v1 lesson: rest at 50% + 1 epoch -> the policy collapsed to one action
$ErrorActionPreference = "Continue"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = "D:\hf_cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$log = "D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic\sft_v2_run.log"

Push-Location D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic

"== v2 1/3 生成教师数据（200 局，seed 7）==" | Out-File $log -Encoding utf8
py -3.12 -m homeo_lm.gen_data --ckpt ..\exp-a-homeostatic-rl\runs\curriculum\checkpoint.pt --episodes 200 --seed 7 --out data/sft_v2.jsonl *>> $log
"gen exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "gen_data failed" }

"== v2 2/3 平衡 SFT（2 epochs，坍缩监控）==" | Out-File $log -Append -Encoding utf8
py -3.12 -m homeo_lm.sft --data data/sft_v2.jsonl --epochs 2 --batch-size 2 --accum 16 --balance --resume --out runs/sft-lora-v2 *>> $log
"sft exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "sft failed" }

"== v2 3/3 行为探针 ==" | Out-File $log -Append -Encoding utf8
py -3.12 -m homeo_lm.probe --ckpt runs/sft-lora-v2 --teacher ..\exp-a-homeostatic-rl\runs\curriculum\checkpoint.pt *>> $log
"probe exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

Pop-Location
"ALL DONE" | Out-File $log -Append -Encoding utf8
