# 单独运行阶段 B 三探针（SFT 已完成，教师数据已在库）
# Run the Stage-B probes alone (SFT finished; teacher data already generated)
$ErrorActionPreference = "Continue"
$env:HF_HOME = "D:\hf_cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$log = "D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic\probe_run.log"

Push-Location D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic
"== probes (run 2) ==" | Out-File $log -Encoding utf8
py -3.12 -m homeo_lm.probe --ckpt runs/sft-lora --teacher ..\exp-a-homeostatic-rl\runs\curriculum\checkpoint.pt *>> $log
"probe exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
Pop-Location
"ALL DONE" | Out-File $log -Append -Encoding utf8
