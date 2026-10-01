# 安装 CUDA torch + exp-b 依赖（日志落盘便于检查）
# Install CUDA torch + exp-b dependencies (log to file for inspection)
$ErrorActionPreference = "Continue"
$log = "D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic\install.log"

"== torch cu126 ==" | Out-File $log -Encoding utf8
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade *>> $log
"torch exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

"== transformers peft accelerate bitsandbytes ==" | Out-File $log -Append -Encoding utf8
py -3.12 -m pip install transformers peft accelerate bitsandbytes datasets *>> $log
"deps exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

"== exp-a + exp-b editable installs ==" | Out-File $log -Append -Encoding utf8
py -3.12 -m pip install -e D:\autopoiesis\autopoiesis\exp-a-homeostatic-rl *>> $log
py -3.12 -m pip install -e D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic *>> $log
"editable exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

"== verify ==" | Out-File $log -Append -Encoding utf8
py -3.12 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" *>> $log
py -3.12 -c "import transformers, peft; print('transformers', transformers.__version__, 'peft', peft.__version__)" *>> $log
"DONE" | Out-File $log -Append -Encoding utf8
