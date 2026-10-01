# 安装 CUDA torch（C 盘紧张：先卸 CPU 版，临时目录放 D 盘，不留缓存）
# Install CUDA torch (C: is tight: uninstall CPU build first, temp on D:, no cache)
$ErrorActionPreference = "Continue"
$log = "D:\autopoiesis\autopoiesis\exp-b-qwen-homeostatic\install_cuda.log"

New-Item -ItemType Directory -Force D:\tmp | Out-Null
$env:TMP = "D:\tmp"; $env:TEMP = "D:\tmp"

"== uninstall cpu torch ==" | Out-File $log -Encoding utf8
py -3.12 -m pip uninstall -y torch *>> $log

"== install cu126 torch ==" | Out-File $log -Append -Encoding utf8
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir *>> $log
"torch exit: $LASTEXITCODE" | Out-File $log -Append -Encoding utf8

"== verify ==" | Out-File $log -Append -Encoding utf8
py -3.12 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" *>> $log
Get-PSDrive C | ForEach-Object { "C free GB: {0:N1}" -f ($_.Free/1GB) } | Out-File $log -Append -Encoding utf8
"DONE" | Out-File $log -Append -Encoding utf8
