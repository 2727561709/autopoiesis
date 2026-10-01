# 断点续传下载 cu126 torch wheel（官方源超时也能续传）
# Resumable download of the cu126 torch wheel (survives timeouts)
$url = "https://download.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-win_amd64.whl"
$out = "D:\tmp\torch_cu126.whl"
for ($i = 1; $i -le 60; $i++) {
    curl.exe -L -C - --retry 3 --retry-delay 3 --connect-timeout 30 --speed-time 60 --speed-limit 10000 -o $out $url
    if ($LASTEXITCODE -eq 0) { "download complete: $out"; break }
    "attempt $i failed (exit $LASTEXITCODE), resuming in 5s..."
    Start-Sleep 5
}
if ($LASTEXITCODE -ne 0) { "DOWNLOAD FAILED after retries"; exit 1 }
"size: {0:N2} GB" -f ((Get-Item $out).Length / 1GB)
