$ErrorActionPreference = 'Continue'
$base = "C:\Users\zhouhuajian\Desktop\sdxl_files"
$sdxl = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
$cn   = "diffusers/controlnet-canny-sdxl-1.0"

# SDXL inpaint 需要的文件（fp16 版）
$sdxlFiles = @(
  "model_index.json",
  "scheduler/scheduler_config.json",
  "tokenizer/tokenizer_config.json","tokenizer/special_tokens_map.json","tokenizer/vocab.json","tokenizer/merges.txt",
  "tokenizer_2/tokenizer_config.json","tokenizer_2/special_tokens_map.json","tokenizer_2/vocab.json","tokenizer_2/merges.txt",
  "text_encoder/config.json","text_encoder/model.fp16.safetensors",
  "text_encoder_2/config.json","text_encoder_2/model.fp16.safetensors",
  "unet/config.json","unet/diffusion_pytorch_model.fp16.safetensors",
  "vae/config.json","vae/diffusion_pytorch_model.fp16.safetensors"
)
$cnFiles = @(
  "config.json","diffusion_pytorch_model.fp16.safetensors"
)

$sw = [Diagnostics.Stopwatch]::StartNew()

function Get-File($repo, $relPath, $destRoot) {
  $dest = Join-Path $destRoot ($relPath -replace '/', '\')
  $d = Split-Path $dest -Parent
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
  if ((Test-Path $dest) -and (Get-Item $dest).Length -gt 0) {
    Write-Output ("  [跳过] {0}" -f $relPath)
    return
  }
  $url = "https://hf-mirror.com/$repo/resolve/main/$relPath"
  & curl.exe -sL --max-time 3000 -o $dest $url 2>$null
  if ((Test-Path $dest) -and (Get-Item $dest).Length -gt 0) {
    $mb = [math]::Round((Get-Item $dest).Length/1MB,1)
    Write-Output ("  [完成] {0,-58} {1,7} MB  ({2}m)" -f $relPath, $mb, [math]::Round($sw.Elapsed.TotalMinutes,1))
  } else {
    Write-Output ("  [失败] {0}" -f $relPath)
  }
}

Write-Output "=== SDXL inpaint (7GB) ==="
foreach ($f in $sdxlFiles) { Get-File $sdxl $f "$base\sdxl-inpaint" }

Write-Output ""
Write-Output "=== ControlNet canny SDXL (2.5GB) ==="
foreach ($f in $cnFiles) { Get-File $cn $f "$base\controlnet-canny-sdxl" }

Write-Output ""
$total = (Get-ChildItem $base -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Output ("总计 {0} GB，用时 {1} 分钟" -f [math]::Round($total/1GB,2), [math]::Round($sw.Elapsed.TotalMinutes,1))
