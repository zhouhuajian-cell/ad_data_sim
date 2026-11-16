# 一键下发工程（Windows 侧）
# 干什么：在 Bypass 执行策略下调用 sync_project.ps1（脚本未签名，受执行策略限制时用本文件绕过），
#         下发完成后自动在服务器上跑大小写校验。
# 怎么跑：右键「使用 PowerShell 运行」，或：
#         powershell -ExecutionPolicy Bypass -File windows\sync_avsynth.ps1
#         可选：-ProjectDir / -Remote / -RemoteDir / -KeyFile
# 成功判据：屏幕最后出现「case_check ALL PASS」；随后回到操作台第 3 步点「开始出图」。
param(
    [string]$ProjectDir = "C:\UE_Projects\avsynth",
    [string]$Remote = "root@10.2.248.34",
    [string]$RemoteDir = "/data/avsynth",
    [string]$KeyFile = ""
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$sync = Join-Path $here "sync_project.ps1"
if (-not (Test-Path $sync)) { throw "找不到 $sync" }
$args1 = @{ ProjectDir = $ProjectDir; Remote = $Remote; RemoteDir = $RemoteDir }
if ($KeyFile) { $args1["KeyFile"] = $KeyFile }
& $sync @args1
if ($LASTEXITCODE -ne 0) { throw "同步失败 code=$LASTEXITCODE" }

Write-Host "== 在服务器上跑大小写/模块一致性校验"
$sshOpts = @("-o", "BatchMode=yes")
if ($KeyFile) { $sshOpts += @("-i", $KeyFile) }
& ssh @sshOpts $Remote "cd $RemoteDir && python3 Content/Python/case_check.py $RemoteDir"
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! 校验没通过：按上面的 FAIL 行改大小写/模块名后重新下发（ALL PASS 才继续出图）" -ForegroundColor Yellow
} else {
    Write-Host "== 全部就绪：回到操作台 http://10.2.248.34:8010 第 3 步点「开始出图」" -ForegroundColor Green
}
