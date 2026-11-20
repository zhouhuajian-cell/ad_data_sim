# sync_project.ps1 — 工程单向下发 Ubuntu（方案 2.3 / 3.6）
# 干什么：把 uproject/Content/Config/Source/Plugins 从 Windows 同步到 Ubuntu，排除中间产物。
# 怎么跑：powershell -File sync_project.ps1 -ProjectDir C:\...\avsynth -Remote root@10.2.248.34 -RemoteDir /data/avsynth
# 成功判据：远端 python3 case_check.py <工程目录> ALL PASS；同步**单向**，不做回传。
# ⚠️ 不做双向同步：同一 uasset 两端同时改写会版本分叉（方案 2.3）。
# ⚠️ 排除清单与方案表 7 一致：Intermediate/Saved/Binaries/.vs 跨平台无效，DerivedDataCache 仅版本完全一致才考虑。
param(
    [Parameter(Mandatory = $true)][string]$ProjectDir,
    [Parameter(Mandatory = $true)][string]$Remote,
    [Parameter(Mandatory = $true)][string]$RemoteDir
)
$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path $ProjectDir).Path
if (-not (Test-Path "$ProjectDir\*.uproject")) { throw "目录里没有 .uproject: $ProjectDir" }
$ex = @("--exclude=Intermediate", "--exclude=Saved", "--exclude=Binaries", "--exclude=.vs", "--exclude=DerivedDataCache")
if (Get-Command rsync -ErrorAction SilentlyContinue) {
    & rsync -rltD @ex -e "ssh -o BatchMode=yes" "$ProjectDir/" "${Remote}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) { throw "rsync 失败 code=$LASTEXITCODE" }
} else {
    Write-Host "未找到 rsync，退化为 tar-over-ssh（同排除清单；需 Win10 1803+ 内置 bsdtar）"
    $tarArgs = ($ex -join " ")
    $remoteCmd = "mkdir -p $RemoteDir && tar -xf - -C $RemoteDir"
    $cmdLine = "tar $tarArgs -cf - . | ssh -o BatchMode=yes $Remote `"$remoteCmd`""
    cmd /c $cmdLine
    if ($LASTEXITCODE -ne 0) { throw "tar-over-ssh 失败 code=$LASTEXITCODE" }
}
Write-Host "同步完成 -> ${Remote}:${RemoteDir}"
Write-Host "下一步(必须): 在 Ubuntu 上跑 python3 case_check.py $RemoteDir —— 大小写不一致是加载失败最高频原因"
