# sync_project.ps1 - 工程单向下发 Ubuntu（方案 2.3 / 3.6）
# 干什么：把 uproject/Content/Config/Source/Plugins 从 Windows 同步到 Ubuntu，排除中间产物。
# 怎么跑：powershell -ExecutionPolicy Bypass -File sync_project.ps1 -ProjectDir C:\...\avsynth -Remote root@10.2.248.34 -RemoteDir /data/avsynth
#         可选 -KeyFile <私钥路径>（默认用 ~/.ssh 下已有的免密 key）。
# 成功判据：远端 python3 case_check.py <工程目录> ALL PASS；同步单向，不做回传。
# 注意：不做双向同步（同一 uasset 两端同时改写会版本分叉，方案 2.3）。
# 排除清单与方案表 7 一致：Intermediate/Saved/Binaries/.vs 跨平台无效，DerivedDataCache 仅版本完全一致才考虑。
# 为什么不用 tar | ssh 管道：Windows 自带 tar 与 ssh 之间是命令管道，大文件会被截断
#   （实测 22MB 只传过去 655 字节，解包时报 "Skipping to next header"）。这里改成
#   tar -> 本地临时包 -> scp -> 远端解包，全程原生进程调用，路径带空格也不会断。
param(
    [Parameter(Mandatory = $true)][string]$ProjectDir,
    [Parameter(Mandatory = $true)][string]$Remote,
    [Parameter(Mandatory = $true)][string]$RemoteDir,
    [string]$KeyFile = "",
    [string]$KeepArchive = ""
)
$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path $ProjectDir).Path
$up = @(Get-ChildItem -Path $ProjectDir -Filter *.uproject -File)
if ($up.Count -eq 0) { throw "目录里没有 .uproject: $ProjectDir" }
$ex = @("--exclude=Intermediate", "--exclude=Saved", "--exclude=Binaries", "--exclude=.vs", "--exclude=DerivedDataCache")
$sshOpts = @("-o", "BatchMode=yes")
if ($KeyFile) {
    if (-not (Test-Path $KeyFile)) { throw "私钥不存在: $KeyFile" }
    $sshOpts += @("-i", $KeyFile)
}
if ($KeepArchive) {
    $tarPath = [IO.Path]::GetFullPath($KeepArchive)
} else {
    $tarPath = Join-Path ([IO.Path]::GetTempPath()) ("avsynth_sync_" + [guid]::NewGuid().ToString("N") + ".tar")
}
$remoteTar = "/tmp/" + [IO.Path]::GetFileName($tarPath)

if (Get-Command rsync -ErrorAction SilentlyContinue) {
    & rsync -rltD @ex -e ("ssh " + ($sshOpts -join " ")) "$ProjectDir/" "${Remote}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) { throw "rsync 失败 code=$LASTEXITCODE" }
} else {
    Write-Host "未找到 rsync，走 tar -> 本地临时包 -> scp -> 远端解包（同排除清单）"
    # 排除选项必须写在 -f 之前，否则 Windows tar 会把它当文件参数
    $tarOpts = @($ex) + @("-cf", $tarPath, ".")
    Push-Location $ProjectDir
    try {
        & tar @tarOpts
        if ($LASTEXITCODE -ne 0) { throw "打 tar 失败 code=$LASTEXITCODE" }
    } finally {
        Pop-Location
    }
    $size = (Get-Item $tarPath).Length
    if ($size -lt 1024) { throw "打包结果异常（仅 $size 字节），中止" }
    Write-Host ("== 打包完成 {0:N1} MB，上传到服务器…" -f ($size / 1MB))
    & scp @sshOpts $tarPath ("{0}:{1}" -f $Remote, $remoteTar)
    if ($LASTEXITCODE -ne 0) { throw "scp 失败 code=$LASTEXITCODE" }
    & ssh @sshOpts $Remote "mkdir -p '$RemoteDir' && tar -xf '$remoteTar' -C '$RemoteDir' && rm -f '$remoteTar' && cd '$RemoteDir' && rm -rf Intermediate Saved Binaries .vs DerivedDataCache"
    if ($LASTEXITCODE -ne 0) { throw "远端解包失败 code=$LASTEXITCODE" }
    if (-not $KeepArchive) { Remove-Item $tarPath -Force -ErrorAction SilentlyContinue }
}
Write-Host "同步完成 -> ${Remote}:${RemoteDir}"
Write-Host "下一步(必须): 在 Ubuntu 上跑 python3 $RemoteDir/Content/Python/case_check.py $RemoteDir - 大小写不一致是加载失败最高频原因"
