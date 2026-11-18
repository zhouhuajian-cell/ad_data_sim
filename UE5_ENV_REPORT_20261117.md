# UE5 感知仿真平台 — 环境搭建与出图验证报告

> 编写时间：2026-11-17
> 目标机器：Ubuntu 20.04.6 LTS / RTX 4070 Ti 12G / 31G 内存 / 20 核
> 结论：**RGB 通道已能真实出图**；深度与语义通道需在编辑器中制作后处理材质后才能正确输出。

---

## 一、今天完成了什么

| 项目 | 状态 | 证据 |
|---|---|---|
| UE 5.3 源码编译 | ✅ 完成 | `[4001/4001] WriteMetadata UnrealEditor.target`，0 错误 |
| ShaderCompileWorker 编译 | ✅ 完成 | `Engine/Binaries/Linux/ShaderCompileWorker` 257 KB |
| 工程 C++ 模块编译 | ✅ 完成 | `libUnrealEditor-avsynth.so` |
| Vulkan RHI 初始化 | ✅ 完成 | 队列/内存/扩展全部正常 |
| 关卡加载与渲染 | ✅ 完成 | `MAP LOAD FILE=".../TestLevel.umap"`，MapCheck 0 错误 |
| **RGB 真实出图** | ✅ **完成** | 640×360，天空+地面+立柱清晰可见 |
| 深度通道 | ⚠️ 有数据但未线性化 | 0 字节 → 7266 字节，但数值是 FP16 原始范围 |
| 语义通道 | ⚠️ 输出成了颜色图 | 373 KB，内容为渲染结果而非 stencil |
| 真值 JSON 输出 | ✅ 完成 | 结构完整，相机参数/光照/文件路径齐全 |

---

## 二、今天踩过的坑（按重要性排序）

### 坑 1：预编译引擎 Vulkan 崩溃 —— 换源码编译解决【最关键】

**现象**：预编译版 `UnrealEditor-Cmd` 开 Vulkan 后必定 SIGSEGV。

**错误栈**：
```
FMallocMimalloc::Malloc(unsigned long, unsigned int) [MallocMimalloc.cpp:64]
Signal 11 caught.
```

**排查过的方向（都无效）**：
- 内存不足 → 排除（空闲 29G，无 ad_mining 进程）
- fd 上限 1024 → 调到 1048576 后仍崩
- 插件缺失（ImageWriteQueue / Substrate 曾被误写进 .uproject）→ 移除后仍崩
- root 权限 → 已用 algo 用户运行

**最终结论**：预编译版自带的 mimalloc 在本机（Ubuntu 20.04 + glibc 2.31 + 内核 5.15）上有兼容问题。**只有源码编译能绕开。**

---

### 坑 2：SPIR-V 优化器无限递归崩溃 —— 关掉 Substrate 解决

**现象**：源码引擎编译后，Vulkan 初始化都成功了，但着色器编译阶段崩。

**错误栈**（堆栈里同一对函数重复几十次）：
```
libdxcompiler.so!spvtools::opt::SSARewriter::AddPhiOperands(...)
libdxcompiler.so!spvtools::opt::SSARewriter::GetReachingDef(...)
```

**排查过的方向（都无效）**：
- 加大栈 `ulimit -s unlimited` → 仍崩
- 禁用 SPIR-V 优化 CVar → 仍崩
- 单线程着色器编译 → 仍崩
- 换 OpenGL 后端 → 仍崩（崩在同一个库）
- 替换官方新版 `libdxcompiler.so` → **不可行**：UE 的是 410 MB 定制版，官方版只有 12 MB，ABI 不兼容

**真正有效的解法**：
```ini
# /data/avsynth/Config/DefaultEngine.ini
r.Substrate=False
```

**原因**：`r.Substrate=True` 会强制编译大量 Strata/Substrate 调试可视化着色器（如 `FMaterialDebugStrataTreePS`、`StrataVisualize.usf`），这些着色器会触发 DXC 的 SPIR-V 优化器递归 bug。关掉后不再编译它们，崩溃消失。

> ⚠️ **注意**：方案文档里写「反光膜依赖 Substrate 材质系统」，所以关掉后**逆反射材质需要改用传统材质方案**（BRDF 近似即可，见第五节）。

---

### 坑 3：UBT 找不到 Linux SDK —— 环境变量解决

**现象**：
```
Platform Linux is not a valid platform to build. Check that the SDK is installed properly.
```
UBT 日志里所有平台都是 `buildable: False`。

**原因**：`Setup.sh` 虽然报 `SUCCESS`，但没把自带工具链路径写进 UBT 的识别配置。

**解法**：编译时显式传环境变量
```bash
SDK=/data/UnrealEngine/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v22_clang-16.0.6-centos7
BIN=$SDK/x86_64-unknown-linux-gnu/bin
export LINUX_MULTIARCH_ROOT="$SDK"
export CLANG="$BIN/clang++"
export CC="$BIN/clang"
export CXX="$BIN/clang++"
./Engine/Build/BatchFiles/Linux/Build.sh UnrealEditor Linux Development -j20 -Progress
```

> 自带工具链版本 `v22_clang-16.0.6-centos7` 正是 UBT 要求的版本，无需额外装 clang-16（apt 源里也没有）。

---

### 坑 4：Setup.sh 因缺 .git 失败 —— 手工跑 GitDependencies

**现象**：
```
fatal: not a git repository (or any of the parent directories): .git
Result: 126
```

**原因**：源码是从 Windows 本机打包传过来的，没带 `.git` 目录（也不必带）。

**解法**：跳过 Setup.sh 的 git 检查，直接跑依赖拉取
```bash
cd /data/UnrealEngine/Engine
../Engine/Binaries/DotNET/GitDependencies/linux-x64/GitDependencies --prompt
```
依赖共 **43374 个文件 / 7.7 GB**，会自动补齐整个 `Engine/Extras/ThirdPartyNotUE/SDKs/`。

---

### 坑 5：SceneCapture 出黑图 —— xvfb 虚拟显示解决

**现象**：`-RenderOffscreen` 模式下，RT 能创建、`capture_scene()` 能调、`export_render_target` 能写出文件，**但像素全是 RGB(0,0,0)**。

**原因**：`-RenderOffscreen` 只创建 Vulkan 设备，**不建立完整渲染线程和视口**。SceneCapture2D 需要渲染线程存在才能工作。

**解法**：用 xvfb 提供虚拟 X11 显示，走标准渲染路径
```bash
apt-get install -y xvfb
xvfb-run -a --server-args="-screen 0 1280x720x24" \
  ./Engine/Binaries/Linux/UnrealEditor-Cmd ... -Vulkan
```

**验证**：同样是 `-RenderOffscreen`，改用 xvfb 后图像从 7109 字节空图变成 119840 字节真实画面。

---

### 坑 6：UE 5.3 Python API 签名变化（3 处）

| API | 错误用法 | 5.3 正确用法 |
|---|---|---|
| `ImageWriteBlueprintLibrary.export_to_disk` | `(None, rt, path, options)` 4 参 | `(rt, path, options)` **3 参** |
| `RenderingLibrary.export_render_target` | `(world, rt, path)` 3 参 | `(world, rt, dir, filename)` **4 参** |
| 模块名 `KismetRenderingLibrary` | 不存在 | 改名为 `RenderingLibrary` |

**另外 `ImageWriteOptions.async` → `async_`**（多了下划线）。

**SceneCaptureSource 枚举全部改名**：
```
旧：SCS_SCENE_COLOR / SCS_FINAL_COLOR    ← 5.3 已删除
新：SCS_SCENE_COLOR_HDR / SCS_SCENE_COLOR_HDR_NO_ALPHA
    SCS_FINAL_COLOR_HDR / SCS_FINAL_COLOR_LDR
    SCS_SCENE_DEPTH / SCS_DEVICE_DEPTH
```
代码里已改成**候选链**（按顺序找第一个存在的），以后换版本不会挂。

---

### 坑 7：只有 RGBA 格式的 RT 能落盘

**实测结果**（本机 Vulkan）：

| 格式 | 落盘结果 |
|---|---|
| `RTF_RGBA8` | ✅ 652 字节（PNG） |
| `RTF_RGBA16F` | ✅ 990 字节（EXR） |
| `RTF_RG16F` | ❌ **0 字节** |
| `RTF_R8` | ❌ **0 字节** |
| `RTF_RGBA32F` | ❌ 0 字节 |
| `RTF_R32F` | ❌ 0 字节 |

**结论**：单通道或双通道浮点 RT 在本机 Vulkan 上写不出来。**所有通道统一用 RGBA 承载**（数据放 R 通道）。

---

### 坑 8：服务器访问 GitHub 被 DNS 劫持

**现象**：`git clone` 反复 `GnuTLS recv error (-110): TLS connection was non-properly terminated`，而 `curl https://github.com/` 有时能通。

**原因**：DNS 把 `github.com` 解析到 `20.6.7.17`，这个 IP 被企业网络设备拦截。

**解法**：`/etc/hosts` 固定到可用 IP
```
140.82.112.3  github.com
140.82.112.3  codeload.github.com
140.82.112.6  api.github.com
```

**另一个关键点**：git 默认低速保护（1KB/s 持续 10 秒即断开）会把慢速下载误判为失败，必须放宽：
```bash
git -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=999 \
    clone ...
```

**但即使这样，服务器克隆速度只有 150~500 KB/s**（20G 要 30 小时）。**最终方案是 Windows 本机克隆（9.35 MB/s，6 分钟完成）再流式传服务器（tar over ssh，约 3 MB/s）。**

---

## 三、当前环境状态

```
引擎源码     /data/UnrealEngine          60 GB（含依赖）
引擎产物     Engine/Binaries/Linux/UnrealEditor-Cmd      ✅
着色器编译器 Engine/Binaries/Linux/ShaderCompileWorker   ✅
工程         /data/avsynth
工程模块     Binaries/Linux/libUnrealEditor-avsynth.so   ✅
测试关卡     Content/Maps/TestLevel.umap  （含地面+2立柱+太阳光+天光+大气+雾）
xvfb         /usr/bin/xvfb-run           ✅
OpenCV       python3-opencv              ✅
旧预编译引擎 /data/UE_5.3                 57 GB（可删，已不用）
```

---

## 四、标准出图命令

```bash
UE=/data/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd
W=/tmp/run; mkdir -p $W/raw $W/meta $W/done; chown -R algo:algo $W
cat > $W/q.jsonl <<'EOF'
{"id":"f_0000","frame_id":"f_0000","seed":1,"map":"/Game/Maps/TestLevel",
 "targets":[],"camera":{"name":"front","width":640,"height":360,"hfov_deg":90,
 "location":[-800,0,200],"rotation":[0,0,0]},"lighting":{"condition":"front_light","weather":"clear"}}
EOF
chown algo:algo $W/q.jsonl

cd /data/UnrealEngine
runuser -u algo -- xvfb-run -a --server-args="-screen 0 1280x720x24" \
  env AD_UE_QUEUE=$W/q.jsonl AD_UE_RAW=$W/raw AD_UE_META=$W/meta \
      AD_UE_DONE=$W/done AD_UE_ERRLOG=$W/err.log \
  timeout 900 "$UE" /data/avsynth/avsynth.uproject \
  -ExecutePythonScript=/data/avsynth/Content/Python/batch_capture.py \
  -unattended -nosplash -noSound -NoLiveView -Vulkan -stdout
```

**成功判据**：`[capture] 完成 ok=1 fail=0`，且 `raw/` 下三个文件都非 0 字节。

**性能参考**：640×360 单帧约 2 秒（0.51 帧/秒），1080P 预计 4~8 秒/帧。

---

## 五、剩余工作（需在 UE 编辑器里做）

### 5.1 深度线性化后处理材质【必须】

**现状**：`SCS_SCENE_DEPTH` 输出的是 NDC 深度（0~1，非线性），当前写入 RGBA16F 的 R 通道，数值范围 0~65504，无法直接使用。

**要做**：在编辑器里创建一个 Post Process Material：
```
输入 SceneTexture:SceneDepth（或 SceneDepth 节点）
→ 线性化：LinearDepth = (1/d - (1-1/Far)) / ((1/Near)-(1-1/Far))
   或直接用 UE 的 LinearDepth 输出节点
→ 除以 200（方案规定深度范围 0~200m）
→ 输出到 Emissive Color（关掉所有光照影响，ShadingModel 设为 Unlit）
```

**验收**：输出的 R 通道值范围应在 0~1，且地面（近处）值小、远处值大。

### 5.2 语义模板值后处理材质【必须】

**现状**：`SCS_FINAL_COLOR_LDR` 输出的是渲染颜色，不是语义掩码。

**要做**：
1. 所有感知 Actor 打开 `Render CustomDepth Pass`，并设 `CustomDepth Stencil Value`（road=1 / vehicle=2 / pedestrian=3 / traffic_sign=4 / cone=5 / vegetation=6 / landline=7）
2. 创建 Post Process Material：
```
SceneTexture:CustomStencil（输出 0~255 整数）
→ 除以 255.0 归一化
→ 输出到 Emissive Color（Unlit）
```
3. 在 `batch_capture.py` 里把 sem 通道的捕获源改成挂载该材质的路径

**验收**：语义图中每个类别的像素值应精确等于 `stencil/255`。

### 5.3 正式场景资产【必须】

当前 `TestLevel.umap` 只有 3 个 Cube，用于验证管线。要产数据需要：
- 道路（可用 UE 的 Landscape 或静态网格）
- GOST 路标资产（限速/让行/禁停等）
- 车道线（Spline 或贴花）
- 车辆/行人/锥桶等交通目标

### 5.4 TSR 逆反射材质【方案要求】

因为关掉了 Substrate，逆反射需要用传统材质实现：
```
Base Color × (1 - DirtLevel) × (1 - FadeLevel)   ← 底色随脏污/褪色变化
Emissive = RetroIntensity × pow(saturate(dot(-ViewDir, LightDir)), 4) × LightColor
           × (1 + WetIntensity)
```
**硬要求**：不要用自发光直接给牌面提亮（那样任何角度都亮，与实车不符）。

---

## 六、已修复的代码位置

| 文件 | 改动 |
|---|---|
| `engine/Content/Python/batch_capture.py` | 捕获源候选链、`export_to_disk` 3 参、`export_render_target` 4 参、`async_` 字段、depth/sem 改 RGBA 格式、EXR 产物自动改名 |
| `/data/avsynth/Config/DefaultEngine.ini` | `r.Substrate=False` |
| `linux/fetch_ue5.sh`、`linux/clone_ue5_resilient.sh`、`linux/build_ue5_from_source.sh` | 新增，处理 GitHub 访问与编译 |

---

## 七、快速恢复环境的步骤（如果重装）

```bash
# 1. 依赖
apt-get install -y build-essential clang lld libc++-dev libc++abi-dev mono-devel \
  libx11-dev libxcb1-dev libgl1-mesa-dev libglu1-mesa-dev libxkbcommon-dev \
  libfontconfig1-dev libfreetype6-dev xvfb python3-opencv

# 2. GitHub 可达（DNS 被劫持时）
sed -i '/github\.com/d' /etc/hosts
echo "140.82.112.3  github.com" >> /etc/hosts
echo "140.82.112.3  codeload.github.com" >> /etc/hosts
echo "140.82.112.6  api.github.com" >> /etc/hosts

# 3. 源码（推荐 Windows 本机克隆后传，服务器直连太慢）
#    本机：git clone --depth 1 --branch 5.3 https://github.com/EpicGames/UnrealEngine.git
#    传输：tar -cf - --exclude=.git . | ssh root@server "tar -xf - -C /data/UnrealEngine"

# 4. 依赖（跳过 Setup.sh 的 git 检查）
cd /data/UnrealEngine/Engine
../Engine/Binaries/DotNET/GitDependencies/linux-x64/GitDependencies --prompt

# 5. 编译（必须带 SDK 环境变量）
SDK=/data/UnrealEngine/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v22_clang-16.0.6-centos7
export LINUX_MULTIARCH_ROOT="$SDK" CLANG="$SDK/x86_64-unknown-linux-gnu/bin/clang++" \
       CC="$SDK/x86_64-unknown-linux-gnu/bin/clang" CXX="$SDK/x86_64-unknown-linux-gnu/bin/clang++"
cd /data/UnrealEngine
./Engine/Build/BatchFiles/Linux/Build.sh UnrealEditor Linux Development -j20 -Progress
./Engine/Build/BatchFiles/Linux/Build.sh ShaderCompileWorker Linux Development -j20 -Progress
chown -R algo:algo /data/UnrealEngine

# 6. 工程
python3 -c "
import json
p='/data/avsynth/avsynth.uproject'
d=json.load(open(p,encoding='utf-8-sig'))
d['EngineAssociation']='/data/UnrealEngine'
json.dump(d,open(p,'w',encoding='utf-8'),ensure_ascii=False,indent='\t')
"
cd /data/UnrealEngine
./Engine/Build/BatchFiles/Linux/Build.sh avsynthEditor Linux Development \
  -Project=/data/avsynth/avsynth.uproject -Progress
chown -R algo:algo /data/avsynth
```

---

## 八、经验教训

1. **预编译引擎在这台机器上不可用**，必须源码编译。判断依据：`FMallocMimalloc::Malloc` 崩溃。
2. **`-RenderOffscreen` 不建立渲染线程**，SceneCapture 会出黑图，必须配 xvfb。
3. **UE 5.3 的 Python API 与文档/旧版本差异很大**，改代码前先用 `__doc__` 探测真实签名，不要照抄。
4. **`Setup.sh` 报 SUCCESS 不代表工具链配好了**，编译前必须验证 UBT 能识别平台。
5. **服务器访问 GitHub 极慢且不稳定**，大文件走「本机下载 → 传服务器」比服务器直连快 50 倍。
6. **判断「完成了」必须验证产物**，不能只看进程退出码（今天因 `GitDependencies` 中途退出误判，白等 1 小时）。
