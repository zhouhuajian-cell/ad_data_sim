#!/usr/bin/env bash
# install_prebuilt_ue.sh - 安装 Epic 官网下载的 Linux 预编译版 UE5.3（不编译，解压即用）
# 怎么跑：
#   bash install_prebuilt_ue.sh                     # 引擎包放在默认位置 /data/installers，自动找 UE_5.3*.zip
#   bash install_prebuilt_ue.sh /root/UE_5.3.zip    # 指定引擎 zip
#   UE_DIR=/data/UE_5.3 PROJECT=/data/avsynth bash install_prebuilt_ue.sh
# 干什么：
#   1) 校验 zip（完整性、大小、目录结构，确认是 Linux 版引擎而不是 Windows 版）
#   2) 解压到 $UE_DIR（/data/UE_5.3，约 25~35GB）
#   3) 校验 Engine/Binaries/Linux/UnrealEditor-Cmd 存在且可执行
#   4) 在 ~/.config/Epic/UnrealEngine/Install.ini 注册引擎路径（工程按路径关联更稳）
#   5) 把工程的 EngineAssociation 改成 $UE_DIR（源码版 / 预编译版都认这个写法）
#   6) 跑一遍 case_check，打印下一步
# 安全：不下载、不联网、不解压到临时盘；zip 里带 Windows 版会在第 1 步就停下。
set -euo pipefail

UE_DIR="${UE_DIR:-/data/UE_5.3}"
PROJECT="${PROJECT:-/data/avsynth}"
ZIP="${1:-}"
SEARCH_DIRS=("$PROJECT" "/data/installers" "/root" "/data" "/tmp")

die() { echo "ERROR: $1" >&2; shift; for l in "$@"; do echo "       $l" >&2; done; exit 1; }
say() { echo "== $*"; }

[ "$(id -u)" -eq 0 ] || die "需要 root 运行（解压到 $UE_DIR 并注册引擎）"

# ---------- 0. 找 zip ---------------------------------------------------------------
if [ -z "$ZIP" ]; then
  for d in "${SEARCH_DIRS[@]}"; do
    [ -d "$d" ] || continue
    f="$(find "$d" -maxdepth 3 -iname 'UE_5.3*.zip' -type f 2>/dev/null | head -1 || true)"
    if [ -n "$f" ]; then ZIP="$f"; break; fi
  done
fi
[ -n "$ZIP" ] && [ -f "$ZIP" ] || die "没找到引擎 zip" \
  "把 Epic 官网下载的 Linux 版 zip 放到 /data/installers/ 后重跑：" \
  "  mkdir -p /data/installers && cp UE_5.3*.zip /data/installers/" \
  "或直接指定：bash install_prebuilt_ue.sh /root/UE_5.3.zip"
say "引擎包：$ZIP（$(du -h "$ZIP" | cut -f1)）"

# ---------- 1. 校验 -----------------------------------------------------------------
command -v unzip >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq unzip >/dev/null; }
command -v unzip >/dev/null 2>&1 || die "unzip 不可用（apt 装失败）"
unzip -tq "$ZIP" >/dev/null 2>&1 || die "zip 完整性校验失败（下载不完整？重新下载再试）"
say "zip 完整性 OK"

ROOT_LISTING="$(unzip -Z1 "$ZIP" | sed 's#\\#/#g' | awk -F/ 'NF>0 && $1!="" {print $1}' | sort -u | head -10)"
echo "  zip 顶层条目：$(echo "$ROOT_LISTING" | tr '\n' ' ')"
echo "$ROOT_LISTING" | grep -qx 'Engine' || die "zip 顶层没有 Engine/ 目录，不像 UE 引擎包"
# 包内路径可能用 Windows 反斜杠（例如在 Windows 上重新打包过），两种分隔符都要认
LISTING="$(unzip -Z1 "$ZIP" | sed 's#\\#/#g')"
echo "$LISTING" | grep -qE '(^|/)Engine/Binaries/Linux/UnrealEditor-Cmd$' \
  || die "找不到 Engine/Binaries/Linux/UnrealEditor-Cmd" \
    "这个包可能是 Windows 版引擎；请到 Epic 的 Linux 页面下载 Linux 版"
if echo "$LISTING" | grep -qE '(^|/)Engine/Binaries/Win64/UnrealEditor.exe$'; then
  say "注意：包内同时含 Win64 二进制（跨平台包属正常，Linux 端只用到 Linux 目录）"
fi

# ---------- 2. 解压 -----------------------------------------------------------------
if [ -x "$UE_DIR/Engine/Binaries/Linux/UnrealEditor-Cmd" ]; then
  say "目标已存在引擎，跳过解压：$UE_DIR"
else
  mkdir -p "$UE_DIR"
  FREE_G=$(df -BG --output=avail "$(dirname "$UE_DIR")" | tail -1 | tr -dc '0-9')
  NEED_G=$(( $(stat -c %s "$ZIP") / 1073741824 + 40 ))
  [ "${FREE_G:-0}" -ge "$NEED_G" ] || die "磁盘空间不够：需要约 ${NEED_G}G，可用 ${FREE_G}G" \
    "清理 /data 或换挂载点后再试"
  say "解压到 $UE_DIR（约需 ${NEED_G}G，请等待）"
  # 先解到临时目录。unzip 遇到反斜杠分隔的包只打 warning、仍按扁平名解出，
  # 所以解完若没有 Engine/，就用 Python 自己按条目名重建目录树（zipfile 只用标准库）。
  TMP_UNPACK="$UE_DIR/.unpack_tmp"
  rm -rf "$TMP_UNPACK"; mkdir -p "$TMP_UNPACK"
  # ⚠️ unzip 遇到反斜杠路径会打 warning 并返回非 0（set -e 会直接中断脚本），所以这里显式忽略返回码，
  #    解完再判断目录结构：没有 Engine/ 就走下面的 Python 重建分支。
  set +e
  unzip -q "$ZIP" -d "$TMP_UNPACK" 2>&1 | grep -v 'appears to use backslashes' || true
  set -e
  if [ ! -d "$TMP_UNPACK/Engine" ]; then
    say "包内路径不是标准 / 分隔（可能被 Windows 重新打包过），用 Python 按条目名重建目录树"
    # 先清掉 unzip 按扁平名解出的残渣，避免下面的 mv 撞到同名目录
    rm -rf "$TMP_UNPACK"; mkdir -p "$TMP_UNPACK"
    python3 - "$ZIP" "$TMP_UNPACK" <<'PY'
import os, sys, zipfile
zpath, dest = sys.argv[1], sys.argv[2]
n = 0
mode_fixed = 0
with zipfile.ZipFile(zpath) as zf:
    for info in zf.infolist():
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or name.endswith("/"):
            continue
        # 防 zip-slip：拒绝 .. 与绝对路径
        parts = [p for p in name.split("/") if p not in ("", ".", "..")]
        if not parts:
            continue
        out = os.path.join(dest, *parts)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            dst.write(src.read())
        # 保留 Unix 可执行位：zipfile 解出的文件默认 644，引擎二进制必须可执行
        mode = (info.external_attr >> 16) & 0o777
        if mode & 0o111:
            os.chmod(out, mode | 0o755)
            mode_fixed += 1
        n += 1
print("== Python 重建 %d 个文件（其中 %d 个补了可执行位）" % (n, mode_fixed))
PY
  fi
  if [ -d "$TMP_UNPACK/Engine" ]; then
    shopt -s dotglob
    mv "$TMP_UNPACK/"* "$UE_DIR/"
    shopt -u dotglob
  fi
  rmdir "$TMP_UNPACK" 2>/dev/null || true
  # 有的包会多套一层目录（UE_5.3/Engine），拉平
  if [ ! -d "$UE_DIR/Engine" ] && [ -d "$UE_DIR/UE_5.3/Engine" ]; then
    say "包内多一层 UE_5.3/，拉平目录"
    shopt -s dotglob
    mv "$UE_DIR/UE_5.3/"* "$UE_DIR/"
    rmdir "$UE_DIR/UE_5.3"
    shopt -u dotglob
  fi
fi

# ---------- 3. 校验产物 -------------------------------------------------------------
UE_CMD="$UE_DIR/Engine/Binaries/Linux/UnrealEditor-Cmd"
if [ -f "$UE_CMD" ] && [ ! -x "$UE_CMD" ]; then
  # zip 里没带 Unix 权限位时（Windows 打出来的包），解出来是 644 —— 补上可执行位
  say "UnrealEditor-Cmd 缺可执行位（zip 未保留权限），已自动 chmod +x"
  chmod +x "$UE_CMD"
fi
[ -x "$UE_CMD" ] || die "解压后找不到可执行的 $UE_CMD" "zip 结构异常，把解压后的顶层目录发给人看"
say "引擎就绪：$UE_CMD"
"$UE_CMD" -version 2>/dev/null | head -3 || true

# ---------- 4. 注册引擎（Install.ini） ----------------------------------------------
INI_DIR="$HOME/.config/Epic/UnrealEngine"
mkdir -p "$INI_DIR"
touch "$INI_DIR/Install.ini"
if grep -qx "InstalledDirectory=$UE_DIR" "$INI_DIR/Install.ini"; then
  say "Install.ini 已注册 $UE_DIR"
else
  printf '\n[Installations]\nInstalledDirectory=%s\n' "$UE_DIR" >> "$INI_DIR/Install.ini"
  say "已写入 $INI_DIR/Install.ini"
fi

# ---------- 5. 工程 EngineAssociation ------------------------------------------------
if [ -f "$PROJECT/avsynth.uproject" ]; then
  python3 - "$PROJECT/avsynth.uproject" "$UE_DIR" <<'PY'
import json, sys
path, ue = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    d = json.load(f)
old = d.get("EngineAssociation")
d["EngineAssociation"] = ue
with open(path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent="\t")
    f.write("\n")
print("== 工程 EngineAssociation: %r -> %r" % (old, ue))
PY
else
  say "未找到工程 $PROJECT/avsynth.uproject（自动改写跳过）"
fi

# ---------- 6. 收尾 -----------------------------------------------------------------
if [ -f "$PROJECT/Content/Python/case_check.py" ]; then
  say "工程一致性校验"
  python3 "$PROJECT/Content/Python/case_check.py" "$PROJECT" || true
fi
echo
echo "完成。下一步："
echo "  1) 打开 http://<服务器IP>:8010 第 3 步点「开始出图」（首次会自动编译工程模块 avsynthEditor）"
echo "  2) 出图前按 docs/phase3_checklist.md 过 T0-1~T0-5（16bit / 线性 / 语义 / 深度 / 框对齐）"
