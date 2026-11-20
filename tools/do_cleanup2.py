"""第二轮清理：废弃 HTML、jobs 日志、旧备份。"""
import datetime
import os
import shutil

BASE = "/opt/ue5_synth"
DASH = os.path.join(BASE, "dashboard")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TRASH = os.path.join(BASE, "_trash_" + STAMP)

# server.py 里没有任何路由引用的
DEAD_HTML = ["perceptionsim.html", "simple.html"]

# 有路由但已被 index.html 完全取代的（确认后可删）
OLD_HTML = ["pro.html", "tsr.html"]

# 保留的
KEEP_HTML = ["index.html", "gallery.html", "obj_new.html", "obj.html"]

moved = {}


def move_tree(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src.rstrip("/"))
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        dst = dst + "_dup"
    shutil.move(src, dst)
    return dst


def main():
    os.makedirs(TRASH, exist_ok=True)

    # ---------- 1) 废弃 HTML ----------
    hd = os.path.join(TRASH, "dashboard")
    cnt = 0
    for f in DEAD_HTML + OLD_HTML:
        p = os.path.join(DASH, f)
        if os.path.isfile(p):
            move_ok = move_tree(p, hd)
            print("  HTML 移走: %-24s %8d 字节" % (f, os.path.getsize(move_ok)))
            cnt += 1
    moved["html"] = cnt

    # ---------- 2) gallery.html 备份 ----------
    for f in os.listdir(DASH):
        if f.startswith("gallery.html.") or f.startswith("index.html."):
            p = os.path.join(DASH, f)
            move_tree(p, hd)
            cnt += 1
    moved["backup"] = cnt - moved["html"]

    # ---------- 3) jobs 日志（旧任务记录） ----------
    jobs = os.path.join(DASH, "jobs")
    if os.path.isdir(jobs):
        n = len(os.listdir(jobs))
        move_ok = move_tree(jobs, TRASH)
        print("  jobs 移走: %d 个文件 → %s" % (n, move_ok))
        moved["jobs"] = n

    # ---------- 4) logs ----------
    logs = os.path.join(DASH, "logs")
    if os.path.isdir(logs):
        n = len(os.listdir(logs))
        move_tree(logs, TRASH)
        moved["logs"] = n

    # ---------- 5) post 目录下的旧备份 ----------
    post = os.path.join(BASE, "post")
    if os.path.isdir(post):
        for f in os.listdir(post):
            if ".bak" in f:
                move_tree(os.path.join(post, f), os.path.join(TRASH, "post"))

    print("\n" + "=" * 60)
    print("第二轮清理完成 → " + TRASH)
    print("=" * 60)
    print("废弃 HTML:   %d" % moved.get("html", 0))
    print("HTML 备份:   %d" % moved.get("backup", 0))
    print("jobs 日志:   %d 个" % moved.get("jobs", 0))
    print("logs:        %d 个" % moved.get("logs", 0))

    print("\n剩余 dashboard/*.html:")
    for f in sorted(os.listdir(DASH)):
        if f.endswith(".html"):
            print("  %-22s %8d 字节" % (f, os.path.getsize(os.path.join(DASH, f))))


if __name__ == "__main__":
    main()
