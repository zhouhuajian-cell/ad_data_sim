#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""walk_cache.py — 遍历/仿真结果缓存（SQLite）

解决两个问题：
  ① 浏览器打开卡：一次只返回 N 张，靠 /api/tsr/gallery 分页
  ② 重复遍历：已经检测过/仿真过的图片直接读缓存，秒出结果

表结构：
  scan_cache   源图路径 → 检测结果 JSON（避免重复跑 YOLO）
  render_cache 源图路径 + 参数指纹 → 产出图路径（避免重复渲染）
  walk_log     每次遍历的统计（增量续跑用）

缓存键：
  scan_cache   : src_path
  render_cache : src_path + param_hash（参数变了要重渲）
"""
import hashlib
import json
import os
import sqlite3
import time

DB_PATH = os.environ.get("WALK_CACHE_DB", "/data/avsynth/state/walk_cache.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("""CREATE TABLE IF NOT EXISTS scan_cache (
        src        TEXT PRIMARY KEY,
        mtime      REAL,
        n_objects  INTEGER,
        detections TEXT,
        created_at REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS render_cache (
        key        TEXT PRIMARY KEY,
        src        TEXT,
        param_hash TEXT,
        out_path   TEXT,
        cls        TEXT,
        created_at REAL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_render_src ON render_cache(src)")
    c.execute("""CREATE TABLE IF NOT EXISTS walk_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        root       TEXT,
        country    TEXT,
        scanned    INTEGER,
        produced   INTEGER,
        started_at REAL,
        ended_at   REAL
    )""")
    c.commit()
    return c


def param_hash(params):
    """参数指纹：只对影响渲染的字段做 hash。"""
    keys = ["overcover", "black_p5", "white_p95", "color_blend",
            "blur_sigma", "jpeg_quality", "ellipse_fit", "safe_erase",
            "double_gate"]
    d = {k: params.get(k) for k in keys if k in params}
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# ---------------- 扫描缓存 ----------------

def get_scan(src):
    """返回缓存的检测结果；源图改动过则失效。"""
    try:
        st = os.stat(src)
    except OSError:
        return None
    c = _conn()
    try:
        row = c.execute(
            "SELECT mtime, detections FROM scan_cache WHERE src=?", (src,)).fetchone()
        if not row:
            return None
        mtime, det_json = row
        if abs(mtime - st.st_mtime) > 1e-3:
            return None          # 文件变了，缓存作废
        try:
            return json.loads(det_json)
        except Exception:
            return None
    finally:
        c.close()


def put_scan(src, detections):
    try:
        st = os.stat(src)
    except OSError:
        return
    c = _conn()
    try:
        c.execute(
            "INSERT OR REPLACE INTO scan_cache(src,mtime,n_objects,detections,created_at) "
            "VALUES(?,?,?,?,?)",
            (src, st.st_mtime, len(detections),
             json.dumps(detections, ensure_ascii=False), time.time()))
        c.commit()
    finally:
        c.close()


# ---------------- 渲染缓存 ----------------

def get_render(src, params):
    """返回已渲染的产出图路径（参数一致且文件还在）。"""
    ph = param_hash(params)
    c = _conn()
    try:
        row = c.execute(
            "SELECT out_path FROM render_cache WHERE src=? AND param_hash=?",
            (src, ph)).fetchone()
        if not row:
            return None
        out = row[0]
        return out if os.path.isfile(out) else None
    finally:
        c.close()


def put_render(src, params, out_path, cls=""):
    ph = param_hash(params)
    c = _conn()
    try:
        c.execute(
            "INSERT OR REPLACE INTO render_cache(key,src,param_hash,out_path,cls,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (src + "|" + ph, src, ph, out_path, cls, time.time()))
        c.commit()
    finally:
        c.close()


# ---------------- 统计 ----------------

def stats():
    c = _conn()
    try:
        n_scan = c.execute("SELECT COUNT(*) FROM scan_cache").fetchone()[0]
        n_render = c.execute("SELECT COUNT(*) FROM render_cache").fetchone()[0]
        return {"scanned": n_scan, "rendered": n_render, "db": DB_PATH}
    finally:
        c.close()


def log_walk(root, country, scanned, produced, started, ended=None):
    c = _conn()
    try:
        c.execute(
            "INSERT INTO walk_log(root,country,scanned,produced,started_at,ended_at) "
            "VALUES(?,?,?,?,?,?)",
            (root, country, scanned, produced, started, ended or time.time()))
        c.commit()
    finally:
        c.close()


if __name__ == "__main__":
    print(json.dumps(stats(), ensure_ascii=False, indent=1))
