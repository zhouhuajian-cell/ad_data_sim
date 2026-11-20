#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""walk_index.py — 按国家 + 图片名的增量遍历索引（SQLite）

设计：
  以「国家」为分区，以「图片文件名（不含目录）」为唯一键。
  同一张图（文件名相同）无论从哪个目录扫到，只处理一次。

表结构：
  walk_index
    country    国家代码（BR / AR / ...）
    img_name   图片文件名（含扩展名，全局唯一键）
    src_path   完整源路径
    status     pending | rendering | done | skip_no_sign | skip_small | fail
    out_path   产出图路径
    cls        YOLO 检测到的原牌类别
    conf       置信度
    bbox       检测框 JSON
    material   用的素材文件名
    sim_cls    仿真后类别
    created_at / updated_at

增量逻辑：
  · 遍历时对每张图先查 walk_index：
      - status=done 且产出文件存在 → 直接跳过（不跑 YOLO、不渲染）
      - 其他状态 → 重新处理
  · 处理完立刻写回索引
"""
import json
import os
import sqlite3
import time

DB_PATH = os.environ.get("WALK_INDEX_DB", "/data/avsynth/state/walk_index.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("""CREATE TABLE IF NOT EXISTS walk_index (
        country     TEXT NOT NULL,
        img_name    TEXT NOT NULL,
        src_path    TEXT,
        status      TEXT DEFAULT 'pending',
        out_path    TEXT,
        cls         TEXT,
        conf        REAL,
        bbox        TEXT,
        material    TEXT,
        sim_cls     TEXT,
        created_at  REAL,
        updated_at  REAL,
        PRIMARY KEY (country, img_name)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_wi_status ON walk_index(country, status)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_wi_path ON walk_index(src_path)")
    c.commit()
    return c


def is_done(country, img_name):
    """已经处理完成且产出文件还在？"""
    c = _conn()
    try:
        row = c.execute(
            "SELECT status, out_path FROM walk_index WHERE country=? AND img_name=?",
            (country, img_name)).fetchone()
        if not row:
            return False
        status, out_path = row
        if status != "done":
            return False
        return bool(out_path) and os.path.isfile(out_path)
    finally:
        c.close()


def get(country, img_name):
    c = _conn()
    try:
        row = c.execute(
            "SELECT src_path, status, out_path, cls, conf, bbox, material, sim_cls "
            "FROM walk_index WHERE country=? AND img_name=?",
            (country, img_name)).fetchone()
        if not row:
            return None
        keys = ["src_path", "status", "out_path", "cls", "conf", "bbox",
                "material", "sim_cls"]
        d = dict(zip(keys, row))
        if d.get("bbox"):
            try:
                d["bbox"] = json.loads(d["bbox"])
            except Exception:
                pass
        return d
    finally:
        c.close()


def mark(country, img_name, status, **kw):
    """写入/更新一条索引。"""
    now = time.time()
    c = _conn()
    try:
        row = c.execute(
            "SELECT created_at FROM walk_index WHERE country=? AND img_name=?",
            (country, img_name)).fetchone()
        created = row[0] if row else now
        bbox = kw.get("bbox")
        if bbox is not None and not isinstance(bbox, str):
            bbox = json.dumps(bbox, ensure_ascii=False)
        c.execute(
            "INSERT OR REPLACE INTO walk_index"
            "(country,img_name,src_path,status,out_path,cls,conf,bbox,material,sim_cls,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (country, img_name,
             kw.get("src_path", ""), status,
             kw.get("out_path", ""), kw.get("cls", ""),
             kw.get("conf"), bbox,
             kw.get("material", ""), kw.get("sim_cls", ""),
             created, now))
        c.commit()
    finally:
        c.close()


def stats(country=None):
    c = _conn()
    try:
        if country:
            rows = c.execute(
                "SELECT status, COUNT(*) FROM walk_index WHERE country=? GROUP BY status",
                (country,)).fetchall()
        else:
            rows = c.execute(
                "SELECT country, status, COUNT(*) FROM walk_index GROUP BY country, status"
            ).fetchall()
        return [{"country": r[0] if not country else country,
                 "status": r[1] if not country else r[0],
                 "count": r[2] if not country else r[1]} for r in rows]
    finally:
        c.close()


def missing_outputs(country=None):
    """status=done 但产出文件不存在 → 需要重跑。"""
    c = _conn()
    try:
        q = "SELECT country, img_name, out_path FROM walk_index WHERE status='done'"
        args = ()
        if country:
            q += " AND country=?"
            args = (country,)
        bad = []
        for co, nm, op in c.execute(q, args):
            if not op or not os.path.isfile(op):
                bad.append((co, nm))
        return bad
    finally:
        c.close()


if __name__ == "__main__":
    print(json.dumps(stats(), ensure_ascii=False, indent=1))
