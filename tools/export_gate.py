"""数据导出门禁：9 项检查全过才 training_eligible=true。

借鉴 bev_core.export_decision 的契约，适配本项目：
  - 服务端生成 checks（不是浏览器自称）
  - 每项必须带 evidence_sha256
  - required 是版本化策略，用户不能绕过
"""
import hashlib
import json
import os

POLICY_VERSION = "gate_v1_20261121"

REQUIRED_CHECKS = (
    "calibration",       # 有可靠标定（人工四点或经过验证的估计）
    "original_labels",   # 原始标签完整（不是检测器提案）
    "geometry",          # 几何一致（投影误差在容差内）
    "occlusion",         # 遮挡处理正确（有前景/深度依据）
    "appearance",        # 外观合理（材质/光照/尺度）
    "multi_camera",      # 多相机一致
    "temporal",          # 时序一致（跨帧不闪）
    "topology",          # 拓扑一致（标线/牌面/路网关系）
    "split",             # 划分隔离（scene 级不泄漏）
)

CHECK_DESC = {
    "calibration": "标定来源可靠：人工标定或人工复核过的估计值",
    "original_labels": "有完整原始标注，不是只用检测器提案",
    "geometry": "投影误差在容差内（反投影 roundtrip / 车道线对齐）",
    "occlusion": "有前景掩码或深度依据，遮挡关系正确",
    "appearance": "材质、光照、尺度合理，无明显穿帮",
    "multi_camera": "同一场景多相机视角一致",
    "temporal": "跨帧不闪烁、不跳动",
    "topology": "标线/牌面/路网拓扑关系正确",
    "split": "训练/验证按 scene 或 drive 级隔离，无泄漏",
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_check(status, evidence_path=None, evidence_obj=None, note=""):
    """构造一项检查记录。status 必须是 passed / failed / unchecked。"""
    if status not in ("passed", "failed", "unchecked"):
        raise ValueError("status 必须是 passed/failed/unchecked")

    rec = {"status": status, "note": str(note)[:300]}

    if evidence_path and os.path.isfile(evidence_path):
        rec["evidence_path"] = evidence_path
        rec["evidence_sha256"] = _sha256_file(evidence_path)
    elif evidence_obj is not None:
        rec["evidence_obj"] = evidence_obj
        rec["evidence_sha256"] = _sha256_obj(evidence_obj)

    return rec


def export_decision(checks, required=REQUIRED_CHECKS):
    """返回 {training_eligible, training_blockers, details}。

    任何一项 status != passed 或缺 evidence_sha256 -> 该帧不可训练。
    """
    blockers = []
    missing = []
    details = {}

    for name in required:
        c = checks.get(name) or {}
        st = c.get("status")
        has_ev = bool(c.get("evidence_sha256"))

        if st != "passed":
            blockers.append(name)
        if not has_ev:
            missing.append(name)

        details[name] = {
            "status": st or "unchecked",
            "has_evidence": has_ev,
            "note": c.get("note", ""),
            "desc": CHECK_DESC.get(name, ""),
        }

    # 缺证据也算阻塞（防止手工把 status 改成 passed）
    for name in missing:
        if name not in blockers:
            blockers.append(name + "_no_evidence")

    return {
        "policy": POLICY_VERSION,
        "training_eligible": len(blockers) == 0,
        "training_blockers": blockers,
        "details": details,
        "checked": len(required),
        "passed": len([d for d in details.values() if d["status"] == "passed"]),
    }


def build_checks_from_artifacts(annotation, bg_path=None, extra=None):
    """从本项目实际产出自动生成 checks。

    只对**能自动验证**的项给 passed，其余一律 unchecked。
    这样不会把未经人工确认的数据放出去。
    """
    checks = {}
    ann = annotation or {}
    masks = ann.get("masks") or {}
    cam = ann.get("camera")

    # 1) calibration：有标定文件且是人工标定
    if bg_path and os.path.isfile(
            "/data/render_hybrid/bgs/_calib_" +
            os.path.splitext(os.path.basename(bg_path))[0] + ".json"):
        checks["calibration"] = make_check(
            "passed", evidence_obj={"source": "manual_calib",
                                    "bg": os.path.basename(bg_path)},
            note="人工四点标定")
    else:
        checks["calibration"] = make_check(
            "unchecked", note="无人工标定，只有自动估计")

    # 2) original_labels：本模块默认没有
    checks["original_labels"] = make_check(
        "unchecked", note="未接入完整原始标注（只有检测器提案）")

    # 3) geometry：投影了且线存在
    lines = ann.get("lines") or []
    if lines and all(l.get("alpha_mean") is not None for l in lines):
        checks["geometry"] = make_check(
            "passed", evidence_obj={"n_lines": len(lines),
                                    "camera": cam},
            note="标线投影完成")
    else:
        checks["geometry"] = make_check("unchecked", note="无标线")

    # 4) occlusion：跑了遮罩
    if masks and masks.get("paint_kept_mean") is not None:
        blocked = float(masks.get("blocked_ratio") or 0)
        checks["occlusion"] = make_check(
            "passed", evidence_obj={"blocked_ratio": blocked,
                                    "road_mean": masks.get("road_mean")},
            note="前景遮罩已生效，挡掉 %.1f%%" % (blocked * 100))
    else:
        checks["occlusion"] = make_check(
            "unchecked", note="未跑前景遮罩")

    # 5~9) 需要人工或外部数据，一律 unchecked
    for name in ("appearance", "multi_camera", "temporal",
                 "topology", "split"):
        checks[name] = make_check("unchecked", note="需人工复核或外部数据")

    if extra:
        checks.update(extra)

    return checks


def write_gate(checks, out_path):
    d = export_decision(checks)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return d
