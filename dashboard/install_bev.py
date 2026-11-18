"""Add the BEV entry to the dashboard that already serves OBJ and GND."""
from datetime import datetime
from pathlib import Path
import sys


dashboard = Path(sys.argv[1])
server_file = dashboard / "server.py"
server = server_file.read_text(encoding="utf-8")
index_file = dashboard / "index.html"
index = index_file.read_text(encoding="utf-8")


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Missing or duplicate BEV deployment anchor: " + old)
    return source.replace(old, new, 1)


updated = replace(server, 'if u.path in ("/obj", "/gnd"):', 'if u.path in ("/bev", "/obj", "/gnd"):')
updated = replace(updated, '''    return {"job": job, "busy": busy,
            "result": max(completed, default=(0, None))[1]}''', '''    result = max(completed, default=(0, None))[1]
    is_truck = result and mode == "obj" and os.path.isfile(
        os.path.join(SMOKE_ROOT, result, "obj_truck_0001.json"))
    return {"job": job, "busy": busy, "result": result,
            "result_sample": "truck" if is_truck else "traffic"}''')
updated = replace(updated,
    '        return ["runuser", "-u", "algo", "--", "bash", SMOKE_SCRIPT, action.split("_")[0]]',
    '''        mode = action.split("_")[0]
        sample = p.get("sample", "traffic")
        if sample != "traffic" and (mode != "obj" or sample != "truck"):
            raise ValueError("不支持的小样类型")
        return ["runuser", "-u", "algo", "--", "bash", SMOKE_SCRIPT, mode, sample]''')
updated = replace(updated, '''        if smoke:
            with SMOKE_LAUNCH_LOCK:
                rec, error = start_job(smoke.group(1) + "_smoke", {})''', '''        if smoke:
            payload = self._body()
            if not isinstance(payload, dict):
                return self._send(400, {"error": "小样参数须为 JSON 对象"})
            with SMOKE_LAUNCH_LOCK:
                rec, error = start_job(smoke.group(1) + "_smoke", payload)''')
new_index = replace(index, '<a href="/obj" style="color:#76dfdb">OBJ 交通目标</a>',
                    '<a href="/bev" style="color:#76dfdb">BEV · OBJ 交通目标</a>')
new_index = replace(new_index, '<a href="/gnd" style="color:#76dfdb">GND 地面标线</a>',
                    '<a href="/bev?module=gnd" style="color:#76dfdb">BEV · GND 地面标线</a>')

compile(updated, str(server_file), "exec")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for path, original, replacement in ((server_file, server, updated), (index_file, index, new_index)):
    path.with_name(path.name + ".bak.bev." + stamp).write_text(original, encoding="utf-8")
    pending = path.with_suffix(path.suffix + ".bev-tmp")
    pending.write_text(replacement, encoding="utf-8")
    pending.chmod(path.stat().st_mode & 0o777)
    pending.replace(path)
