import unreal, json, traceback

r = {"meshes": [], "stage": "start"}

try:
    r["stage"] = "loaded_unreal"
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    r["stage"] = "got_les"
    ok = les.load_level("/Game/Maps/codex_validation")
    r["load_ok"] = bool(ok)
    r["stage"] = "level_loaded"

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = eas.get_all_level_actors()
    r["stage"] = "got_actors_%d" % len(actors)

    for a in actors:
        try:
            label = a.get_actor_label()
            comp = a.get_component_by_class(unreal.StaticMeshComponent)
            if comp is None:
                r["meshes"].append({"actor": label, "no_mesh": True})
                continue
            r["meshes"].append({
                "actor": label,
                "render_custom_depth": bool(comp.get_editor_property("render_custom_depth")),
                "stencil": int(comp.get_editor_property("custom_depth_stencil_value")),
            })
        except Exception as e:
            r["meshes"].append({"actor": "?", "err": str(e)[:80]})
    r["stage"] = "done"
except Exception:
    r["stage"] = "exception"
    r["traceback"] = traceback.format_exc()[:800]

with open("/tmp/cs2.json", "w") as f:
    json.dump(r, f, indent=1, ensure_ascii=False)
print("WROTE")
