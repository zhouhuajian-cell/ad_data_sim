import unreal, json
res = {}
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
les.load_level("/Game/Maps/TestLevel")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# 1) 方向光（太阳）
try:
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000), unreal.Rotator(-45, 0, 0))
    sun_comp = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_comp.set_editor_property("intensity", 10.0)
    res["sun"] = "ok"
except Exception as e:
    res["sun"] = "ERR " + str(e)

# 2) 天光
try:
    sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    sky_comp = sky.get_component_by_class(unreal.SkyLightComponent)
    sky_comp.set_editor_property("intensity", 1.0)
    sky_comp.set_editor_property("real_time_capture", True)
    res["skylight"] = "ok"
except Exception as e:
    res["skylight"] = "ERR " + str(e)

# 3) 天空大气（背景不黑）
try:
    eas.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
    res["atmosphere"] = "ok"
except Exception as e:
    res["atmosphere"] = "ERR " + str(e)

# 4) 指数高度雾（可选，让远处有层次）
try:
    eas.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0))
    res["fog"] = "ok"
except Exception as e:
    res["fog"] = "ERR " + str(e)

# 5) 保存
try:
    les.save_current_level()
    res["save"] = "ok"
except Exception as e:
    res["save"] = "ERR " + str(e)

with open('/tmp/fl.json', 'w') as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("DONE")
