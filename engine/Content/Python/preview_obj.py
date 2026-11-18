"""Render three OBJ or GND samples using the installed server project."""
import json
import os
import sys
from pathlib import Path

import unreal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "modules")]
import batch_capture as bc
from obj import spawn_object
from gnd import spawn_line


def ground_lines(actors, workzone=False):
    specs = []
    layout = [
        ("road_edge_white", (200, -320), 1700, 0),
        ("lane_dashed_white", (200, 0), 1700, 0),
        ("lane_solid_yellow", (200, 320), 1700, 0),
        ("stop_line", (1350, -300), 600, 90),
    ]
    if workzone:
        layout = [
            ("road_edge_white", (200, -365), 1700, 0),
            ("lane_dashed_white", (200, -120), 1700, 0),
            ("lane_dashed_white", (200, 120), 1700, 0),
        ]
    for kind, start, length, yaw in layout:
        segments = spawn_line(unreal, actors, kind, start, length, yaw=yaw)
        if not segments:
            raise RuntimeError("No segments spawned for " + kind)
        for segment in segments:
            # The installed GND helper uses a positional Rotator constructor.
            # Set the intended horizontal rotation explicitly for this layout.
            segment.set_actor_rotation(unreal.Rotator(pitch=0, yaw=yaw, roll=0), False)
            center = segment.get_actor_location()
            size = segment.get_actor_scale3d()
            specs.append({
                "class": kind, "instance_id": len(specs) + 1,
                "center": [center.x, center.y, center.z],
                "size": [size.x * 100, size.y * 100, size.z * 100],
                "yaw": segment.get_actor_rotation().yaw,
            })
    return specs


def main():
    output = Path(os.environ["OBJ_SMOKE_OUTPUT"])
    module = os.environ.get("UE_SMOKE_MODULE", "obj")
    sample = os.environ.get("UE_SMOKE_SAMPLE", "traffic")
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not levels.load_level("/Game/Maps/codex_validation"):
        raise RuntimeError("Cannot load validation map")
    prefixes = ("OBJ_", "TrafficSignProxy", "VehicleProxy", "ConeProxy")
    if module == "gnd" or sample == "workzone":
        prefixes += ("GND_", "LaneMark")
    for actor in actors.get_all_level_actors():
        if actor.get_actor_label().startswith(prefixes):
            if not actors.destroy_actor(actor):
                raise RuntimeError("Unable to clear previous demo object")

    sun = actors.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000),
                                       unreal.Rotator(pitch=-40, yaw=-30, roll=0))
    light = sun.get_component_by_class(unreal.DirectionalLightComponent)
    light.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    light.set_editor_property("intensity", 10.0)
    sky = actors.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1000))
    light = sky.get_component_by_class(unreal.SkyLightComponent)
    light.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    light.set_editor_property("intensity", 1.0)
    light.set_editor_property("real_time_capture", True)

    objects = []
    obj_layout = (
        ("car", (1250, -160, 0)),
        ("pedestrian", (1075, 175, 0)),
        ("cone", (750, 50, 0)),
    )
    if sample == "truck":
        obj_layout = (("truck", (1400, -80, 0)),)
    if sample == "workzone":
        obj_layout = (
            ("truck", (1450, 320, 0)),
            ("cone", (1130, 265, 176)),
            ("cone", (1130, 365, 176)),
            ("cone", (850, 223, 0)),
        )
    for index, (kind, position) in enumerate(obj_layout if module == "obj" else (), 1):
        parts = spawn_object(unreal, actors, kind, position)
        if not parts:
            raise RuntimeError("No actors spawned for " + kind)
        bounds = [part.get_actor_bounds(False) for part in parts]
        lower = [min(getattr(c, axis) - getattr(e, axis) for c, e in bounds) for axis in "xyz"]
        upper = [max(getattr(c, axis) + getattr(e, axis) for c, e in bounds) for axis in "xyz"]
        objects.append({
            "class": "vehicle" if kind in ("car", "truck") else kind,
            "instance_id": index,
            "center": [(a + b) / 2 for a, b in zip(lower, upper)],
            "size": [b - a for a, b in zip(lower, upper)],
            "yaw": 0,
        })
    if module == "gnd":
        objects = ground_lines(actors)
    if sample == "workzone":
        for line in ground_lines(actors, workzone=True):
            line["instance_id"] = len(objects) + 1
            objects.append(line)

    camera = actors.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-700, 0, 190))
    capture = bc._scene_capture_component(camera)
    target = bc._create_render_target("obj_smoke", 960, 540, "RTF_RGBA16F")
    source, _ = bc._resolve_capture_source("rgb")
    capture.set_editor_property("texture_target", target)
    capture.set_editor_property("capture_source", source)
    capture.set_editor_property("fov_angle", 65.0)
    capture.set_editor_property("always_persist_rendering_state", True)
    samples = ((-700, 0, 190), (-250, -230, 205), (100, 200, 220))
    pitch = -8 if sample == "workzone" else (-12 if module == "gnd" else -3)
    for number, location in enumerate(samples, 1):
        camera.set_actor_location(unreal.Vector(*location), False, False)
        camera.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=0, roll=0), False)
        for _ in range(8):
            capture.capture_scene()
        prefix = "obj_" + sample if module == "obj" and sample != "traffic" else module
        frame = "%s_%04d" % (prefix, number)
        raw = Path(bc._save_render_target(target, str(output / (frame + "_rgb.png"))))
        with raw.open("rb") as stream:
            signature = stream.read(8)
        if signature.startswith(b"\x76\x2f\x31\x01"):
            actual = raw.with_name(frame + "_rgb.exr")
        elif signature == b"\x89PNG\r\n\x1a\n":
            actual = raw.with_name(frame + "_rgb.png")
        else:
            raise RuntimeError("Unknown RGB image encoding")
        if actual != raw:
            raw.replace(actual)
        state = bc.parse_camera_spec({"location": location, "rotation": [pitch, 0, 0],
                                      "width": 960, "height": 540, "hfov_deg": 65})
        truth = bc.build_manifest({"frame_id": frame, "lighting": {"condition": "front_light"}},
                                  state, objects, bit_depth=bc.png_header_bit_depth(str(actual)) or 16)
        seen = {ob["class"] for ob in truth["objects"]}
        expected = {ob["class"] for ob in objects}
        if seen != expected:
            raise RuntimeError("Missing annotations in %s: %s" % (frame, sorted(expected - seen)))
        for item in truth["objects"]:
            item["bbox_3d"][:6] = [value / 100 for value in item["bbox_3d"][:6]]
            item["distance"] /= 100
        truth["image_files"] = {"rgb": actual.name, "bit_depth": truth["image_files"]["bit_depth"]}
        (output / (frame + ".json")).write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (output / "capture.ok").touch()


main()
