import sys
sys.path.insert(0, "/opt/ue5_synth/tools")
from scene_fusion import SceneFusionTSRSimulator
print("import OK")
import inspect
print("方法:", [m for m in dir(SceneFusionTSRSimulator) if not m.startswith("_")])
