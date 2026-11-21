import io

P = "/opt/ue5_synth/dashboard/index.html"
s = io.open(P, encoding="utf-8").read()

if "MODULE-TO-CLIP" in s:
    print("已存在")
    raise SystemExit(0)

# 顶部切到 OBJ/GND 时，直接打开 clip 播放器
OLD = """    // 切到对应视图：TSR -> view-workbench；OBJ/GND -> view-wb-*
    var targetId = (MOD === 'tsr') ? 'view-workbench' : ('view-wb-' + MOD);
    var target = document.getElementById(targetId);

    // 视图懒创建：不存在就先建
    if (!target && MOD !== 'tsr' && window.__ensureModuleView) {
      target = window.__ensureModuleView(MOD);
    }"""

NEW = """    // MODULE-TO-CLIP：OBJ/GND 直接进 clip 播放器（原图|仿真 连续播放）
    if (MOD !== 'tsr' && window.__openClipPlayer) {
      try {
        window.__openClipPlayer(MOD);
        window.__MODFILTER = MOD;
        if (window.__jtRender) window.__jtRender();
        console.log('[MODULE]', MOD.toUpperCase(), '-> clip player');
      } catch (e) {
        console.error('[MODULE] clip player 失败', e);
      }
      return;
    }

    // TSR：原有工作台
    var targetId = 'view-workbench';
    var target = document.getElementById(targetId);

    if (!target && window.__ensureModuleView) {
      target = window.__ensureModuleView('tsr');
    }"""

if OLD in s:
    s = s.replace(OLD, NEW, 1)
    print("1) OK OBJ/GND 直接进播放器")
else:
    print("1) MISS 模块切换")

io.open(P, "w", encoding="utf-8").write(s)
print("saved")
