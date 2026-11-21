import io

P = "/opt/ue5_synth/dashboard/index.html"
s = io.open(P, encoding="utf-8").read()

if "MODULE-PARAMS" in s:
    print("已存在")
    raise SystemExit(0)

JS = r'''

/* ===== MODULE-PARAMS：OBJ/GND 右侧参数面板（对齐 TSR） ===== */
(function(){
  'use strict';

  var P = {
    obj: {
      count: 3,           // 物体数量
      cls: 'cone',        // 类别
      dist_min: 8,        // 最近距离
      dist_max: 28,       // 最远距离
      lat: 3.0,           // 横向范围 ±米
      scale: 1.0,         // 尺寸倍率
      shadow: 0.35,       // 阴影强度
      occlusion: true,    // 遮挡处理
      light: 'auto',      // day/night/auto
    },
    gnd: {
      lanes: 3,           // 车道数
      dash: true,         // 中间虚线
      stop: false,        // 停止线
      edge: 0.20,         // 边线宽度 m
      width: 0.15,        // 虚线宽度 m
      wear: 0.25,         // 磨损
      opacity: 1.0,       // 不透明度
      occlusion: true,
    },
  };

  function $(id){ return document.getElementById(id); }

  function buildPanel(){
    if ($('modParams')) return $('modParams');

    var v = $('view-clip');
    if (!v) return null;

    var d = document.createElement('div');
    d.id = 'modParams';
    d.style.cssText = 'width:250px;flex:0 0 auto;display:flex;' +
      'flex-direction:column;overflow-y:auto;background:#0E1219;' +
      'border-left:1px solid #1F2630';

    // 插入到双栏右边：找 view-clip 里的双栏容器
    d.innerHTML = '<div id="mpBody" style="padding:8px 10px"></div>';
    v.appendChild(d);
    return d;
  }

  function slider(key, label, mod, min, max, step, unit, fmt){
    var val = P[mod][key];
    return '<div style="margin-bottom:9px">' +
      '<div style="display:flex;justify-content:space-between;' +
        'color:#8B95A5;font-size:10.5px;margin-bottom:3px">' +
        '<span>' + label + '</span>' +
        '<span id="pv_' + mod + '_' + key + '" style="color:#58A6FF;' +
          'font-family:monospace">' + (fmt ? fmt(val) : val) + '</span>' +
      '</div>' +
      '<input type="range" class="mpSlider" data-mod="' + mod + '" ' +
        'data-key="' + key + '" min="' + min + '" max="' + max + '" ' +
        'step="' + step + '" value="' + val + '" ' +
        'style="width:100%;height:3px;accent-color:#1F6FEB;cursor:pointer">' +
      '</div>';
  }

  function toggle(key, label, mod){
    var val = P[mod][key];
    return '<div style="display:flex;justify-content:space-between;' +
      'align-items:center;margin-bottom:7px">' +
      '<span style="color:#8B95A5;font-size:10.5px">' + label + '</span>' +
      '<input type="checkbox" class="mpToggle" data-mod="' + mod + '" ' +
        'data-key="' + key + '" ' + (val ? 'checked' : '') +
        ' style="accent-color:#1F6FEB;cursor:pointer">' +
      '</div>';
  }

  function render(){
    var body = $('mpBody');
    if (!body) return;
    var mod = (window.__clipPlayer || {}).module || 'obj';

    if (mod === 'obj') {
      body.innerHTML =
        '<div style="color:#D8DEE9;font-size:11px;font-weight:600;' +
          'margin-bottom:8px;padding-bottom:5px;' +
          'border-bottom:1px solid #1F2630">OBJ 参数</div>' +
        '<div style="margin-bottom:9px">' +
          '<div style="color:#8B95A5;font-size:10.5px;margin-bottom:3px">' +
            '物体类别</div>' +
          '<select id="mpObjCls" style="width:100%;background:#161B24;' +
            'border:1px solid #2A323D;color:#8B95A5;border-radius:4px;' +
            'padding:3px 6px;font-size:10.5px">' +
            ['cone', 'car', 'suv', 'truck', 'truck_flat', 'van',
             'pedestrian', 'rider'].map(function(c){
              return '<option ' + (c === P.obj.cls ? 'selected' : '') +
                     '>' + c + '</option>';
            }).join('') +
          '</select></div>' +
        slider('count', '数量', 'obj', 1, 12, 1, '个') +
        slider('dist_min', '最近距离', 'obj', 4, 40, 1, 'm',
               function(v){ return v + 'm'; }) +
        slider('dist_max', '最远距离', 'obj', 8, 80, 2, 'm',
               function(v){ return v + 'm'; }) +
        slider('lat', '横向范围', 'obj', 0.5, 8, 0.5, 'm',
               function(v){ return '±' + v + 'm'; }) +
        slider('scale', '尺寸倍率', 'obj', 0.5, 2.0, 0.05, '',
               function(v){ return v.toFixed(2) + 'x'; }) +
        slider('shadow', '阴影强度', 'obj', 0, 1, 0.05, '',
               function(v){ return Math.round(v * 100) + '%'; }) +
        toggle('occlusion', '遮挡裁剪', 'obj') +
        '<div style="color:#8B95A5;font-size:10.5px;margin:8px 0 3px">光照</div>' +
        '<select id="mpObjLight" style="width:100%;background:#161B24;' +
          'border:1px solid #2A323D;color:#8B95A5;border-radius:4px;' +
          'padding:3px 6px;font-size:10.5px;margin-bottom:10px">' +
          ['auto', 'day', 'night'].map(function(c){
            return '<option ' + (c === P.obj.light ? 'selected' : '') +
                   '>' + c + '</option>';
          }).join('') +
        '</select>' +
        '<button id="mpApply" style="width:100%;background:#22C55E;' +
          'color:#000;border:0;border-radius:5px;padding:7px;' +
          'font-size:11px;font-weight:600;cursor:pointer">' +
          '应用参数并重绘</button>' +
        '<div id="mpStatus" style="color:#5C6675;font-size:10px;' +
          'margin-top:6px;text-align:center"></div>';
    } else {
      body.innerHTML =
        '<div style="color:#D8DEE9;font-size:11px;font-weight:600;' +
          'margin-bottom:8px;padding-bottom:5px;' +
          'border-bottom:1px solid #1F2630">GND 参数</div>' +
        slider('lanes', '车道数', 'gnd', 1, 5, 1, '条') +
        toggle('dash', '中间虚线', 'gnd') +
        toggle('stop', '停止线', 'gnd') +
        slider('edge', '边线宽度', 'gnd', 0.10, 0.40, 0.01, 'm',
               function(v){ return v.toFixed(2) + 'm'; }) +
        slider('width', '虚线宽度', 'gnd', 0.10, 0.30, 0.01, 'm',
               function(v){ return v.toFixed(2) + 'm'; }) +
        slider('wear', '磨损程度', 'gnd', 0, 1, 0.05, '',
               function(v){ return Math.round(v * 100) + '%'; }) +
        slider('opacity', '不透明度', 'gnd', 0.2, 1.0, 0.05, '',
               function(v){ return Math.round(v * 100) + '%'; }) +
        toggle('occlusion', '遮挡裁剪', 'gnd') +
        '<button id="mpApply" style="width:100%;background:#EAB308;' +
          'color:#000;border:0;border-radius:5px;padding:7px;' +
          'font-size:11px;font-weight:600;cursor:pointer">' +
          '应用参数并重绘</button>' +
        '<div id="mpStatus" style="color:#5C6675;font-size:10px;' +
          'margin-top:6px;text-align:center"></div>';
    }
    bind();
  }

  function bind(){
    // 滑块
    Array.prototype.forEach.call(
      document.querySelectorAll('#mpBody .mpSlider'), function(el){
      el.oninput = function(){
        var mod = el.dataset.mod, key = el.dataset.key;
        var v = parseFloat(el.value);
        P[mod][key] = v;
        var pv = $('pv_' + mod + '_' + key);
        if (pv) {
          var fmt = {
            dist_min: function(x){ return x + 'm'; },
            dist_max: function(x){ return x + 'm'; },
            lat: function(x){ return '±' + x + 'm'; },
            scale: function(x){ return x.toFixed(2) + 'x'; },
            shadow: function(x){ return Math.round(x * 100) + '%'; },
            edge: function(x){ return x.toFixed(2) + 'm'; },
            width: function(x){ return x.toFixed(2) + 'm'; },
            wear: function(x){ return Math.round(x * 100) + '%'; },
            opacity: function(x){ return Math.round(x * 100) + '%'; },
          }[key];
          pv.textContent = fmt ? fmt(v) : v;
        }
      };
    });

    // 开关
    Array.prototype.forEach.call(
      document.querySelectorAll('#mpBody .mpToggle'), function(el){
      el.onchange = function(){
        P[el.dataset.mod][el.dataset.key] = el.checked;
      };
    });

    var cls = $('mpObjCls');
    if (cls) cls.onchange = function(){ P.obj.cls = this.value; };
    var lt = $('mpObjLight');
    if (lt) lt.onchange = function(){ P.obj.light = this.value; };

    var ap = $('mpApply');
    if (ap) ap.onclick = apply;
  }

  async function apply(){
    var mod = (window.__clipPlayer || {}).module || 'obj';
    var p = (window.__clipPlayer || {});
    var batch = p.batch, clip = p.clip;
    if (!batch || !clip) {
      $('mpStatus').textContent = '先选数据包';
      return;
    }
    $('mpStatus').textContent = '生成中…';
    try {
      var r = await fetch('/api/real/render_params', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ module: mod, batch: batch, clip: clip,
                               params: P[mod] })
      });
      var d = await r.json();
      if (!d.ok) { $('mpStatus').textContent = '失败: ' + (d.error || ''); return; }
      $('mpStatus').textContent = '✓ ' + d.n + ' 帧已生成';
      // 刷新播放器
      if (window.__openClipPlayer) {
        var wasPlaying = p.playing;
        // 强制重载帧
        var ev = new Event('x');
        if (window.__reloadClipFrames) window.__reloadClipFrames();
      }
    } catch(e) {
      $('mpStatus').textContent = '失败: ' + e.message;
    }
  }

  // 模块切换时重绘面板
  var lastMod = null;
  setInterval(function(){
    var v = $('view-clip');
    if (!v || getComputedStyle(v).display === 'none') return;
    buildPanel();
    var mod = (window.__clipPlayer || {}).module;
    if (mod !== lastMod) {
      lastMod = mod;
      render();
    }
  }, 800);

  window.__modParams = P;
  console.log('[MODULE-PARAMS] ready');
})();
'''

idx = s.rfind("</script>")
s = s[:idx] + JS + "\n" + s[idx:]
io.open(P, "w", encoding="utf-8").write(s)
print("已注入 MODULE-PARAMS")
