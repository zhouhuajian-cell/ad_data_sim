import io

P = "/opt/ue5_synth/dashboard/index.html"
s = io.open(P, encoding="utf-8").read()

if "CLIP-PLAYER" in s:
    print("已存在")
    raise SystemExit(0)

JS = r'''

/* ===== CLIP-PLAYER：仿真台内双栏连续播放（原图 | 仿真） ===== */
(function(){
  'use strict';

  var P = {
    module: 'obj', batch: '', clip: '',
    frames: [], idx: 0, playing: false, fps: 6,
    timer: null, loaded: {}, bats: []
  };

  function $(id){ return document.getElementById(id); }

  /* ---------- 建视图 ---------- */
  function ensureView(){
    var id = 'view-clip';
    var v = $(id);
    if (v) return v;
    var host = document.querySelector('main') ||
               $('view-workbench')?.parentElement;
    if (!host) return null;

    v = document.createElement('div');
    v.id = id;
    v.className = 'view-pane';
    v.style.cssText = 'display:none;flex:1;min-height:0;flex-direction:column;' +
      'overflow:hidden;background:#0B0E14';

    v.innerHTML =
      // 工具栏
      '<div style="display:flex;align-items:center;gap:8px;padding:6px 12px;' +
        'background:#11151C;border-bottom:1px solid #1F2630;flex:0 0 auto">' +
        '<span id="cpMod" style="font-size:11px;font-weight:600;' +
          'font-family:monospace;color:#22C55E">OBJ</span>' +
        '<select id="cpBatch" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:3px 8px;font-size:11px;' +
          'max-width:300px"><option>加载中…</option></select>' +
        '<select id="cpClip" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:3px 8px;font-size:11px;' +
          'max-width:230px"><option>--</option></select>' +
        '<span id="cpCount" style="color:#5C6675;font-size:10.5px"></span>' +
        '<span style="flex:1"></span>' +
        '<button id="cpPrev" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:3px 10px;font-size:11px;' +
          'cursor:pointer">◀</button>' +
        '<button id="cpPlay" style="background:#1F6FEB;color:#fff;border:0;' +
          'border-radius:4px;padding:3px 16px;font-size:11px;cursor:pointer">' +
          '▶ 播放</button>' +
        '<button id="cpNext" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:3px 10px;font-size:11px;' +
          'cursor:pointer">▶</button>' +
        '<span id="cpPos" style="color:#5C6675;font-size:10.5px;' +
          'min-width:60px;text-align:center">0/0</span>' +
        '<span style="color:#3B4048;font-size:10px">速度</span>' +
        '<select id="cpFps" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:2px 6px;font-size:10.5px">' +
          '<option value="3">3fps</option>' +
          '<option value="6" selected>6fps</option>' +
          '<option value="10">10fps</option>' +
          '<option value="15">15fps</option></select>' +
      '</div>' +
      // 双栏
      '<div style="flex:1;min-height:0;display:flex;background:#000">' +
        '<div style="flex:1;min-width:0;display:flex;flex-direction:column;' +
          'border-right:1px solid #1F2630">' +
          '<div style="padding:3px 10px;background:#0E1219;color:#8B95A5;' +
            'font-size:10px;border-bottom:1px solid #1F2630;display:flex;' +
            'align-items:center;gap:8px">' +
            '<span>原图 (Original)</span><span style="flex:1"></span>' +
            '<button id="cpToggleBBox" style="background:#1F6FEB;color:#fff;' +
              'border:0;border-radius:3px;padding:1px 8px;font-size:9.5px;' +
              'cursor:pointer">BBox</button></div>' +
          '<div id="cpOrigWrap" style="flex:1;min-height:0;display:flex;' +
            'align-items:center;justify-content:center;overflow:hidden;' +
            'position:relative">' +
            '<img id="cpOrig" style="max-width:100%;max-height:100%;' +
              'object-fit:contain">' +
            '<canvas id="cpCv" style="position:absolute;left:0;top:0;' +
              'pointer-events:none"></canvas>' +
          '</div>' +
        '</div>' +
        '<div style="flex:1;min-width:0;display:flex;flex-direction:column">' +
          '<div id="cpSimTitle" style="padding:3px 10px;background:#0E1219;' +
            'color:#22C55E;font-size:10px;border-bottom:1px solid #1F2630">' +
            '仿真 (Synthetic)</div>' +
          '<div style="flex:1;min-height:0;display:flex;align-items:center;' +
            'justify-content:center;overflow:hidden">' +
            '<img id="cpSim" style="max-width:100%;max-height:100%;' +
              'object-fit:contain">' +
          '</div>' +
        '</div>' +
      '</div>' +
      // 进度条
      '<div style="flex:0 0 auto;padding:4px 12px;background:#0E1219;' +
        'border-top:1px solid #1F2630;display:flex;align-items:center;gap:10px">' +
        '<input id="cpSeek" type="range" min="0" max="0" value="0" ' +
          'style="flex:1;height:4px;accent-color:#1F6FEB;cursor:pointer">' +
        '<span id="cpStem" style="color:#3B4048;font-size:9.5px;' +
          'font-family:monospace;min-width:130px">--</span>' +
      '</div>';

    host.appendChild(v);
    bind(v);
    return v;
  }

  /* ---------- 播放控制 ---------- */
  function showFrame(i){
    if (!P.frames.length) return;
    i = Math.max(0, Math.min(i, P.frames.length - 1));
    P.idx = i;
    var f = P.frames[i];

    var oi = $('cpOrig');
    var si = $('cpSim');
    // 预加载下一帧
    preload(i + 1);

    oi.src = f.orig + '&t=' + P.loaded.t;
    si.src = f.sim + '&t=' + P.loaded.t;

    oi.onload = function(){ drawBBox(); };
    $('cpPos').textContent = (i + 1) + '/' + P.frames.length;
    $('cpSeek').value = i;
    $('cpStem').textContent = f.stem;
  }

  function preload(i){
    if (i >= P.frames.length) return;
    var f = P.frames[i];
    var a = new Image(); a.src = f.orig + '&t=' + P.loaded.t;
    var b = new Image(); b.src = f.sim + '&t=' + P.loaded.t;
  }

  function play(){
    if (P.playing) { pause(); return; }
    if (!P.frames.length) return;
    P.playing = true;
    $('cpPlay').textContent = '⏸ 暂停';
    $('cpPlay').style.background = '#EAB308';
    $('cpPlay').style.color = '#000';
    var ms = Math.round(1000 / P.fps);
    P.timer = setInterval(function(){
      var n = P.idx + 1;
      if (n >= P.frames.length) n = 0;
      showFrame(n);
    }, ms);
  }

  function pause(){
    P.playing = false;
    if (P.timer) { clearInterval(P.timer); P.timer = null; }
    $('cpPlay').textContent = '▶ 播放';
    $('cpPlay').style.background = '#1F6FEB';
    $('cpPlay').style.color = '#fff';
  }

  /* ---------- 标注框 ---------- */
  function drawBBox(){
    var im = $('cpOrig');
    var cv = $('cpCv');
    if (!im || !cv || !im.naturalWidth) return;
    var wrap = $('cpOrigWrap');
    var wr = wrap.getBoundingClientRect();
    var ir = im.getBoundingClientRect();
    cv.style.left = (ir.left - wr.left) + 'px';
    cv.style.top = (ir.top - wr.top) + 'px';
    cv.style.width = ir.width + 'px';
    cv.style.height = ir.height + 'px';
    cv.width = Math.round(ir.width);
    cv.height = Math.round(ir.height);

    var ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!P.showBBox || !P.ann) return;

    var sx = cv.width / im.naturalWidth;
    var sy = cv.height / im.naturalHeight;
    (P.ann.objects || []).forEach(function(o){
      if (!o.in_front || !o.bbox2d) return;
      var b = o.bbox2d;
      var x0 = b[0] * sx, y0 = b[1] * sy;
      var x1 = b[2] * sx, y1 = b[3] * sy;
      if (x1 < 0 || x0 > cv.width || y1 < 0 || y0 > cv.height) return;
      var col = ({
        car: '#F59E0B', truck: '#EF4444', van: '#F97316',
        pedestrian: '#3B82F6', rider: '#A855F7', three: '#22C55E',
        special: '#EC4899', bus: '#14B8A6'
      })[o.kind] || '#8B95A5';
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      var txt = o.kind + ' ' + Math.round(o.depth_m || 0) + 'm';
      ctx.font = 'bold 11px monospace';
      var tw = ctx.measureText(txt).width;
      ctx.fillStyle = col;
      ctx.fillRect(x0, Math.max(y0 - 15, 0), tw + 6, 15);
      ctx.fillStyle = '#000';
      ctx.fillText(txt, x0 + 3, Math.max(y0 - 4, 11));
    });
  }

  function loadAnn(stem){
    if (!P.batch || !P.clip) return;
    fetch('/api/real/labels?batch=' + encodeURIComponent(P.batch) +
          '&clip=' + encodeURIComponent(P.clip) +
          '&stem=' + encodeURIComponent(stem), {cache:'no-store'})
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(d){ P.ann = d; drawBBox(); })
      .catch(function(){ P.ann = null; });
  }

  /* ---------- 绑定 ---------- */
  function bind(v){
    $('cpPlay').onclick = play;
    $('cpPrev').onclick = function(){ pause(); showFrame(P.idx - 1); };
    $('cpNext').onclick = function(){ pause(); showFrame(P.idx + 1); };
    $('cpFps').onchange = function(){
      P.fps = parseInt(this.value, 10) || 6;
      if (P.playing) { pause(); play(); }
    };
    $('cpSeek').oninput = function(){
      pause();
      showFrame(parseInt(this.value, 10) || 0);
    };
    $('cpBatch').onchange = function(){ loadClips(this.value); };
    $('cpClip').onchange = function(){ loadFrames(P.batch, this.value); };
    $('cpToggleBBox').onclick = function(){
      P.showBBox = !P.showBBox;
      this.style.background = P.showBBox ? '#1F6FEB' : '#161B24';
      this.style.color = P.showBBox ? '#fff' : '#8B95A5';
      drawBBox();
    };
    P.showBBox = true;
    window.addEventListener('resize', drawBBox);
  }

  function loadClips(batch){
    P.batch = batch;
    fetch('/api/real/batch?batch=' + encodeURIComponent(batch),
          {cache:'no-store'})
      .then(function(r){ return r.json(); })
      .then(function(d){
        var cs = d.clips || [];
        var sel = $('cpClip');
        sel.innerHTML = cs.map(function(c){
          return '<option value="' + c.clip + '">' + c.clip +
                 ' (' + c.frames + '帧)</option>';
        }).join('');
        if (cs.length) loadFrames(batch, cs[0].clip);
      });
  }

  function loadFrames(batch, clip){
    pause();
    P.batch = batch; P.clip = clip;
    P.loaded.t = Date.now();
    fetch('/api/real/clip_play?batch=' + encodeURIComponent(batch) +
          '&clip=' + encodeURIComponent(clip) + '&module=' + P.module,
          {cache:'no-store'})
      .then(function(r){ return r.json(); })
      .then(function(d){
        P.frames = d.frames || [];
        $('cpCount').textContent = P.frames.length + ' 帧可播放';
        var seek = $('cpSeek');
        seek.max = Math.max(0, P.frames.length - 1);
        seek.value = 0;
        if (P.frames.length) showFrame(0); else {
          $('cpOrig').src = ''; $('cpSim').src = '';
          $('cpPos').textContent = '0/0';
        }
      });
  }

  /* ---------- 打开 ---------- */
  async function open(module){
    P.module = (module || 'obj').toLowerCase();
    var v = ensureView();
    if (!v) return;
    Array.prototype.forEach.call(document.querySelectorAll('.view-pane'), function(p){
      p.style.display = (p.id === 'view-clip') ? 'flex' : 'none';
    });
    $('cpMod').textContent = P.module.toUpperCase();
    $('cpMod').style.color = P.module === 'gnd' ? '#EAB308' : '#22C55E';
    $('cpSimTitle').style.color = P.module === 'gnd' ? '#EAB308' : '#22C55E';
    $('cpSimTitle').textContent = (P.module === 'gnd' ? 'GND 仿真' : 'OBJ 仿真') +
                                   ' (Synthetic)';

    if (!P.bats.length) {
      try {
        var r = await fetch('/api/real/clips', {cache:'no-store'});
        var d = await r.json();
        P.bats = d.clips || [];
        $('cpBatch').innerHTML = P.bats.map(function(c){
          return '<option value="' + c.name + '">' + c.name + '</option>';
        }).join('');
        // 优先选能放的包
        var pref = P.bats.find(function(c){
          return c.name.indexOf('extremely_closetruck') >= 0;
        }) || P.bats[0];
        if (pref) { $('cpBatch').value = pref.name; loadClips(pref.name); }
      } catch(e) { console.warn(e); }
    } else if (P.batch) {
      loadFrames(P.batch, P.clip || '');
    }
  }

  window.__openClipPlayer = open;
  window.__clipPlayer = P;
  console.log('[CLIP-PLAYER] ready');
})();
'''

idx = s.rfind("</script>")
s = s[:idx] + JS + "\n" + s[idx:]
io.open(P, "w", encoding="utf-8").write(s)
print("已注入 CLIP-PLAYER")
