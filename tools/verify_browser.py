import io

P = "/opt/ue5_synth/dashboard/index.html"
s = io.open(P, encoding="utf-8").read()

if "VERIFY-BROWSER" in s:
    print("已存在")
    raise SystemExit(0)

JS = r'''

/* ===== VERIFY-BROWSER：浏览已跑好的批量验证产出 ===== */
(function(){
  'use strict';

  function $(id){ return document.getElementById(id); }

  var V = { data: null, idx: 0, zoom: 1 };

  function build(){
    if ($('vfyPanel')) return $('vfyPanel');
    var d = document.createElement('div');
    d.id = 'vfyPanel';
    d.style.cssText = 'display:none;position:fixed;inset:0;z-index:85;' +
      'background:#0B0E14;flex-direction:column';
    d.innerHTML =
      '<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;' +
        'background:#11151C;border-bottom:1px solid #1F2630;flex:0 0 auto">' +
        '<span style="color:#EAB308;font-size:12px;font-weight:600">' +
          '\u6279\u91cf\u9a8c\u8bc1\u4ea7\u51fa</span>' +
        '<select id="vfyClip" style="background:#161B24;border:1px solid #2A323D;' +
          'color:#8B95A5;border-radius:4px;padding:3px 8px;font-size:11px;' +
          'max-width:340px"><option>\u52a0\u8f7d\u4e2d\u2026</option></select>' +
        '<span id="vfyInfo" style="color:#5C6675;font-size:10.5px"></span>' +
        '<span style="flex:1"></span>' +
        '<button class="vfyZ" data-z="1" style="background:#1F6FEB;color:#fff;' +
          'border:0;border-radius:4px;padding:3px 10px;font-size:10.5px;' +
          'cursor:pointer">1x</button>' +
        '<button class="vfyZ" data-z="2" style="background:#161B24;color:#8B95A5;' +
          'border:1px solid #2A323D;border-radius:4px;padding:3px 10px;' +
          'font-size:10.5px;cursor:pointer;margin-left:4px">2x</button>' +
        '<button id="vfyClose" style="background:transparent;color:#8B95A5;' +
          'border:1px solid #2A323D;border-radius:4px;padding:3px 12px;' +
          'font-size:10.5px;cursor:pointer;margin-left:8px">\u5173\u95ed</button>' +
      '</div>' +
      '<div style="flex:1;min-height:0;display:flex">' +
        '<div style="width:340px;flex:0 0 auto;overflow-y:auto;' +
          'background:#0E1219;border-right:1px solid #1F2630">' +
          '<div style="padding:6px 12px;border-bottom:1px solid #1F2630;' +
            'color:#D8DEE9;font-size:11.5px;font-weight:600">\u95e8\u7981\u6c47\u603b</div>' +
          '<div id="vfySum" style="padding:6px 12px;font-size:10.5px"></div>' +
        '</div>' +
        '<div style="flex:1;min-width:0;display:flex;flex-direction:column">' +
          '<div style="flex:1;min-height:0;display:flex;align-items:center;' +
            'justify-content:center;overflow:hidden;background:#000">' +
            '<img id="vfyImg" style="max-width:100%;max-height:100%;' +
              'object-fit:contain">' +
          '</div>' +
          '<div id="vfyList" style="display:flex;gap:6px;padding:6px 12px;' +
            'overflow-x:auto;background:#0E1219;border-top:1px solid #1F2630;' +
            'flex:0 0 auto;min-height:70px"></div>' +
        '</div>' +
      '</div>';
    document.body.appendChild(d);

    $('vfyClose').onclick = function(){ d.style.display = 'none'; };
    $('vfyClip').onchange = function(){ renderClip(this.value); };
    Array.prototype.forEach.call(d.querySelectorAll('.vfyZ'), function(b){
      b.onclick = function(){
        V.zoom = parseFloat(b.dataset.z);
        Array.prototype.forEach.call(d.querySelectorAll('.vfyZ'), function(x){
          x.style.background = '#161B24';
          x.style.color = '#8B95A5';
          x.style.border = '1px solid #2A323D';
        });
        b.style.background = '#1F6FEB';
        b.style.color = '#fff';
        b.style.border = '0';
        var im = $('vfyImg');
        im.style.transform = 'scale(' + V.zoom + ')';
      };
    });
    return d;
  }

  function renderClip(clipName){
    if (!V.data) return;
    var sum = V.data.summary || [];
    var imgs = V.data.images || [];

    // 门禁汇总
    var rows = sum.map(function(x){
      var col = x.eligible ? '#22C55E' : (x.passed >= 7 ? '#EAB308' : '#EF4444');
      return '<div style="display:flex;gap:6px;padding:2px 0;' +
        'border-bottom:1px solid #161B24">' +
        '<span style="color:' + col + ';font-family:monospace;min-width:30px">' +
          (x.passed || 0) + '/9</span>' +
        '<span style="color:#8B95A5;flex:1;overflow:hidden;' +
          'text-overflow:ellipsis;white-space:nowrap">' + x.clip + '</span>' +
        '<span style="color:#5C6675">' + (x.n || 0) + '\u5e27</span></div>';
    }).join('');
    var avg = sum.length ? (sum.reduce(function(a, b){
      return a + (b.passed || 0);
    }, 0) / sum.length).toFixed(1) : '--';
    $('vfySum').innerHTML =
      '<div style="color:#8B95A5;margin-bottom:6px">\u5e73\u5747 <b style="color:#EAB308">' +
      avg + '/9</b> \u00b7 \u5171 ' + sum.length + ' clip</div>' + rows;

    // 缩略图
    var sel = imgs.filter(function(p){
      return p.indexOf(clipName) >= 0;
    });
    if (!sel.length) sel = imgs.slice(0, 20);

    $('vfyList').innerHTML = sel.map(function(p, i){
      var src = '/api/real/verify_image?path=' + encodeURIComponent(p) +
                '&t=' + Date.now();
      return '<div class="vfyTh" data-i="' + i + '" data-p="' + p + '" ' +
        'style="flex:0 0 96px;height:56px;cursor:pointer;border:2px solid ' +
        (i === 0 ? '#EAB308' : 'transparent') + ';border-radius:4px;' +
        'overflow:hidden;background:#111827">' +
        '<img loading="lazy" style="width:100%;height:100%;object-fit:cover;' +
        'display:block" src="' + src + '"></div>';
    }).join('');

    Array.prototype.forEach.call($('vfyList').querySelectorAll('.vfyTh'), function(el){
      el.onclick = function(){
        Array.prototype.forEach.call($('vfyList').querySelectorAll('.vfyTh'), function(x){
          x.style.border = '2px solid transparent';
        });
        el.style.border = '2px solid #EAB308';
        var p = el.dataset.p;
        $('vfyImg').src = '/api/real/verify_image?path=' +
          encodeURIComponent(p) + '&t=' + Date.now();
        $('vfyInfo').textContent = p;
      };
    });

    if (sel.length) {
      $('vfyImg').src = '/api/real/verify_image?path=' +
        encodeURIComponent(sel[0]) + '&t=' + Date.now();
      $('vfyInfo').textContent = sel[0];
    } else {
      $('vfyImg').src = '';
      $('vfyInfo').textContent = '\u65e0\u4ea7\u51fa';
    }
  }

  async function open(){
    var p = build();
    p.style.display = 'flex';
    try {
      var r = await fetch('/api/real/verify', {cache:'no-store'});
      if (!r.ok) {
        $('vfySum').innerHTML = '<span style="color:#EF4444">' +
          '\u8fd8\u6ca1\u8dd1\u8fc7\u6279\u91cf\u9a8c\u8bc1</span>';
        return;
      }
      V.data = await r.json();
      var sum = V.data.summary || [];
      if (!sum.length) {
        $('vfySum').innerHTML = '<span style="color:#5C6675">\u65e0\u6570\u636e</span>';
        return;
      }
      var sel = $('vfyClip');
      sel.innerHTML = sum.map(function(x){
        return '<option value="' + x.clip + '">' + x.clip +
               ' (' + x.passed + '/9)</option>';
      }).join('');
      renderClip(sum[0].clip);
    } catch(e) {
      $('vfySum').innerHTML = '<span style="color:#EF4444">' + e.message + '</span>';
    }
  }

  window.__openVerify = open;

  // 按钮
  function addBtn(){
    if ($('btnVerify')) return;
    var b = document.createElement('button');
    b.id = 'btnVerify';
    b.innerHTML = '\u{1F4CA} \u9a8c\u8bc1\u4ea7\u51fa';
    b.style.cssText = 'position:fixed;right:262px;bottom:18px;z-index:60;' +
      'display:inline-flex;align-items:center;gap:6px;padding:8px 14px;' +
      'border-radius:8px;background:#0E1219;border:1px solid #2A323D;' +
      'color:#EAB308;font-size:12px;font-weight:500;cursor:pointer;' +
      'box-shadow:0 4px 14px rgba(0,0,0,.5);transition:all .15s';
    b.onmouseenter = function(){ b.style.background = '#161B24';
                                 b.style.borderColor = '#EAB308'; };
    b.onmouseleave = function(){ b.style.background = '#0E1219';
                                 b.style.borderColor = '#2A323D'; };
    b.onclick = open;
    document.body.appendChild(b);
  }

  setInterval(addBtn, 1500);
  setTimeout(addBtn, 800);
  console.log('[VERIFY-BROWSER] ready');
})();
'''

idx = s.rfind("</script>")
s = s[:idx] + JS + "\n" + s[idx:]
io.open(P, "w", encoding="utf-8").write(s)
print("已注入 VERIFY-BROWSER")
