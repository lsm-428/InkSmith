"""网页版对照：上传图 -> 三条路线出位图 + 指标（1-bit film 主线）。

用法：uv run python webapp.py [--port 8123] [--ckpt runs/onebit/best.pt]
浏览器打开 http://localhost:8123
标准库实现，无需新依赖。

接口：
  GET  /              页面
  POST /api/convert   {names[], images[dataURL], dpi?}  三路线对照
  POST /api/wedge     {dpi?, levels?}                   灰度尺诊断
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval import load_model_and_printer, psnr, triplet

MAX_SIDE = 512          # 单图最长边，控制显存与传输体积


# ---------------------------------------------------------------- 图像工具

def png_dataurl(arr, mode=None):
    """np/PIL -> dataURL PNG。"""
    if isinstance(arr, Image.Image):
        im = arr
    else:
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode(dataurl):
    b = dataurl.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b)))


def fit(img, max_side=MAX_SIDE):
    """限制最长边；返回 (图, 是否缩放过)。"""
    if max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)),
                         Image.LANCZOS)
        return img, True
    return img, False


# ---------------------------------------------------------------- 1-bit 路线

_CACHE = {}


def get_model(ckpt, dpi=None):
    key = (ckpt, dpi)
    if key not in _CACHE:
        _CACHE[key] = load_model_and_printer(ckpt, dpi=dpi)
    return _CACHE[key]


def _triplet(img, dpi, ckpt):
    global CKPT
    CKPT = ckpt
    return triplet(img, ckpt, dpi)


def _gray_mode(ckpt):
    """ckpt 是否 1-bit 灰度路线（决定 wedge/convert 是否可用）。"""
    _m, _p, a, _d = load_model_and_printer(ckpt)
    return not a.get("cmyk")


def process_one(name, dataurl, ckpt, dpi=None):
    """1-bit 灰度三路线对照（原有逻辑）。"""
    t0 = time.time()
    img, scaled = fit(decode(dataurl).convert("L"))
    inks, prints, target, a = _triplet(img, dpi, ckpt)
    ink2img = lambda ink: np.uint8((1 - ink[0, 0].cpu().numpy()) * 255)
    clip = lambda x: x[0, 0].cpu().numpy().clip(0, 1)
    return {
        "name": name, "size": [img.width, img.height], "scaled": scaled,
        "dpi": dpi or a.get("dpi"),
        "ms": round((time.time() - t0) * 1000),
        "printer": a.get("printer", "paper"),
        "ink": {"nn": round(inks["nn"].mean().item(), 4),
                "fs": round(inks["fs"].mean().item(), 4),
                "gs": round(inks["gs"].mean().item(), 4),
                "target": round(target.mean().item(), 4)},
        "psnr": {"nn": round(psnr(prints["nn"], target), 2),
                 "fs": round(psnr(prints["fs"], target), 2),
                 "gs": round(psnr(prints["gs"], target), 2),
                 "vs_fs": round(psnr(prints["nn"], target)
                                - psnr(prints["fs"], target), 2),
                 "vs_gs": round(psnr(prints["nn"], target)
                                - psnr(prints["gs"], target), 2)},
        "img": {
            "orig": png_dataurl(np.asarray(img, np.uint8)),
            "nn_bitmap": png_dataurl(ink2img(inks["nn"])),
            "nn_preview": png_dataurl(clip(prints["nn"])),
            "fs_bitmap": png_dataurl(ink2img(inks["fs"])),
            "fs_preview": png_dataurl(clip(prints["fs"])),
            "gs_bitmap": png_dataurl(ink2img(inks["gs"])),
            "gs_preview": png_dataurl(clip(prints["gs"])),
        },
    }


def wedge(n_levels=16, dpi=None, ckpt=None):
    """1-bit 灰度尺诊断：分段覆盖率偏差（模型最敏感的体检项）。"""
    t0 = time.time()
    n_levels = max(2, min(32, n_levels))
    seg = 96
    levels = np.arange(n_levels) / (n_levels - 1)
    arr = np.concatenate([np.full((seg, seg), l, np.float32) for l in levels],
                         axis=1)
    img = Image.fromarray((arr * 255).astype("uint8"), "L")
    inks, prints, target, a = _triplet(img, dpi, ckpt)
    dev = lambda t, k: t[0, 0, :, k * seg:(k + 1) * seg].mean().item()
    rows = [{"level": round(float(l), 3),
             "nn": round(dev(prints["nn"], k), 3),
             "fs": round(dev(prints["fs"], k), 3),
             "gs": round(dev(prints["gs"], k), 3)}
            for k, l in enumerate(levels)]
    for r in rows:
        for m in ("nn", "fs", "gs"):
            r[f"d_{m}"] = round(r[m] - r["level"], 3)
    g = lambda x: np.uint8((1 - x[0, 0].cpu().numpy()) * 255)
    return {
        "rows": rows, "dpi": dpi or a.get("dpi"),
        "ms": round((time.time() - t0) * 1000),
        "psnr": {"nn": round(psnr(prints["nn"], target), 2),
                 "fs": round(psnr(prints["fs"], target), 2),
                 "gs": round(psnr(prints["gs"], target), 2)},
        "img": {"orig": png_dataurl(np.asarray(img, np.uint8)),
                "nn": png_dataurl(g(inks["nn"])),
                "fs": png_dataurl(g(inks["fs"])),
                "gs": png_dataurl(g(inks["gs"]))},
    }


# ---------------------------------------------------------------- 页面

def page_html(ckpt):
    """渲染整页（暗色，单文件无外部依赖）。"""
    return GRAY_HTML.replace("__CKPT__", ckpt)


BASE_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--txt:#e6e9ef;--mut:#8b93a7;
--acc:#5b8cff;--good:#3ecf8e;--bad:#ff6b6b;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,#151922,#0f1115);position:sticky;top:0;z-index:9}
h1{margin:0;font-size:17px;letter-spacing:.3px}
h1 .tag{font-size:11px;font-weight:500;color:var(--mut);border:1px solid var(--line);
border-radius:20px;padding:2px 9px;margin-left:9px;vertical-align:2px}
.sub{color:var(--mut);font-size:12.5px;margin-top:5px}
main{padding:18px 22px;max-width:1280px;margin:0 auto}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
button,.btn{background:var(--acc);color:#fff;border:0;border-radius:7px;
padding:8px 15px;font-size:13.5px;cursor:pointer;font-weight:500}
button:hover{filter:brightness(1.12)}
button.ghost{background:#232834;color:var(--txt);border:1px solid var(--line)}
button.ghost:hover{background:#2b313e}
label.chk{color:var(--mut);font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer}
input[type=file]{color:var(--mut);font-size:13px;max-width:290px}
#st{color:var(--mut);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
padding:14px 16px;margin-bottom:16px}
.card h3{margin:0 0 10px;font-size:14.5px;display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.card h3 .meta{color:var(--mut);font-size:12px;font-weight:400}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:11px}
.chip{background:#1e222c;border:1px solid var(--line);border-radius:7px;
padding:5px 10px;font-size:12.5px;white-space:nowrap}
.chip b{font-size:14px;font-weight:600}
.chip .k{color:var(--mut);margin-right:5px;font-size:11.5px}
.good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:9px}
.fig{background:#0b0d11;border:1px solid var(--line);border-radius:8px;
overflow:hidden;position:relative}
.fig img{display:block;width:100%;image-rendering:pixelated;cursor:zoom-in}
.fig .cap{font-size:11.5px;color:var(--mut);padding:6px 8px;border-top:1px solid var(--line);
display:flex;justify-content:space-between;gap:6px}
.fig .cap a{color:var(--acc);text-decoration:none}
#zoom{position:fixed;inset:0;background:rgba(6,8,12,.93);display:none;
align-items:center;justify-content:center;z-index:50;cursor:zoom-out;padding:24px}
#zoom img{max-width:96vw;max-height:92vh;image-rendering:pixelated;
border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:center}
th{background:#1b1f28;color:var(--mut);font-weight:500}
.note{color:var(--mut);font-size:12px;margin:9px 0 0;padding-left:11px;border-left:2px solid var(--line)}
.big{font-size:22px;font-weight:650}
"""

GRAY_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>神经半色调 · 1-bit 对照</title><style>__CSS__</style></head><body>
<header><h1>神经半色调 · 1-bit 灰度<span class="tag" id="ck">__CKPT__</span></h1>
<div class="sub">模型位图 vs Floyd-Steinberg vs Ghostscript 筛网。film 模式下灰度=相对膜厚，基线自动反相对齐。</div></header>
<main>
<div class="bar">
  <input id="f" type="file" accept="image/*" multiple>
  <label class="chk">评估 dpi <input id="dpi" type="number" placeholder="ckpt值"
    style="width:78px;padding:5px 7px;background:#12151c;border:1px solid var(--line);
    border-radius:6px;color:var(--txt)"></label>
  <button onclick="go()">开始转换</button>
  <button class="ghost" onclick="toggleWedge()">灰度尺诊断</button>
  <span id="st"></span>
</div>
<div id="summary"></div><div id="wedge"></div><div id="out"></div>
</main>
<div id="zoom" onclick="this.style.display='none'"><img></div>
<script>
const $=id=>document.getElementById(id);
let LAST=[];
const dpiVal=()=>{const v=$('dpi').value.trim();return v?parseInt(v):null;};
async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
const zoom=src=>{const z=$('zoom');z.style.display='flex';z.querySelector('img').src=src;};
const dl=(url,n)=>{const a=document.createElement('a');a.href=url;a.download=n;
  document.body.appendChild(a);a.click();a.remove();};
function dlAll(){
  LAST.forEach(x=>{const n=x.name.replace(/\\.[^.]+$/,'');
    dl(x.img.nn_bitmap,n+'_nn_bitmap.png');dl(x.img.nn_preview,n+'_nn_preview.png');
    dl(x.img.fs_bitmap,n+'_fs_bitmap.png');dl(x.img.fs_preview,n+'_fs_preview.png');
    dl(x.img.gs_bitmap,n+'_gs_bitmap.png');dl(x.img.gs_preview,n+'_gs_preview.png');});
}
function fig(cap,src,name){
  return `<div class="fig"><img src="${src}" onclick="zoom('${src}')" alt="${cap}">
  <div class="cap"><span>${cap}</span>${name?`<a href="${src}" download="${name}">下载</a>`:''}</div></div>`;
}
async function toggleWedge(){
  const box=$('wedge');
  if(box.innerHTML){box.innerHTML='';return;}
  $('st').innerText='灰度尺诊断中...';
  try{
    const w=await post('/api/wedge',{dpi:dpiVal()});
    const c=v=>Math.abs(v)<=0.02?'':(v>0?'good':'bad');
    const s=v=>(v>=0?'+':'')+v.toFixed(3);
    const rows=w.rows.map(r=>`<tr><td>${r.level.toFixed(3)}</td>
      <td>${r.nn.toFixed(3)}</td><td class="${c(r.d_nn)}">${s(r.d_nn)}</td>
      <td>${r.fs.toFixed(3)}</td><td class="${c(r.d_fs)}">${s(r.d_fs)}</td>
      <td>${r.gs.toFixed(3)}</td><td class="${c(r.d_gs)}">${s(r.d_gs)}</td></tr>`).join('');
    const worst=w.rows.reduce((a,b)=>Math.abs(b.d_nn)>Math.abs(a.d_nn)?b:a);
    box.innerHTML=`<div class="card"><h3>灰度尺诊断<span class="meta">${w.rows.length} 阶${w.dpi?' @'+w.dpi+'dpi':''} · ${w.ms} ms · PSNR 模型 ${w.psnr.nn} / FS ${w.psnr.fs} / GS ${w.psnr.gs} dB</span></h3>
    <p class="note">偏差 = 实际覆盖率 − 目标覆盖率，|偏差|&gt;0.02 标色。模型最大偏差在 <b>${worst.level.toFixed(3)}</b> 灰阶（<span class="${worst.d_nn>0?'good':'bad'}">${s(worst.d_nn)}</span>）——判断灰阶映射是否畸变（S 形）最直接的指标。</p>
    <table><tr><th>灰阶</th><th colspan=2>模型</th><th colspan=2>FS</th><th colspan=2>GS</th></tr>
    <tr><th></th><th>覆盖</th><th>偏差</th><th>覆盖</th><th>偏差</th><th>覆盖</th><th>偏差</th></tr>${rows}</table>
    <div class="imgs">${[['原图',w.img.orig],['模型位图',w.img.nn],['FS位图',w.img.fs],['GS位图',w.img.gs]]
      .map(([a,b])=>fig(a,b)).join('')}</div></div>`;
    $('st').innerText='';
  }catch(e){$('st').innerText='失败: '+e.message;}
}
async function go(){
  const fs=$('f').files;
  if(!fs.length){$('st').innerText='请选择图片';return;}
  $('st').innerText='读取中...';$('out').innerHTML='';$('summary').innerHTML='';
  const imgs=[],names=[];
  for(const file of fs){
    names.push(file.name);
    imgs.push(await new Promise(r=>{const rd=new FileReader();
      rd.onload=()=>r(rd.result);rd.readAsDataURL(file);}));
  }
  $('st').innerText='转换中...';
  let data;
  try{data=await post('/api/convert',{names,images:imgs,dpi:dpiVal()});}
  catch(e){$('st').innerText='失败: '+e.message;return;}
  LAST=data;
  const t=data.reduce((a,d)=>a+d.ms,0);
  $('st').innerText=`完成 ${data.length} 张 · ${(t/1000).toFixed(1)} s`;
  $('out').innerHTML=data.map(card).join('');
  if(data.length>1){
    const m=k=>data.reduce((a,d)=>a+d.psnr[k],0)/data.length;
    const s=v=>(v>=0?'+':'')+v.toFixed(2);
    const c=v=>v>=0?'good':'bad';
    $('summary').innerHTML=`<div class="card"><h3>均值（${data.length} 张）</h3><div class="chips">
      <span class="chip"><span class="k">模型</span><b>${m('nn').toFixed(2)}</b> dB</span>
      <span class="chip"><span class="k">FS</span><b>${m('fs').toFixed(2)}</b></span>
      <span class="chip"><span class="k">GS</span><b>${m('gs').toFixed(2)}</b></span>
      <span class="chip ${c(m('vs_fs'))}"><span class="k">vs FS</span><b>${s(m('vs_fs'))}</b></span>
      <span class="chip ${c(m('vs_gs'))}"><span class="k">vs GS</span><b>${s(m('vs_gs'))}</b></span>
      </div><button class="ghost" onclick="dlAll()">下载全部位图</button></div>`;
  }
}
function card(d){
  const s=v=>(v>=0?'+':'')+v;
  const c=v=>v>=0?'good':'bad';
  const n=d.name.replace(/\\.[^.]+$/,'');
  return `<div class="card"><h3>${d.name}<span class="meta">${d.size[0]}×${d.size[1]}${d.dpi?' @'+d.dpi+'dpi':''}${d.printer?' ['+d.printer+']':''}${d.scaled?' · 已缩放':''} · ${d.ms} ms</span></h3>
  <div class="chips">
    <span class="chip"><span class="k">PSNR</span><b>${d.psnr.nn}</b></span>
    <span class="chip"><span class="k">FS</span><b>${d.psnr.fs}</b></span>
    <span class="chip"><span class="k">GS</span><b>${d.psnr.gs}</b></span>
    <span class="chip ${c(d.psnr.vs_fs)}"><span class="k">vs FS</span><b>${s(d.psnr.vs_fs)}</b></span>
    <span class="chip ${c(d.psnr.vs_gs)}"><span class="k">vs GS</span><b>${s(d.psnr.vs_gs)}</b></span>
    <span class="chip"><span class="k">墨量 模型/FS/GS/目标</span><b>${d.ink.nn}</b> / ${d.ink.fs} / ${d.ink.gs} / ${d.ink.target}</span>
  </div>
  <div class="imgs">
    ${fig('原图',d.img.orig)}
    ${fig('模型位图',d.img.nn_bitmap,n+'_nn_bitmap.png')}
    ${fig('模型模拟打印',d.img.nn_preview,n+'_nn_preview.png')}
    ${fig('FS 位图',d.img.fs_bitmap,n+'_fs_bitmap.png')}
    ${fig('FS 模拟打印',d.img.fs_preview,n+'_fs_preview.png')}
    ${fig('GS 位图',d.img.gs_bitmap,n+'_gs_bitmap.png')}
    ${fig('GS 模拟打印',d.img.gs_preview,n+'_gs_preview.png')}
  </div></div>`;
}
</script></body></html>""".replace("__CSS__", BASE_CSS)

# ---------------------------------------------------------------- HTTP

CKPT = None


class Handler:
    def __init__(self):
        from http.server import BaseHTTPRequestHandler

        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = page_html(CKPT).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, code, obj, ctype="application/json"):
                body = (json.dumps(obj, ensure_ascii=False)
                        if ctype.startswith("application/json")
                        else str(obj)).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8"
                                 if ctype.startswith("text/") else ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path not in ("/api/convert", "/api/wedge"):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    ln = int(self.headers["Content-Length"])
                    req = json.loads(self.rfile.read(ln))
                    if not _gray_mode(CKPT):
                        raise ValueError(
                            "CMYK 路线已于 2026-10-21 停止支持，请用 1-bit ckpt")
                    if self.path == "/api/wedge":
                        self._json(200, wedge(int(req.get("levels", 16)),
                                              req.get("dpi") or None, CKPT))
                        return
                    names, imgs = req["names"], req["images"]
                    if len(names) != len(imgs) or len(imgs) > 20:
                        raise ValueError("一次最多 20 张，且 names/images 对齐")
                    out = [process_one(n, u, CKPT, req.get("dpi") or None)
                           for n, u in zip(names, imgs)]
                    self._json(200, out)
                except Exception as e:  # noqa: BLE001
                    self._json(400, e, "text/plain")
            def log_message(self, *a):
                pass

        self.H = H


def main():
    global CKPT
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--ckpt", default="runs/onebit/best.pt")
    args = p.parse_args()
    CKPT = args.ckpt
    model, _printer, a, _device = load_model_and_printer(CKPT)
    kind = f"1-bit（printer={a.get('printer', 'paper')}）"
    H = Handler().H
    from http.server import HTTPServer
    httpd = HTTPServer(("127.0.0.1", args.port), H)
    print(f"网页已启动：http://localhost:{args.port}  (ckpt={args.ckpt} · {kind})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
