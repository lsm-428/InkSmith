"""网页版神经半色调：上传单图/多图 -> 模型 vs FS vs GS 筛网 对照。

用法：uv run python webapp.py [--port 8123] [--ckpt runs/halftone/best.pt]
浏览器打开 http://localhost:8123
标准库实现，无需新依赖。

接口：
  GET  /            页面
  POST /api/convert {names[], images[dataURL], dpi?} -> 每图对照结果
  POST /api/wedge   {dpi?, levels?}                -> 灰度尺分段覆盖偏差诊断
"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval import (film_target_and_base, load_model_and_printer, psnr,
                  triplet)  # 复用评测核心：webapp 与 CLI 共用同一套三条路线逻辑

HTML = """
<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>神经半色调对照</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 12px;color:#222}
 .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 button{padding:8px 14px;font-size:15px}
 input[type=number]{width:90px;padding:6px}
 .card{border:1px solid #ddd;border-radius:8px;padding:10px;margin:14px 0}
 img{display:block;margin:4px auto;padding:2px;background:#eee;image-rendering:pixelated;max-width:440px;max-height:440px}
 .stack .blk{margin:10px 0;text-align:center}
 .stack .blk b{display:block;margin-bottom:4px}
 .stack.cols{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px}
 .stack.cols .blk{margin:2px}
 .muted{color:#777;font-size:13px}
 .good{color:#0a7f3f;font-weight:600} .bad{color:#c02020;font-weight:600}
 #summary{font-size:15px;margin:10px 0;padding:8px;background:#f4f4f4;border-radius:6px}
 .meta{color:#888;font-size:12px}
 table.wt{border-collapse:collapse;margin:10px 0;font-size:13px;width:100%}
 table.wt th,table.wt td{border:1px solid #e4e4e4;padding:4px 8px;text-align:center}
</style></head><body>
<h2>神经半色调对照：模型 vs Floyd-Steinberg vs Ghostscript 筛网</h2>
<p class="muted">上传单图或多图，后端用当前 ckpt 推理，给出三条路线的位图、模拟打印与 PSNR。
film 模型下灰度=相对膜厚，FS/GS 基线自动反相对齐。</p>
<div class="row">
  <input id="f" type="file" accept="image/*" multiple>
  <label class="muted">评估 dpi <input id="dpi" type="number" placeholder="ckpt值"></label>
  <label class="muted">布局 <select id="layout"><option value="v">纵向</option><option value="cols">两列并排</option></select></label>
  <button onclick="go()">开始转换</button>
  <button onclick="dlAll()">下载全部</button>
  <button onclick="toggleWedge()">灰度尺诊断</button>
  <span id="st" class="muted"></span>
</div>
<div id="summary"></div>
<div id="wedge"></div>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
let LAST=[];
const dpiVal=()=>{const v=$('dpi').value.trim();return v?parseInt(v):null;};
const cls=()=> 'stack'+($('layout').value==='cols'?' cols':'');

async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function dlOne(url,name){const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();}
function dlAll(){
  const d=new Date().toISOString().slice(0,10);
  LAST.forEach(x=>{
    dlOne(x.img.nn_bitmap, x.name+'_'+d+'_nn_bitmap.png');
    dlOne(x.img.nn_preview,x.name+'_'+d+'_nn_preview.png');
    dlOne(x.img.fs_bitmap, x.name+'_'+d+'_fs_bitmap.png');
    dlOne(x.img.fs_preview,x.name+'_'+d+'_fs_preview.png');
    dlOne(x.img.gs_bitmap, x.name+'_'+d+'_gs_bitmap.png');
    dlOne(x.img.gs_preview,x.name+'_'+d+'_gs_preview.png');
  });
}
async function toggleWedge(){
  const box=$('wedge');
  if(box.innerHTML){box.innerHTML='';return;}
  $('st').innerText='灰度尺诊断中...';
  try{
    const w=await post('/api/wedge',{dpi:dpiVal()});
    const c=v=>Math.abs(v)<=0.02?'':(v>0?'good':'bad');
    const rows=w.rows.map(r=>`<tr><td>${r.level.toFixed(3)}</td>
      <td>${r.nn.toFixed(3)}</td><td class="${c(r.d_nn)}">${r.d_nn>=0?'+':''}${r.d_nn.toFixed(3)}</td>
      <td>${r.fs.toFixed(3)}</td><td class="${c(r.d_fs)}">${r.d_fs>=0?'+':''}${r.d_fs.toFixed(3)}</td>
      <td>${r.gs.toFixed(3)}</td><td class="${c(r.d_gs)}">${r.d_gs>=0?'+':''}${r.d_gs.toFixed(3)}</td></tr>`).join('');
    const worst=w.rows.reduce((a,b)=>Math.abs(b.d_nn)>Math.abs(a.d_nn)?b:a);
    box.innerHTML=`<div class="card"><h3>灰度尺诊断（${w.rows.length} 阶${w.dpi?' @'+w.dpi+'dpi':''}，${w.ms} ms）</h3>
      <p class="muted">全图 PSNR：模型 <b>${w.psnr.nn}</b> | FS ${w.psnr.fs} | GS ${w.psnr.gs} dB。
      偏差 = 实际覆盖率 − 目标覆盖率，|偏差|&gt;0.02 标色。模型最大偏差出现在 ${worst.level.toFixed(3)} 灰阶（${worst.d_nn>=0?'+':''}${worst.d_nn.toFixed(3)}）——
      这是判断灰阶映射是否畸变（S 形）的最直接指标。</p>
      <table class="wt"><tr><th>灰阶</th><th colspan=2>模型</th><th colspan=2>FS</th><th colspan=2>GS</th></tr>
      <tr><th></th><th>覆盖</th><th>偏差</th><th>覆盖</th><th>偏差</th><th>覆盖</th><th>偏差</th></tr>${rows}</table>
      <div class="stack cols">${[['原图',w.img.orig],['模型',w.img.nn],['FS',w.img.fs],['GS',w.img.gs]]
        .map(([a,b])=>`<div class="blk"><b>${a}</b><img src="${b}"></div>`).join('')}</div></div>`;
    $('st').innerText='';
  }catch(e){$('st').innerText='失败: '+e.message;}
}
async function go(){
  const fs=$('f').files;
  if(!fs.length){$('st').innerText='请选择图片'; return;}
  $('st').innerText='读取中...'; $('out').innerHTML=''; $('summary').innerHTML='';
  const imgs=[],names=[];
  for(const file of fs){
    names.push(file.name);
    imgs.push(await new Promise(r=>{const rd=new FileReader();rd.onload=()=>r(rd.result);rd.readAsDataURL(file);}));
  }
  $('st').innerText='转换中...';
  let data;
  try{ data=await post('/api/convert',{names,images:imgs,dpi:dpiVal()}); }
  catch(e){ $('st').innerText='失败: '+e.message; return; }
  LAST=data;
  const t=data.reduce((a,d)=>a+d.ms,0);
  $('st').innerText=`完成 ${data.length} 张，用时 ${(t/1000).toFixed(1)} s`;
  $('out').innerHTML=data.map(d=>card(d,cls())).join('');
  if(data.length>1){
    const m=k=>data.reduce((a,d)=>a+d.psnr[k],0)/data.length;
    const c=v=>v>=0?'good':'bad';
    const s=v=>(v>=0?'+':'')+v.toFixed(2);
    $('summary').innerHTML=`<b>均值</b>（${data.length} 张） 模型 <b>${m('nn').toFixed(2)}</b> |
     FS ${m('fs').toFixed(2)} | GS ${m('gs').toFixed(2)} dB ｜
     <span class="${c(m('vs_fs'))}">vs FS ${s(m('vs_fs'))}</span> |
     <span class="${c(m('vs_gs'))}">vs GS ${s(m('vs_gs'))}</span>`;
  }
}
function card(d,cl){
  const c=v=>v>=0?'good':'bad';
  const s=v=>(v>=0?'+':'')+v;
  const dl=(a,b,n)=>`<div class="blk"><b>${a}</b><img src="${b}"><div><a href="${b}" download="${n}">下载</a></div></div>`;
  const inkTip=`墨覆盖率：模型 ${d.ink.nn} · FS ${d.ink.fs} · GS ${d.ink.gs} · 目标 ${d.ink.target}`;
  return `<div class="card"><h3>${d.name} <span class="meta">${d.size[0]}×${d.size[1]}${d.dpi?' @'+d.dpi+'dpi':''}
   ${d.printer?'['+d.printer+']':''}${d.scaled?'（已缩放）':''} · ${d.ms} ms</span></h3>
   <p>PSNR：模型 <b>${d.psnr.nn}</b> | FS ${d.psnr.fs} | GS ${d.psnr.gs} dB
   ｜ <span class="${c(d.psnr.vs_fs)}">vs FS ${s(d.psnr.vs_fs)}</span>
   ｜ <span class="${c(d.psnr.vs_gs)}">vs GS ${s(d.psnr.vs_gs)}</span>
   <span class="meta">｜${inkTip}</span></p>
   <div class="${cl}">
    <div class="blk"><b>原图</b><img src="${d.img.orig}"></div>
    ${dl('模型位图',d.img.nn_bitmap,d.name+'_nn_bitmap.png')}
    ${dl('模型模拟打印',d.img.nn_preview,d.name+'_nn_preview.png')}
    ${dl('FS位图',d.img.fs_bitmap,d.name+'_fs_bitmap.png')}
    ${dl('FS模拟打印',d.img.fs_preview,d.name+'_fs_preview.png')}
    ${dl('GS位图',d.img.gs_bitmap,d.name+'_gs_bitmap.png')}
    ${dl('GS模拟打印',d.img.gs_preview,d.name+'_gs_preview.png')}
   </div></div>`;
}
</script></body></html>
"""

_CACHE = {}


def get_model(ckpt, dpi=None):
    """按 (ckpt, dpi) 缓存重建好的 UNet + printer。"""
    key = (ckpt, dpi)
    if key not in _CACHE:
        _CACHE[key] = load_model_and_printer(ckpt, dpi=dpi)
    return _CACHE[key]


MAX_SIDE = 1024


def png_dataurl(arr):
    """PIL/np -> dataURL PNG。墨位图用 墨=黑、白=纸 显示。"""
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _triplet(img, dpi):
    """跑三条路线（复用 eval.triplet），返回 (inks, prints, target, ckpt_args)。"""
    return triplet(img, CKPT, dpi)


def process_one(name, dataurl, dpi=None):
    t0 = time.time()
    b = dataurl.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b))).convert("L")
    scaled = False
    if max(img.size) > MAX_SIDE:  # 控制单次推理显存/时间
        r = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        scaled = True
    inks, prints, target, a = _triplet(img, dpi)
    ink2img = lambda ink: np.uint8((1 - ink[0, 0].cpu().numpy()) * 255)  # 墨=黑
    clip = lambda x: x[0, 0].cpu().numpy().clip(0, 1)
    p_nn, p_fs, p_gs = (psnr(prints["nn"], target), psnr(prints["fs"], target),
                        psnr(prints["gs"], target))
    return {
        "name": name, "size": [img.width, img.height], "scaled": scaled,
        "dpi": dpi or a.get("dpi"), "ms": round((time.time() - t0) * 1000),
        "printer": a.get("printer", "paper"),
        "ink": {"nn": round(inks["nn"].mean().item(), 4),
                "fs": round(inks["fs"].mean().item(), 4),
                "gs": round(inks["gs"].mean().item(), 4),
                "target": round(target.mean().item(), 4)},
        "psnr": {"nn": round(p_nn, 2), "fs": round(p_fs, 2), "gs": round(p_gs, 2),
                 "vs_fs": round(p_nn - p_fs, 2), "vs_gs": round(p_nn - p_gs, 2)},
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


def wedge(n_levels=16, dpi=None):
    """灰度尺诊断：分段覆盖率偏差。这是模型最敏感的体检项（PITFALLS #11 系列问题先看这里）。"""
    t0 = time.time()
    n_levels = max(2, min(32, n_levels))
    seg = 96
    levels = np.arange(n_levels) / (n_levels - 1)
    arr = np.concatenate([np.full((seg, seg), l, np.float32) for l in levels], axis=1)
    img = Image.fromarray((arr * 255).astype("uint8"), "L")
    inks, prints, target, a = _triplet(img, dpi)
    dev = lambda t, k: t[0, 0, :, k * seg:(k + 1) * seg].mean().item()
    rows = [{"level": round(float(l), 3),
             "nn": round(dev(prints["nn"], k), 3),
             "fs": round(dev(prints["fs"], k), 3),
             "gs": round(dev(prints["gs"], k), 3)}
            for k, l in enumerate(levels)]
    for r in rows:
        for m in ("nn", "fs", "gs"):
            r[f"d_{m}"] = round(r[m] - r["level"], 3)
    imgs = {m: png_dataurl(t[0, 0].cpu().numpy().clip(0, 1))
            for m, t in prints.items()}
    return {
        "rows": rows, "dpi": dpi or a.get("dpi"),
        "ms": round((time.time() - t0) * 1000),
        "psnr": {"nn": round(psnr(prints["nn"], target), 2),
                 "fs": round(psnr(prints["fs"], target), 2),
                 "gs": round(psnr(prints["gs"], target), 2)},
        "img": {"orig": png_dataurl(np.asarray(img, np.uint8)), **imgs},
    }


class Handler:
    def __init__(self):
        from http.server import BaseHTTPRequestHandler

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()

            def _json(self, code, obj, ctype="application/json"):
                body = (json.dumps(obj, ensure_ascii=False) if ctype.startswith("application/json")
                        else str(obj)).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8"
                                 if ctype.startswith("text/") else ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path not in ("/api/convert", "/api/wedge"):
                    self.send_response(404); self.end_headers(); return
                try:
                    ln = int(self.headers["Content-Length"])
                    req = json.loads(self.rfile.read(ln))
                    dpi = req.get("dpi") or None
                    Path("out/gs").mkdir(parents=True, exist_ok=True)
                    if self.path == "/api/wedge":
                        self._json(200, wedge(int(req.get("levels", 16)), dpi))
                        return
                    names, imgs = req["names"], req["images"]
                    if len(names) != len(imgs) or len(imgs) > 20:
                        raise ValueError("一次最多 20 张，且 names/images 对齐")
                    out = [process_one(n, u, dpi) for n, u in zip(names, imgs)]
                    self._json(200, out)
                except Exception as e:  # noqa: BLE001
                    self._json(400, e, "text/plain")

            def log_message(self, *a):
                pass

        self.H = H


CKPT = None


def main():
    global CKPT
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--ckpt", default="runs/halftone/best.pt")
    args = p.parse_args()
    CKPT = args.ckpt
    Path("out/gs").mkdir(parents=True, exist_ok=True)
    H = Handler().H
    from http.server import HTTPServer
    httpd = HTTPServer(("127.0.0.1", args.port), H)
    print(f"网页已启动：http://localhost:{args.port}  (ckpt={args.ckpt})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
