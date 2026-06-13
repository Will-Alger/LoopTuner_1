"""Tiny local web UI for LoopTuner.

Run ``python -m looptuner.cli ui`` (or ``python -m looptuner.webui``) and open
the printed http://127.0.0.1:8765 in a browser. Enter your Nightscout URL and
secret/token in the form instead of editing a config file, click Run, and the
recommendations render as tables.

Notes
-----
* Binds to localhost only; your secret is sent to this local process and used
  in-memory to call Nightscout. It is never written to disk or logged.
* Sampling takes ~30-120s; the page polls a status endpoint and shows the
  result when the background job finishes.
"""

from __future__ import annotations

import threading
import uuid

from .config import LoopTunerConfig, NightscoutConfig
from .pipeline import run as run_pipeline
from .report import DISCLAIMER, condense_blocks

try:
    from flask import Flask, Response, jsonify, request
except ImportError:  # pragma: no cover - helpful message if Flask is missing
    raise SystemExit(
        "The web UI needs Flask. Install it with:  pip install flask\n"
        "(or: pip install -e '.[ui]')"
    )

app = Flask(__name__)
_JOBS: dict[str, dict] = {}


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>LoopTuner</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:920px;margin:24px auto;padding:0 16px;color:#1c2230}
 h1{margin:0 0 4px} .sub{color:#667;margin:0 0 16px}
 .warn{background:#fff5f5;border:1px solid #f3c2c2;color:#7a1f1f;padding:12px 14px;border-radius:8px;font-size:14px;margin:14px 0}
 form{background:#f7f8fb;border:1px solid #e2e6ef;border-radius:10px;padding:16px}
 label{display:block;font-weight:600;margin:10px 0 4px;font-size:14px}
 input,select{width:100%;padding:9px 10px;border:1px solid #ccd2e0;border-radius:7px;font-size:14px;box-sizing:border-box}
 .row{display:flex;gap:12px}.row>div{flex:1}
 button{margin-top:16px;background:#2a5bd7;color:#fff;border:0;border-radius:8px;padding:11px 18px;font-size:15px;cursor:pointer}
 button:disabled{background:#9bb0e0;cursor:default}
 table{border-collapse:collapse;width:100%;margin:8px 0 22px;font-size:14px}
 th,td{border:1px solid #e2e6ef;padding:6px 9px;text-align:left}
 th{background:#eef1f8} .chg{background:#fff8e6} .up{color:#0a7a2f}.down{color:#b3261e}
 .summary{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 18px}
 .card{background:#f7f8fb;border:1px solid #e2e6ef;border-radius:8px;padding:10px 14px;min-width:150px}
 .card b{display:block;font-size:20px} .muted{color:#778;font-size:13px}
 #status{margin:16px 0;font-weight:600}
 .spin{display:inline-block;width:14px;height:14px;border:2px solid #ccd;border-top-color:#2a5bd7;border-radius:50%;animation:s 0.8s linear infinite;vertical-align:-2px}
 @keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<h1>LoopTuner</h1>
<p class="sub">Bayesian tuning suggestions from your Nightscout history.</p>
<div class="warn"><b>Not a medical device.</b> Every value is a suggestion from
historical data and a simplified model. Do not change any pump setting without
review by your licensed clinician.</div>

<form id="f">
 <label>Nightscout URL</label>
 <input name="url" placeholder="https://your-site.up.railway.app" required>
 <div class="row">
  <div>
   <label>Authentication</label>
   <select name="auth_type">
     <option value="api_secret">API secret (passphrase)</option>
     <option value="token">Access token</option>
   </select>
  </div>
  <div>
   <label>Days of history</label>
   <input name="days" type="number" value="30" min="3" max="180">
  </div>
 </div>
 <label>Secret / token</label>
 <input name="auth_value" type="password" placeholder="your passphrase or token" required>
 <label>Nightscout profile</label>
 <div class="row">
  <div style="flex:3">
   <select name="profile_name" id="profsel">
     <option value="">Site default</option>
   </select>
  </div>
  <div style="flex:1">
   <button type="button" id="loadprof" style="margin-top:0;width:100%;background:#5a6478">Load profiles</button>
  </div>
 </div>
 <span class="muted" id="profmsg"></span>
 <button type="submit" id="go">Run analysis</button>
</form>

<div id="status"></div>
<div id="out"></div>

<script>
const f=document.getElementById('f'), go=document.getElementById('go'),
      st=document.getElementById('status'), out=document.getElementById('out'),
      loadprof=document.getElementById('loadprof'), profsel=document.getElementById('profsel'),
      profmsg=document.getElementById('profmsg');
loadprof.onclick=async()=>{
 const body=Object.fromEntries(new FormData(f));
 if(!body.url||!body.auth_value){profmsg.textContent='Enter URL and secret/token first.';return;}
 profmsg.textContent='Loading profiles…'; loadprof.disabled=true;
 try{
  const r=await fetch('/profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  if(j.error){profmsg.textContent='Error: '+j.error;}
  else if(!j.names||!j.names.length){profmsg.textContent='No named profiles found.';}
  else{
   profsel.innerHTML='<option value="">Site default ('+(j.default||'?')+')</option>';
   for(const n of j.names){const o=document.createElement('option');o.value=n;o.textContent=(n===j.default?n+' (default)':n);profsel.appendChild(o);}
   profmsg.textContent='Loaded '+j.names.length+' profile(s).';
  }
 }catch(e){profmsg.textContent='Error: '+e;}
 loadprof.disabled=false;
};
f.onsubmit=async e=>{
 e.preventDefault(); go.disabled=true; out.innerHTML='';
 st.innerHTML='<span class="spin"></span> Submitting…';
 const body=Object.fromEntries(new FormData(f));
 const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const {job_id,error}=await r.json();
 if(error){st.textContent='Error: '+error;go.disabled=false;return;}
 poll(job_id);
};
async function poll(id){
 st.innerHTML='<span class="spin"></span> Running — fetching data and sampling the model (this can take 30–120s)…';
 const r=await fetch('/status/'+id); const j=await r.json();
 if(j.state==='done'){st.textContent='Done.';go.disabled=false;render(j.result,id);}
 else if(j.state==='error'){st.textContent='Error: '+j.error;go.disabled=false;}
 else setTimeout(()=>poll(id),1500);
}
function tbl(title,unit,blocks){
 let h=`<h3>${title}</h3><table><tr><th>Time</th><th>Current</th><th>Suggested (${unit})</th><th></th></tr>`;
 for(const b of blocks){
  const cur=b.current==null?'—':b.current;
  let dir='';
  if(b.current!=null){ if(b.recommended>b.current)dir='<span class="up">▲</span>'; else if(b.recommended<b.current)dir='<span class="down">▼</span>'; }
  h+=`<tr class="${b.confident?'chg':''}"><td>${b.label}</td><td>${cur}</td><td>${b.recommended} ${dir}</td><td>${b.confident?'data-supported':''}</td></tr>`;
 }
 return h+'</table>';
}
function render(d,id){
 const s=d.summary;
 let h='<div class="summary">'+
  card('Typical ISF',s.population_isf+' mg/dL/U')+
  card('Typical carb ratio',s.population_carb_ratio+' g/U')+
  card('Total daily basal',s.total_daily_basal+' U')+
  card('Model residual',s.obs_sd_mgdl+' mg/dL')+'</div>';
 h+='<h3>How it adjusted each setting</h3>';
 h+='<img src="/plot/'+id+'.png" alt="current vs recommended chart" style="width:100%;border:1px solid #e2e6ef;border-radius:8px">';
 h+='<p class="muted">Grey dashed = your current schedule, blue = model estimate with its 94% credible band, green = recommended. Red dots mark hours where the data supports a change. Wide blue bands (often overnight) are hours the data can\\'t pin down — left unchanged on purpose.</p>';
 h+='<p class="muted">Rows highlighted in yellow are changes the data actually supports (your current value falls outside the 94% credible interval). Others are left at your current setting.</p>';
 h+=tbl('Insulin sensitivity (ISF)','mg/dL/U',d.isf);
 h+=tbl('Carb ratio','g/U',d.carb_ratio);
 h+=tbl('Basal rate','U/hr',d.basal);
 h+='<h3>Safety delivery limits (suggested)</h3><table>'+
  `<tr><th>Maximum basal</th><td>${d.max_basal} U/hr</td></tr>`+
  `<tr><th>Maximum bolus</th><td>${d.max_bolus} U</td></tr>`+
  (d.suspend_threshold!=null?`<tr><th>Suspend threshold</th><td>${d.suspend_threshold} mg/dL</td></tr>`:'')+
  '<tr><th>Minimum delivery</th><td>0 U/hr (suspend on low)</td></tr></table>';
 h+='<div class="warn">'+d.disclaimer+'</div>';
 out.innerHTML=h;
}
function card(t,v){return `<div class="card"><span class="muted">${t}</span><b>${v}</b></div>`;}
</script>
</body></html>"""


@app.route("/")
def index():
    return PAGE


def _nightscout_from_request(data: dict) -> NightscoutConfig:
    url = (data.get("url") or "").strip()
    auth_type = data.get("auth_type", "api_secret")
    auth_value = (data.get("auth_value") or "").strip()
    if not url or not auth_value:
        raise ValueError("URL and secret/token are required")
    try:
        days = max(3, min(int(data.get("days", 30)), 180))
    except (TypeError, ValueError):
        days = 30
    ns = NightscoutConfig(url=url, days=days)
    if auth_type == "token":
        ns.token = auth_value
    else:
        ns.api_secret = auth_value
    name = (data.get("profile_name") or "").strip()
    if name:
        ns.profile_name = name
    return ns


@app.route("/profiles", methods=["POST"])
def profiles():
    from .nightscout import NightscoutClient
    from .profile import list_profiles

    data = request.get_json(force=True)
    try:
        ns = _nightscout_from_request(data)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    try:
        docs = NightscoutClient(ns).profile(use_cache=False)
        names, default = list_profiles(docs)
    except Exception as err:
        return jsonify({"error": f"{type(err).__name__}: {err}"}), 502
    return jsonify({"names": names, "default": default})


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True)
    try:
        ns = _nightscout_from_request(data)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    cfg = LoopTunerConfig(nightscout=ns)

    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"state": "running"}
    threading.Thread(target=_worker, args=(job_id, cfg), daemon=True).start()
    return jsonify({"job_id": job_id})


def _worker(job_id: str, cfg: LoopTunerConfig):
    try:
        result = run_pipeline(cfg, use_cache=False, progressbar=False)
        from .plots import render_png
        png = render_png(result.fit, result.profile, result.recommendations)
        _JOBS[job_id] = {
            "state": "done",
            "result": _serialize(result.recommendations),
            "png": png,
        }
    except Exception as err:  # surface a readable message to the browser
        _JOBS[job_id] = {"state": "error", "error": f"{type(err).__name__}: {err}"}


def _serialize(recs) -> dict:
    return {
        "summary": recs.summary,
        "max_basal": recs.max_basal,
        "max_bolus": recs.max_bolus,
        "suspend_threshold": recs.suspend_threshold,
        "isf": condense_blocks(recs.isf),
        "carb_ratio": condense_blocks(recs.carb_ratio),
        "basal": condense_blocks(recs.basal),
        "disclaimer": DISCLAIMER,
    }


@app.route("/status/<job_id>")
def status(job_id):
    job = _JOBS.get(job_id)
    if job is None:
        return jsonify({"state": "error", "error": "unknown job"}), 404
    # Exclude the raw PNG bytes (served separately) so the payload is JSON.
    payload = {k: v for k, v in job.items() if k != "png"}
    payload["has_chart"] = "png" in job
    return jsonify(payload)


@app.route("/plot/<job_id>.png")
def plot(job_id):
    job = _JOBS.get(job_id)
    if not job or "png" not in job:
        return "not found", 404
    return Response(job["png"], mimetype="image/png")


def serve(host: str = "127.0.0.1", port: int = 8765):
    print(f"LoopTuner UI running at http://{host}:{port}  (Ctrl-C to stop)")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    serve()
