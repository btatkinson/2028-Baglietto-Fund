"""
webapp.py — password-protected web UI for the wc_props pipeline (Railway-ready).

The pipeline runs as a BACKGROUND JOB: POST /run saves the payloads and returns
immediately; /status polls /status.json until the job finishes. This keeps every
HTTP request fast so Railway's edge proxy never kills a long-running connection
(the cause of ERR_HTTP2_PROTOCOL_ERROR when the capture ran inside the request).

Routes (all behind HTTP Basic auth except /healthz):
  GET  /            latest report (out/report.html), or redirect to /control
  GET  /control     paste/upload underdog.json + prizepicks.json, start a run
  POST /run         save payloads, start background job, redirect to /status
  GET  /status      live job page (auto-polls, links to report when done)
  GET  /status.json job state for the poller
  GET  /healthz     unauthenticated liveness probe

Auth env vars: APP_PASSWORD (required — app refuses to serve without it),
APP_USER (default "blake"). Set BETSAPI_TOKEN for live captures.
"""
from __future__ import annotations

import contextlib
import html
import io
import json
import os
import secrets
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, Response, redirect, request

import config
from run import pipeline

app = Flask(__name__)

APP_USER = os.environ.get("APP_USER", "blake")
APP_PASSWORD = os.environ.get("APP_PASSWORD")


# ----------------------------- auth -----------------------------
def _authorized() -> bool:
    a = request.authorization
    return bool(
        a and a.username is not None and a.password is not None
        and secrets.compare_digest(a.username, APP_USER)
        and secrets.compare_digest(a.password, APP_PASSWORD or "")
    )


def requires_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not APP_PASSWORD:
            return Response("APP_PASSWORD is not set — refusing to serve. "
                            "Set it in the environment.", 503)
        if not _authorized():
            return Response("Authentication required.", 401,
                            {"WWW-Authenticate": 'Basic realm="wc_props"'})
        return f(*args, **kwargs)
    return wrapper


# ----------------------------- background job -----------------------------
# Minimum gap between one-click refreshes — the live pulls (bet365 + UD + PP) are
# rate-limited upstream and the lines barely move minute-to-minute, so throttle.
REFRESH_MIN_INTERVAL = int(os.environ.get("WCP_REFRESH_MIN_SECS", "300"))
_JOB_LOCK = threading.Lock()
_LAST_FINISH = 0.0              # epoch of the last completed run (for the throttle)
JOB = {"status": "idle",        # idle | running | done | error
       "log": "", "error": "",
       "started": None, "finished": None}


class _Tee(io.StringIO):
    """StringIO that mirrors writes into JOB['log'] so /status.json streams live."""
    def write(self, s):
        super().write(s)
        JOB["log"] += s
        return len(s)


def _run_job(scrape: bool, refresh: bool):
    global _LAST_FINISH
    buf = _Tee()
    try:
        with contextlib.redirect_stdout(buf):
            pipeline(scrape=scrape, refresh=refresh)
        JOB["status"] = "done"
    except Exception as e:
        JOB["status"] = "error"
        JOB["error"] = str(e)
        JOB["log"] += "\n" + traceback.format_exc(limit=6)
    finally:
        _LAST_FINISH = time.time()
        JOB["finished"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def _start_job(scrape: bool, refresh: bool) -> bool:
    with _JOB_LOCK:
        if JOB["status"] == "running":
            return False
        JOB.update(status="running", log="", error="",
                   started=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                   finished=None)
        threading.Thread(target=_run_job, args=(scrape, refresh), daemon=True).start()
        return True


def _refresh_wait() -> int:
    """Seconds until another one-click refresh is allowed (0 = ready now). A run in
    progress counts as blocked."""
    if JOB["status"] == "running":
        return REFRESH_MIN_INTERVAL
    if not _LAST_FINISH:
        return 0
    return max(0, int(REFRESH_MIN_INTERVAL - (time.time() - _LAST_FINISH)))


# ----------------------------- pages -----------------------------
_NAV = ("<div style=\"position:fixed;right:14px;bottom:14px;z-index:99\">"
        "<a href='/control' style='background:#1f6feb;color:#fff;text-decoration:none;"
        "padding:9px 14px;border-radius:999px;font:13px sans-serif'>⟳ control panel</a></div>")

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>wc_props — {title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body {{ margin:0; background:#0f1115; color:#e6e9ef; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
 .box {{ max-width:760px; margin:40px auto; padding:0 18px; }}
 h1 {{ font-size:20px; }} a {{ color:#7aa7ff; }}
 textarea {{ width:100%; min-height:110px; background:#171a21; color:#e6e9ef;
             border:1px solid #262b36; border-radius:8px; padding:10px; font:12px monospace; }}
 input[type=file] {{ color:#8b93a7; }}
 label.opt {{ display:flex; gap:8px; align-items:center; color:#8b93a7; margin:14px 0; }}
 button {{ background:#1f6feb; color:#fff; border:0; padding:10px 18px; border-radius:8px;
           font-size:15px; cursor:pointer; }}
 button.big {{ width:100%; padding:26px; font-size:23px; font-weight:700; border-radius:12px;
               letter-spacing:.3px; }}
 button:disabled {{ opacity:.45; cursor:not-allowed; }}
 .hint {{ color:#8b93a7; font-size:13px; }}
 pre {{ background:#171a21; border:1px solid #262b36; border-radius:8px; padding:14px;
        overflow:auto; font-size:12px; white-space:pre-wrap; }}
 .field {{ margin:18px 0 6px; font-weight:600; }}
 .ok {{ color:#5ad19a; }} .err {{ color:#f08a96; }}
 .spin {{ display:inline-block; animation:r 1s linear infinite; }}
 @keyframes r {{ to {{ transform:rotate(360deg); }} }}
</style></head><body><div class="box">{body}</div></body></html>"""


def _file_state(path):
    if not path.exists():
        return "<span class='err'>missing</span>"
    try:
        kb = path.stat().st_size / 1024
        return f"<span class='ok'>present</span> ({kb:,.0f} KB)"
    except OSError:
        return "?"


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/")
@requires_auth
def index():
    if config.REPORT_HTML.exists():
        body = config.REPORT_HTML.read_text(encoding="utf-8")
        return body.replace("</body>", _NAV + "</body>")
    return redirect("/control")


@app.get("/control")
@requires_auth
def control():
    running = JOB["status"] == "running"
    wait = _refresh_wait()
    if running:
        hero = ("<button class='big' disabled><span class='spin'>⟳</span> Running…</button>"
                "<p class='hint'>A run is in progress — <a href='/status'>watch it</a>.</p>")
    elif wait > 0:
        hero = (f"<button class='big' disabled>⟳ Refresh (wait {wait}s)</button>"
                f"<p class='hint'>Just refreshed — throttled for {REFRESH_MIN_INTERVAL // 60} min "
                f"to spare the upstream feeds. Available again in {wait}s.</p>"
                f"<script>setTimeout(()=>location.reload(), {wait * 1000 + 500})</script>")
    else:
        hero = ("<form method='post' action='/refresh'>"
                "<button class='big' type='submit'>⟳ Refresh &amp; run</button></form>"
                "<p class='hint'>One click: pull live bet365 + Underdog + PrizePicks, "
                "then run the pipeline.</p>")
    body = f"""
<h1>wc_props — control panel</h1>
{hero}
<details style="margin-top:26px">
  <summary class="hint" style="cursor:pointer">Manual override — paste/upload payloads (no auto-refresh)</summary>
  <p class="hint">Use when a feed is down or empty. Leave a box empty to keep the file on the server.
  This path does NOT auto-pull (your paste wins).</p>
  <form method="post" action="/run" enctype="multipart/form-data">
    <div class="field">underdog.json — {_file_state(config.UNDERDOG_JSON)}</div>
    <textarea name="underdog_text" placeholder="paste Underdog over_under_lines payload…"></textarea>
    <input type="file" name="underdog_file">
    <div class="field">prizepicks.json — {_file_state(config.PRIZEPICKS_JSON)}</div>
    <textarea name="prizepicks_text" placeholder="paste PrizePicks projections payload…"></textarea>
    <input type="file" name="prizepicks_file">
    <label class="opt"><input type="checkbox" name="scrape" checked>
      capture fresh bet365 lines (uncheck to reuse the last capture)</label>
    <button type="submit">Run with pasted lines</button>
  </form>
</details>
<p class="hint" style="margin-top:18px"><a href="/">← back to report</a></p>"""
    return _PAGE.format(title="control", body=body)


def _save_payload(path, text, file_storage):
    raw = None
    if file_storage and file_storage.filename:
        raw = file_storage.read().decode("utf-8-sig", errors="replace")
    elif text and text.strip():
        raw = text
    if raw is None:
        return None
    try:
        json.loads(raw.lstrip("\ufeff"))
    except ValueError as e:
        raise ValueError(f"{path.name}: not valid JSON ({e})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return f"saved {path.name} ({len(raw)/1024:,.0f} KB)"


@app.post("/run")
@requires_auth
def run_route():
    try:
        notes = []
        for path, tkey, fkey in ((config.UNDERDOG_JSON, "underdog_text", "underdog_file"),
                                 (config.PRIZEPICKS_JSON, "prizepicks_text", "prizepicks_file")):
            msg = _save_payload(path, request.form.get(tkey), request.files.get(fkey))
            if msg:
                notes.append(msg)
    except ValueError as e:
        body = (f"<h1 class='err'>Bad payload</h1><p>{html.escape(str(e))}</p>"
                f"<p><a href='/control'>← back</a></p>")
        return _PAGE.format(title="error", body=body), 400

    scrape = bool(request.form.get("scrape"))
    if not _start_job(scrape, refresh=False):       # manual paste: don't clobber it
        body = ("<h1>Already running</h1><p>A run is in progress — "
                "<a href='/status'>watch it</a>.</p>")
        return _PAGE.format(title="busy", body=body), 409
    if notes:
        JOB["log"] = "\n".join(notes) + "\n\n" + JOB["log"]
    return redirect("/status", code=303)


@app.post("/refresh")
@requires_auth
def refresh_route():
    """One-click: pull live bet365 + UD + PP and run. Throttled to once per
    REFRESH_MIN_INTERVAL so we don't hammer the upstream feeds."""
    if JOB["status"] == "running":
        body = ("<h1>Already running</h1><p>A run is in progress — "
                "<a href='/status'>watch it</a>.</p>")
        return _PAGE.format(title="busy", body=body), 409
    wait = _refresh_wait()
    if wait > 0:
        body = (f"<h1>Too soon</h1><p class='hint'>Lines were refreshed under "
                f"{REFRESH_MIN_INTERVAL // 60} min ago — try again in {wait}s.</p>"
                f"<p><a href='/control'>← back to control</a></p>")
        return _PAGE.format(title="throttled", body=body), 429
    _start_job(scrape=True, refresh=True)
    return redirect("/status", code=303)


@app.get("/status")
@requires_auth
def status_page():
    body = """
<h1 id="h">Run status</h1>
<pre id="log">…</pre>
<p id="links" class="hint"></p>
<script>
async function poll(){
  try{
    const r = await fetch('/status.json');
    const j = await r.json();
    document.getElementById('log').textContent = j.log || '(no output yet)';
    const h = document.getElementById('h'), links = document.getElementById('links');
    if(j.status === 'running'){
      h.innerHTML = '<span class="spin">⟳</span> Running… (started ' + (j.started||'') + ')';
      setTimeout(poll, 2000);
    } else if(j.status === 'done'){
      h.innerHTML = '<span class="ok">✓</span> Done (' + (j.finished||'') + ')';
      links.innerHTML = '<a href="/">→ open the report</a> · <a href="/control">run again</a>';
    } else if(j.status === 'error'){
      h.innerHTML = '<span class="err">✗</span> Failed: ' + (j.error||'');
      links.innerHTML = '<a href="/control">← back to control</a>';
    } else {
      h.textContent = 'No run yet';
      links.innerHTML = '<a href="/control">← control panel</a>';
    }
  }catch(e){ setTimeout(poll, 3000); }
}
poll();
</script>"""
    return _PAGE.format(title="status", body=body)


@app.get("/status.json")
@requires_auth
def status_json():
    return {k: JOB[k] for k in ("status", "log", "error", "started", "finished")}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
