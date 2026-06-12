"""
webapp.py — password-protected web UI for the wc_props pipeline (Railway-ready).

Routes (all behind HTTP Basic auth, credentials from env):
  GET  /         the latest report (out/report.html), or redirect to /control
  GET  /control  paste/upload underdog.json + prizepicks.json, run the pipeline
  POST /run      save payloads if provided, run pipeline, show the log
  GET  /healthz  unauthenticated liveness probe

Auth env vars: APP_PASSWORD (required — app refuses to serve without it),
APP_USER (default "blake"). Set BETSAPI_TOKEN for live captures.

Local:    flask --app webapp run        (or: python webapp.py)
Railway:  Dockerfile runs gunicorn; set env vars in the service settings.
"""
from __future__ import annotations

import contextlib
import html
import io
import json
import os
import secrets
import traceback
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
 .hint {{ color:#8b93a7; font-size:13px; }}
 pre {{ background:#171a21; border:1px solid #262b36; border-radius:8px; padding:14px;
        overflow:auto; font-size:12px; white-space:pre-wrap; }}
 .field {{ margin:18px 0 6px; font-weight:600; }}
 .ok {{ color:#5ad19a; }} .err {{ color:#f08a96; }}
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
    body = f"""
<h1>wc_props — control panel</h1>
<p class="hint">Paste the raw API JSON (or upload the files), then run.
Leave a box empty to keep the file already on the server.</p>
<form method="post" action="/run" enctype="multipart/form-data">
  <div class="field">underdog.json — {_file_state(config.UNDERDOG_JSON)}</div>
  <textarea name="underdog_text" placeholder="paste Underdog over_under_lines payload…"></textarea>
  <input type="file" name="underdog_file">
  <div class="field">prizepicks.json — {_file_state(config.PRIZEPICKS_JSON)}</div>
  <textarea name="prizepicks_text" placeholder="paste PrizePicks projections payload…"></textarea>
  <input type="file" name="prizepicks_file">
  <label class="opt"><input type="checkbox" name="scrape" checked>
    capture fresh bet365 lines (uncheck to reuse the last capture)</label>
  <button type="submit">Run pipeline</button>
</form>
<p class="hint"><a href="/">← back to report</a></p>"""
    return _PAGE.format(title="control", body=body)


def _save_payload(path, text, file_storage):
    """Save pasted text or an uploaded file to `path` (validated as JSON).
    Returns a status string or None if nothing was provided."""
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
    notes = []
    try:
        for path, tkey, fkey in ((config.UNDERDOG_JSON, "underdog_text", "underdog_file"),
                                 (config.PRIZEPICKS_JSON, "prizepicks_text", "prizepicks_file")):
            msg = _save_payload(path, request.form.get(tkey),
                                request.files.get(fkey))
            if msg:
                notes.append(msg)
        scrape = bool(request.form.get("scrape"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline(scrape=scrape)
        log = "\n".join(notes + ["", buf.getvalue()])
        body = (f"<h1>Run complete</h1><pre>{html.escape(log)}</pre>"
                f"<p><a href='/'>→ open the report</a> · <a href='/control'>run again</a></p>")
        return _PAGE.format(title="run", body=body)
    except Exception as e:
        log = "\n".join(notes + ["", traceback.format_exc(limit=4)])
        body = (f"<h1 class='err'>Run failed</h1><p>{html.escape(str(e))}</p>"
                f"<pre>{html.escape(log)}</pre>"
                f"<p><a href='/control'>← back</a></p>")
        return _PAGE.format(title="error", body=body), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
