#!/usr/bin/env python3
"""
Photo Frame Identification System
Photos are NEVER stored locally — only face encodings (~5 MB) are cached.
The face index is persisted back to Dropbox so it survives Railway redeploys.
"""

import os
import json
import pickle
import tempfile
import hmac
import hashlib
import time
import threading
from functools import wraps
from pathlib import Path
from flask import (Flask, request, jsonify, send_file, Response,
                   render_template_string, session, redirect, url_for)

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
ENCODINGS_FILE    = BASE_DIR / 'face_index.pkl'
MANIFEST_FILE     = BASE_DIR / 'dropbox_manifest.json'
SYNC_STATUS_FILE  = BASE_DIR / 'sync_status.json'
INDEX_STATUS_FILE = BASE_DIR / 'index_status.json'
IMG_EXTS          = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}

DROPBOX_TOKEN  = os.environ.get('DROPBOX_TOKEN', '')
DROPBOX_FOLDER = os.environ.get('DROPBOX_FOLDER', '/Student Photos')
APP_PASSWORD   = os.environ.get('APP_PASSWORD', '')

_SECRET = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.secret_key = _SECRET

# The face index is stored back to Dropbox so it persists across redeploys
DROPBOX_INDEX_PATH = DROPBOX_FOLDER.rstrip('/') + '/.face_index.pkl'

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if APP_PASSWORD and not session.get('authenticated'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def sign_photo_url(dropbox_path: str) -> str:
    expiry  = int(time.time()) + 3600
    message = f"{dropbox_path}:{expiry}".encode()
    sig     = hmac.new(_SECRET.encode(), message, hashlib.sha256).hexdigest()
    return f"{expiry}:{sig}"


def verify_photo_token(dropbox_path: str, token: str) -> bool:
    try:
        expiry_str, sig = token.split(':', 1)
        expiry = int(expiry_str)
        if time.time() > expiry:
            return False
        message  = f"{dropbox_path}:{expiry}".encode()
        expected = hmac.new(_SECRET.encode(), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

# ── Lazy imports ──────────────────────────────────────────────────────────────

_fr_error = None

def _patch_pkg_resources():
    try:
        import pkg_resources
        return
    except ImportError:
        pass
    import sys, os, types, importlib.util
    pkg = types.ModuleType('pkg_resources')
    def resource_filename(package_or_req, resource_name):
        name = package_or_req if isinstance(package_or_req, str) else package_or_req.__name__
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            return os.path.join(os.path.dirname(spec.origin), resource_name)
        return resource_name
    pkg.resource_filename = resource_filename
    sys.modules['pkg_resources'] = pkg
    print("[startup] pkg_resources shim installed", flush=True)


_patch_pkg_resources()


def require_fr():
    global _fr_error
    try:
        import face_recognition, numpy as np
        _fr_error = None
        return face_recognition, np
    except BaseException as e:
        _fr_error = f"{type(e).__name__}: {e}"
        print(f"[face_recognition import error] {_fr_error}", flush=True)
        return None, None


def require_dbx():
    if not DROPBOX_TOKEN:
        return None, 'DROPBOX_TOKEN environment variable not set.'
    try:
        import dropbox
        return dropbox.Dropbox(DROPBOX_TOKEN), None
    except ImportError:
        return None, 'dropbox package not installed.'

# ── Index persistence via Dropbox ─────────────────────────────────────────────

def _load_index_from_dropbox():
    """On startup, download the face index from Dropbox if it exists."""
    if ENCODINGS_FILE.exists():
        return
    dbx, err = require_dbx()
    if err:
        return
    try:
        _, response = dbx.files_download(DROPBOX_INDEX_PATH)
        ENCODINGS_FILE.write_bytes(response.content)
        count = len(pickle.loads(response.content))
        print(f"[startup] Loaded face index from Dropbox ({count} entries)", flush=True)
    except Exception as e:
        print(f"[startup] No cached index in Dropbox (will need to build): {e}", flush=True)


def _save_index_to_dropbox(data: list):
    """Save the face index to Dropbox so it survives redeploys."""
    dbx, err = require_dbx()
    if err:
        return
    try:
        import dropbox as dbx_module
        raw = pickle.dumps(data)
        dbx.files_upload(raw, DROPBOX_INDEX_PATH,
                         mode=dbx_module.files.WriteMode.overwrite)
        print(f"[index] Saved {len(data)} encodings to Dropbox", flush=True)
    except Exception as e:
        print(f"[index] Failed to save index to Dropbox: {e}", flush=True)


# Run on startup — restores index from Dropbox after a redeploy
_load_index_from_dropbox()

# ── Background task helpers ───────────────────────────────────────────────────

_sync_lock  = threading.Lock()
_index_lock = threading.Lock()
_sync_running  = False
_index_running = False


def _write_status(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data))
    except Exception as e:
        print(f"[status write error] {e}", flush=True)


def _read_status(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {'state': 'idle'}

# ── Dropbox sync (list only — no downloads) ───────────────────────────────────

def sync_dropbox(progress_cb=None):
    """List all image files in Dropbox and save a manifest. No photos downloaded."""
    dbx, err = require_dbx()
    if err:
        return {'success': False, 'error': err}

    import dropbox as dbx_module

    try:
        if progress_cb: progress_cb('Connecting to Dropbox…')
        result  = dbx.files_list_folder(DROPBOX_FOLDER, recursive=True)
        entries = list(result.entries)
        while result.has_more:
            if progress_cb: progress_cb(f'Listing files… ({len(entries)} found so far)')
            result   = dbx.files_list_folder_continue(result.cursor)
            entries += result.entries
    except Exception as e:
        return {'success': False, 'error': f'Cannot access Dropbox folder "{DROPBOX_FOLDER}": {e}'}

    manifest = []
    for entry in entries:
        if not isinstance(entry, dbx_module.files.FileMetadata):
            continue
        if Path(entry.name).suffix.lower() not in IMG_EXTS:
            continue
        # Strip base folder to get relative path; top-level subfolder = program name
        rel   = entry.path_display[len(DROPBOX_FOLDER):].lstrip('/')
        parts = Path(rel).parts
        folder = parts[0] if len(parts) > 1 else 'Unknown'
        manifest.append({
            'dropbox_path': entry.path_display,
            'folder':       folder,
            'name':         Path(entry.name).stem,
            'size':         entry.size,
        })

    MANIFEST_FILE.write_text(json.dumps(manifest))
    return {'success': True, 'found': len(manifest)}


def _start_sync_background():
    global _sync_running
    with _sync_lock:
        if _sync_running:
            return False
        _sync_running = True

    def run():
        global _sync_running
        try:
            def cb(msg):
                _write_status(SYNC_STATUS_FILE, {'state': 'running', 'message': msg})
            _write_status(SYNC_STATUS_FILE, {'state': 'running', 'message': 'Starting…'})
            result = sync_dropbox(progress_cb=cb)
            if result.get('success'):
                _write_status(SYNC_STATUS_FILE, {
                    'state':   'done',
                    'success': True,
                    'found':   result['found'],
                })
            else:
                _write_status(SYNC_STATUS_FILE, {
                    'state': 'done', 'success': False, 'error': result['error'],
                })
        except Exception as e:
            _write_status(SYNC_STATUS_FILE, {'state': 'done', 'success': False, 'error': str(e)})
        finally:
            _sync_running = False

    threading.Thread(target=run, daemon=True).start()
    return True

# ── Face indexing (download-encode-delete per photo) ──────────────────────────

def _load_resized(path: str, max_dim: int = 800):
    """Load an image and shrink it to max_dim on the longest side before encoding."""
    from PIL import Image
    import numpy as np_
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np_.array(img)


def build_index(progress_cb=None):
    """Download each photo from Dropbox temporarily, encode, delete immediately."""
    fr, np = require_fr()
    if fr is None:
        return {'success': False, 'error': 'face_recognition not installed.'}

    if not MANIFEST_FILE.exists():
        return {'success': False, 'error': 'No file list yet — sync from Dropbox first.'}

    manifest = json.loads(MANIFEST_FILE.read_text())
    if not manifest:
        return {'success': False, 'error': 'Manifest is empty.'}

    dbx, err = require_dbx()
    if err:
        return {'success': False, 'error': err}

    # Load existing index so we can skip already-encoded photos (resume support)
    existing = {}
    if ENCODINGS_FILE.exists():
        try:
            with open(ENCODINGS_FILE, 'rb') as fh:
                for d in pickle.load(fh):
                    existing[d['dropbox_path']] = d
        except Exception:
            pass

    indexed = []
    skipped = []
    total   = len(manifest)

    for i, item in enumerate(manifest):
        if progress_cb and i % 5 == 0:
            progress_cb(f'Processing {i+1} of {total}…')

        # Already indexed — reuse
        if item['dropbox_path'] in existing:
            indexed.append(existing[item['dropbox_path']])
            continue

        tmp_path = None
        try:
            suffix = Path(item['name']).suffix or '.jpg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            _, response = dbx.files_download(item['dropbox_path'])
            Path(tmp_path).write_bytes(response.content)

            # Resize large images before encoding — major speedup on high-res photos
            img = _load_resized(tmp_path, max_dim=800)
            encs = fr.face_encodings(img, model='small')
            if encs:
                indexed.append({
                    'encoding':     encs[0],
                    'folder':       item['folder'],
                    'name':         item['name'],
                    'dropbox_path': item['dropbox_path'],
                })
            else:
                skipped.append(item['name'])
        except Exception as e:
            skipped.append(f"{item['name']} ({e})")
        finally:
            if tmp_path and Path(tmp_path).exists():
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # Checkpoint save every 50 photos so progress isn't lost if interrupted
        if len(indexed) % 50 == 0 and indexed:
            with open(ENCODINGS_FILE, 'wb') as fh:
                pickle.dump(indexed, fh)

    # Final save — local + Dropbox
    with open(ENCODINGS_FILE, 'wb') as fh:
        pickle.dump(indexed, fh)
    _save_index_to_dropbox(indexed)

    return {
        'success':       True,
        'indexed':       len(indexed),
        'skipped':       len(skipped),
        'skipped_names': skipped[:20],
    }


def _start_index_background():
    global _index_running
    with _index_lock:
        if _index_running:
            return False
        _index_running = True

    def run():
        global _index_running
        try:
            def cb(msg):
                _write_status(INDEX_STATUS_FILE, {'state': 'running', 'message': msg})
            _write_status(INDEX_STATUS_FILE, {'state': 'running', 'message': 'Starting…'})
            result = build_index(progress_cb=cb)
            if result.get('success'):
                _write_status(INDEX_STATUS_FILE, {
                    'state':         'done',
                    'success':       True,
                    'indexed':       result['indexed'],
                    'skipped':       result['skipped'],
                    'skipped_names': result.get('skipped_names', []),
                })
            else:
                _write_status(INDEX_STATUS_FILE, {
                    'state': 'done', 'success': False, 'error': result['error'],
                })
        except Exception as e:
            _write_status(INDEX_STATUS_FILE, {'state': 'done', 'success': False, 'error': str(e)})
        finally:
            _index_running = False

    threading.Thread(target=run, daemon=True).start()
    return True

# ── Face search ───────────────────────────────────────────────────────────────

def search_photo(image_path: str):
    fr, np = require_fr()
    if fr is None:
        return None, 'face_recognition not installed.'

    if not ENCODINGS_FILE.exists():
        return None, 'No index yet — sync from Dropbox, then click "Build Index".'

    with open(ENCODINGS_FILE, 'rb') as fh:
        data = pickle.load(fh)

    if not data:
        return None, 'Index is empty. Sync and Build Index first.'

    img  = _load_resized(image_path, max_dim=800)
    encs = fr.face_encodings(img, model='small')

    if not encs:
        return None, 'No face detected. Try a clearer, well-lit photo of the framed portrait.'

    known = np.array([d['encoding'] for d in data])
    dists = fr.face_distance(known, encs[0])
    top   = sorted(range(len(dists)), key=lambda i: dists[i])[:3]

    matches = []
    for idx in top:
        d = float(dists[idx])
        if d < 0.65:
            matches.append({
                'folder':       data[idx]['folder'],
                'name':         data[idx]['name'],
                'dropbox_path': data[idx]['dropbox_path'],
                'confidence':   round((1 - d) * 100, 1),
                'distance':     round(d, 4),
            })

    if not matches:
        best = data[top[0]]
        conf = round((1 - float(dists[top[0]])) * 100, 1)
        return None, (
            f'No confident match found. '
            f'Closest: {best["name"]} ({conf}% — below threshold). '
            f'Try a clearer photo of the frame.'
        )

    return matches, None


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Photo Frame ID</title>
<style>
:root{
  --blue:#1a56db;--blue-dk:#1e429f;--blue-lt:#eff6ff;
  --green:#0e9f6e;--amber:#f59e0b;--red:#dc2626;
  --bg:#f3f4f6;--card:#fff;--text:#111827;--muted:#6b7280;--border:#e5e7eb;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--bg);color:var(--text);min-height:100vh}

header{background:var(--blue);color:#fff;padding:14px 20px;
       display:flex;align-items:center;gap:12px;
       box-shadow:0 2px 10px rgba(0,0,0,.2)}
header h1{font-size:18px;font-weight:700}
header p{font-size:12px;opacity:.7;margin-top:2px}

.wrap{max-width:980px;margin:0 auto;padding:18px 16px}

.sbar{background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:12px 16px;margin-bottom:16px;display:flex;flex-wrap:wrap;
      align-items:center;justify-content:space-between;gap:10px}
.sinfo{display:flex;align-items:center;gap:9px;font-size:13.5px;flex:1;min-width:0}
.dot{width:9px;height:9px;border-radius:50%;background:#d1d5db;flex-shrink:0}
.dot.g{background:var(--green)}.dot.y{background:var(--amber)}.dot.r{background:var(--red)}
.sactions{display:flex;gap:8px;flex-wrap:wrap}

.btn{padding:8px 15px;border-radius:8px;border:none;cursor:pointer;
     font-size:13.5px;font-weight:500;display:inline-flex;align-items:center;
     gap:6px;transition:.15s;white-space:nowrap}
.btn-blue{background:var(--blue);color:#fff}.btn-blue:hover{background:var(--blue-dk)}
.btn-out{background:#fff;color:var(--blue);border:1.5px solid var(--blue)}
.btn-out:hover{background:var(--blue-lt)}
.btn-sm{padding:6px 12px;font-size:12.5px}
.btn:disabled{opacity:.45;cursor:not-allowed}

.upzone{background:var(--card);border:2.5px dashed var(--border);border-radius:14px;
        padding:44px 20px;text-align:center;cursor:pointer;transition:.2s;
        margin-bottom:18px}
.upzone:hover,.upzone.over{border-color:var(--blue);background:var(--blue-lt)}
.upzone .ico{font-size:48px;margin-bottom:12px}
.upzone p{color:var(--muted);font-size:15px;margin-bottom:5px}
.upzone small{color:#9ca3af;font-size:13px}

.mob-btns{display:none;gap:12px;margin-bottom:18px}
.mob-btn{flex:1;padding:18px 10px;border-radius:14px;border:none;cursor:pointer;
         font-size:15px;font-weight:600;display:flex;flex-direction:column;
         align-items:center;gap:8px;transition:.15s;background:var(--card);
         border:1.5px solid var(--border);color:var(--text)}
.mob-btn:hover{background:var(--blue-lt);border-color:var(--blue)}
.mob-btn .mico{font-size:36px}

input[type=file]{display:none}

.results{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}
.clabel{font-size:11px;font-weight:700;color:var(--muted);
        text-transform:uppercase;letter-spacing:.07em;margin-bottom:12px}

.pbox{width:100%;max-height:320px;overflow:hidden;border-radius:10px;
      background:var(--bg);display:flex;align-items:center;justify-content:center}
.pbox img{width:100%;max-height:320px;object-fit:cover;object-position:top;border-radius:10px}
.pph{width:100%;height:240px;background:var(--bg);border-radius:10px;
     display:flex;flex-direction:column;align-items:center;justify-content:center;
     color:var(--muted);gap:8px;font-size:13px}

.mname{font-size:21px;font-weight:800;margin:12px 0 3px;line-height:1.2}
.mfolder{font-size:13px;color:var(--muted);margin-bottom:10px;
         display:flex;align-items:center;gap:5px}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 11px;
       border-radius:100px;font-size:12px;font-weight:600;margin-bottom:12px}
.hi{background:#ecfdf5;color:#065f46}.med{background:#fffbeb;color:#92400e}
.lo{background:#fef2f2;color:var(--red)}
.cbar-lbl{display:flex;justify-content:space-between;font-size:12.5px;
          color:var(--muted);margin-bottom:4px}
.cbar{height:8px;background:var(--bg);border-radius:100px;overflow:hidden}
.cfill{height:100%;border-radius:100px;
       background:linear-gradient(90deg,var(--blue),var(--green));
       transition:width .6s cubic-bezier(.4,0,.2,1)}
.alts{margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.alt-title{font-size:11px;font-weight:700;color:var(--muted);
           text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}
.alt-row{display:flex;justify-content:space-between;align-items:center;
         font-size:13px;padding:5px 0;border-bottom:1px solid #f3f4f6}
.alt-row:last-child{border:none}
.alt-pct{background:var(--bg);padding:2px 8px;border-radius:100px;
         font-size:11px;font-weight:600;color:var(--muted)}

.err{background:#fef2f2;border:1px solid #fecaca;color:var(--red);
     border-radius:10px;padding:14px;font-size:13.5px;line-height:1.5}
.searching{text-align:center;padding:60px 20px;color:var(--muted)}
.bspin{width:38px;height:38px;border:3px solid var(--bg);
       border-top-color:var(--blue);border-radius:50%;
       animation:sp .75s linear infinite;margin:0 auto 12px}
.spin{width:15px;height:15px;border:2px solid rgba(255,255,255,.3);
      border-top-color:#fff;border-radius:50%;
      animation:sp .75s linear infinite;display:inline-block}
@keyframes sp{to{transform:rotate(360deg)}}

@media(max-width:640px){
  .results{grid-template-columns:1fr}
  .upzone{display:none}
  .mob-btns{display:flex}
  .mname{font-size:19px}
}
</style>
</head>
<body>

<header>
  <span style="font-size:26px">📷</span>
  <div style="flex:1">
    <h1>Photo Frame Identification System</h1>
    <p>Scan or upload a framed photo to identify the student</p>
  </div>
  <a href="/logout" style="color:rgba(255,255,255,.7);font-size:13px;text-decoration:none;
     padding:6px 12px;border:1px solid rgba(255,255,255,.3);border-radius:7px;
     white-space:nowrap">Sign out</a>
</header>

<div class="wrap">

  <div class="sbar">
    <div class="sinfo">
      <div class="dot" id="dot"></div>
      <span id="stxt">Checking…</span>
    </div>
    <div class="sactions">
      <button class="btn btn-out btn-sm" id="sync-btn" onclick="doSync()">⬇ Sync from Dropbox</button>
      <button class="btn btn-blue btn-sm" id="idx-btn" onclick="doIndex()">⚙ Build Index</button>
    </div>
  </div>

  <div class="upzone" id="zone"
       onclick="document.getElementById('fi-gallery').click()"
       ondrop="onDrop(event)" ondragover="onDragOver(event)" ondragleave="onDragLeave(event)">
    <div class="ico">🖼️</div>
    <p><strong>Drop the framed photo here</strong></p>
    <small>or click to browse &nbsp;·&nbsp; JPG, PNG, HEIC</small>
  </div>

  <div class="mob-btns">
    <button class="mob-btn" onclick="document.getElementById('fi-camera').click()">
      <span class="mico">📷</span>
      <span>Take Photo</span>
    </button>
    <button class="mob-btn" onclick="document.getElementById('fi-gallery').click()">
      <span class="mico">🖼️</span>
      <span>Choose Photo</span>
    </button>
  </div>

  <input type="file" id="fi-camera"  accept="image/*" capture="environment" onchange="onFile(event)">
  <input type="file" id="fi-gallery" accept="image/*" onchange="onFile(event)">

  <div id="results" style="display:none">
    <div class="results">
      <div class="card">
        <div class="clabel">Uploaded Photo</div>
        <div id="upv"></div>
      </div>
      <div class="card">
        <div class="clabel">Matched Student</div>
        <div id="matchbox"></div>
      </div>
    </div>
  </div>

</div>

<script>
var _syncPoll  = null;
var _indexPoll = null;

window.onload = function(){
  loadStatus();
  resumePolling();
};

async function resumePolling(){
  try{
    const s = await fetch('/api/sync_status').then(r=>r.json());
    if(s.state==='running'){
      setLoading('sync-btn','⬇','Syncing…');
      document.getElementById('dot').className='dot y';
      document.getElementById('stxt').textContent=s.message||'Syncing…';
      scheduleSyncPoll();
    }
  }catch(e){}
  try{
    const s = await fetch('/api/index_status').then(r=>r.json());
    if(s.state==='running'){
      setLoading('idx-btn','⚙','Indexing…');
      document.getElementById('dot').className='dot y';
      document.getElementById('stxt').textContent=s.message||'Building index…';
      scheduleIndexPoll();
    }
  }catch(e){}
}

async function loadStatus(){
  try{
    const d = await fetch('/api/status').then(r=>r.json());
    const dot = document.getElementById('dot');
    const txt = document.getElementById('stxt');

    if(!d.fr_installed){
      dot.className='dot r';
      txt.innerHTML='<strong>face_recognition error:</strong> '+(d.fr_error||'unknown — check Railway Deploy Logs');
      return;
    }
    if(!d.dropbox_configured){
      dot.className='dot y';
      txt.textContent='Dropbox token not configured — set DROPBOX_TOKEN in Railway variables.';
      return;
    }
    if(d.manifest_count===0){
      dot.className='dot y';
      txt.textContent='No files listed yet — click "Sync from Dropbox" to start.';
    } else if(d.indexed===0){
      dot.className='dot y';
      txt.textContent=d.manifest_count+' photos found in Dropbox · Not indexed yet — click "Build Index".';
    } else {
      dot.className='dot g';
      txt.textContent='✓ '+d.indexed+' students indexed · '+d.manifest_count+' photos in Dropbox';
    }
  }catch(e){
    document.getElementById('stxt').textContent='Cannot reach server.';
  }
}

/* ── Sync ── */
async function doSync(){
  setLoading('sync-btn','⬇','Starting…');
  document.getElementById('dot').className='dot y';
  document.getElementById('stxt').textContent='Listing files in Dropbox…';
  try{
    const d = await fetch('/api/sync',{method:'POST'}).then(r=>r.json());
    scheduleSyncPoll();
  }catch(e){
    document.getElementById('stxt').textContent='Could not start sync.';
    resetBtn('sync-btn','⬇ Sync from Dropbox');
  }
}

function scheduleSyncPoll(){ clearTimeout(_syncPoll); _syncPoll=setTimeout(pollSync,2000); }

async function pollSync(){
  try{
    const d = await fetch('/api/sync_status').then(r=>r.json());
    if(d.state==='running'){
      document.getElementById('stxt').textContent=d.message||'Syncing…';
      scheduleSyncPoll();
    } else if(d.state==='done'){
      if(d.success){
        document.getElementById('stxt').textContent=
          '✓ Found '+d.found+' photos in Dropbox. Now click "Build Index".';
        document.getElementById('dot').className='dot y';
      } else {
        document.getElementById('stxt').textContent='Sync error: '+d.error;
        document.getElementById('dot').className='dot r';
      }
      resetBtn('sync-btn','⬇ Sync from Dropbox');
    } else {
      resetBtn('sync-btn','⬇ Sync from Dropbox');
      loadStatus();
    }
  }catch(e){
    scheduleSyncPoll();
  }
}

/* ── Index ── */
async function doIndex(){
  setLoading('idx-btn','⚙','Starting…');
  document.getElementById('dot').className='dot y';
  document.getElementById('stxt').textContent='Starting index build…';
  try{
    const d = await fetch('/api/index',{method:'POST'}).then(r=>r.json());
    document.getElementById('stxt').textContent=
      'Building face index — downloading & processing each photo. This takes a while…';
    scheduleIndexPoll();
  }catch(e){
    document.getElementById('stxt').textContent='Could not start index build.';
    resetBtn('idx-btn','⚙ Build Index');
  }
}

function scheduleIndexPoll(){ clearTimeout(_indexPoll); _indexPoll=setTimeout(pollIndex,3000); }

async function pollIndex(){
  try{
    const d = await fetch('/api/index_status').then(r=>r.json());
    if(d.state==='running'){
      document.getElementById('stxt').textContent=d.message||'Building index…';
      scheduleIndexPoll();
    } else if(d.state==='done'){
      if(d.success){
        document.getElementById('dot').className='dot g';
        let msg='✓ '+d.indexed+' students indexed';
        if(d.skipped>0) msg+=' · '+d.skipped+' skipped (no face detected)';
        document.getElementById('stxt').textContent=msg;
      } else {
        document.getElementById('dot').className='dot r';
        document.getElementById('stxt').textContent='Index error: '+d.error;
      }
      resetBtn('idx-btn','⚙ Build Index');
    } else {
      resetBtn('idx-btn','⚙ Build Index');
      loadStatus();
    }
  }catch(e){
    scheduleIndexPoll();
  }
}

function setLoading(id,icon,label){
  const b=document.getElementById(id);
  b.disabled=true;
  b.innerHTML='<span class="spin"></span> '+label;
}
function resetBtn(id,label){
  const b=document.getElementById(id);
  b.disabled=false;
  b.innerHTML=label;
}

function onDragOver(e){e.preventDefault();document.getElementById('zone').classList.add('over')}
function onDragLeave(){document.getElementById('zone').classList.remove('over')}
function onDrop(e){
  e.preventDefault();document.getElementById('zone').classList.remove('over');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')) processFile(f);
}
function onFile(e){const f=e.target.files[0];if(f)processFile(f);e.target.value='';}

function processFile(file){
  document.getElementById('results').style.display='block';
  const reader=new FileReader();
  reader.onload=e=>{
    document.getElementById('upv').innerHTML=
      '<div class="pbox"><img src="'+e.target.result+'" alt="Uploaded"></div>';
  };
  reader.readAsDataURL(file);
  document.getElementById('matchbox').innerHTML=
    '<div class="searching"><div class="bspin"></div><p>Searching student records…</p></div>';
  const fd=new FormData(); fd.append('image',file);
  fetch('/api/search',{method:'POST',body:fd})
    .then(r=>r.json())
    .then(data=>{
      if(data.error){
        document.getElementById('matchbox').innerHTML=
          '<div class="err">⚠️ '+data.error+'</div>'; return;
      }
      const m=data.matches; const best=m[0];
      const imgUrl='/api/photo?token='+encodeURIComponent(best.token)+'&path='+encodeURIComponent(best.dropbox_path);
      let bc='lo',bt='? Low Confidence';
      if(best.confidence>=85){bc='hi';bt='✓ High Confidence';}
      else if(best.confidence>=70){bc='med';bt='~ Medium Confidence';}
      let altHtml='';
      if(m.length>1){
        altHtml='<div class="alts"><div class="alt-title">Other possible matches</div>'+
          m.slice(1).map(x=>'<div class="alt-row"><div><strong>'+x.name+'</strong>'+
            '<div style="font-size:11px;color:var(--muted)">'+x.folder+'</div></div>'+
            '<span class="alt-pct">'+x.confidence+'%</span></div>').join('')+'</div>';
      }
      document.getElementById('matchbox').innerHTML=
        '<div class="pbox"><img src="'+imgUrl+'" alt="Match"'+
        ' onerror="this.closest(\'.pbox\').innerHTML=\'<div class=pph>📷<span>Photo unavailable</span></div>\'"></div>'+
        '<div class="mname">'+best.name+'</div>'+
        '<div class="mfolder">📁 '+best.folder+'</div>'+
        '<span class="badge '+bc+'">'+bt+'</span>'+
        '<div class="cbar-lbl"><span>Match confidence</span><span>'+best.confidence+'%</span></div>'+
        '<div class="cbar"><div class="cfill" style="width:'+best.confidence+'%"></div></div>'+
        altHtml;
    })
    .catch(()=>{
      document.getElementById('matchbox').innerHTML=
        '<div class="err">⚠️ Connection error. Please try again.</div>';
    });
}
</script>
</body>
</html>"""


# ── Login page ────────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign In — Photo Frame ID</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f3f4f6;min-height:100vh;display:flex;align-items:center;
     justify-content:center;padding:20px}
.card{background:#fff;border-radius:16px;padding:36px 32px;width:100%;
      max-width:380px;box-shadow:0 4px 24px rgba(0,0,0,.08)}
.logo{text-align:center;font-size:40px;margin-bottom:12px}
h1{text-align:center;font-size:20px;font-weight:700;color:#111;margin-bottom:4px}
p{text-align:center;color:#6b7280;font-size:14px;margin-bottom:28px}
label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:6px}
input[type=password]{width:100%;padding:11px 14px;border:1.5px solid #e5e7eb;
  border-radius:9px;font-size:15px;outline:none;transition:.15s}
input[type=password]:focus{border-color:#1a56db;box-shadow:0 0 0 3px #eff6ff}
.btn{width:100%;padding:12px;background:#1a56db;color:#fff;border:none;
     border-radius:9px;font-size:15px;font-weight:600;cursor:pointer;
     margin-top:14px;transition:.15s}
.btn:hover{background:#1e429f}
.err{background:#fef2f2;color:#dc2626;border-radius:8px;padding:10px 14px;
     font-size:13.5px;margin-bottom:14px;border:1px solid #fecaca}
</style>
</head>
<body>
<div class="card">
  <div class="logo">📷</div>
  <h1>Photo Frame ID System</h1>
  <p>Sign in to continue</p>
  {% if error %}<div class="err">Incorrect password. Please try again.</div>{% endif %}
  <form method="POST">
    <label for="pw">Password</label>
    <input type="password" id="pw" name="password" autofocus placeholder="Enter password">
    <button type="submit" class="btn">Sign In</button>
  </form>
</div>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if not APP_PASSWORD:
        return redirect(url_for('index'))
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        return render_template_string(LOGIN_HTML, error=True)
    return render_template_string(LOGIN_HTML, error=False)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/')
@login_required
def index():
    return render_template_string(HTML)


@app.route('/api/status')
@login_required
def api_status():
    fr, _ = require_fr()

    manifest_count = 0
    if MANIFEST_FILE.exists():
        try:
            manifest_count = len(json.loads(MANIFEST_FILE.read_text()))
        except Exception:
            pass

    indexed = 0
    if ENCODINGS_FILE.exists():
        try:
            with open(ENCODINGS_FILE, 'rb') as fh:
                indexed = len(pickle.load(fh))
        except Exception:
            pass

    return jsonify({
        'fr_installed':       fr is not None,
        'fr_error':           _fr_error,
        'dropbox_configured': bool(DROPBOX_TOKEN),
        'manifest_count':     manifest_count,
        'indexed':            indexed,
    })


@app.route('/api/sync', methods=['POST'])
@login_required
def api_sync():
    started = _start_sync_background()
    if not started:
        return jsonify({'started': False, 'already_running': True})
    return jsonify({'started': True})


@app.route('/api/sync_status')
@login_required
def api_sync_status():
    return jsonify(_read_status(SYNC_STATUS_FILE))


@app.route('/api/index', methods=['POST'])
@login_required
def api_index():
    started = _start_index_background()
    if not started:
        return jsonify({'started': False, 'already_running': True})
    return jsonify({'started': True})


@app.route('/api/index_status')
@login_required
def api_index_status():
    return jsonify(_read_status(INDEX_STATUS_FILE))


@app.route('/api/search', methods=['POST'])
@login_required
def api_search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file   = request.files['image']
    suffix = Path(file.filename).suffix or '.jpg'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        matches, err = search_photo(tmp.name)
        os.unlink(tmp.name)
    if err:
        return jsonify({'error': err})

    for m in matches:
        m['token'] = sign_photo_url(m['dropbox_path'])
    return jsonify({'matches': matches})


@app.route('/api/photo')
@login_required
def api_photo():
    """Proxy a student photo from Dropbox — only with a valid signed token."""
    dropbox_path = request.args.get('path', '')
    token        = request.args.get('token', '')

    if not dropbox_path or not token:
        return 'Forbidden', 403
    if not verify_photo_token(dropbox_path, token):
        return 'Link expired or invalid', 403

    dbx, err = require_dbx()
    if err:
        return err, 500

    try:
        _, response = dbx.files_download(dropbox_path)
        suffix = Path(dropbox_path).suffix.lower()
        mime = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',  '.webp': 'image/webp',
            '.bmp': 'image/bmp',  '.tiff': 'image/tiff',
        }.get(suffix, 'image/jpeg')
        return Response(response.content, mimetype=mime)
    except Exception as e:
        return f'Error fetching photo: {e}', 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Photo Frame ID System → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
