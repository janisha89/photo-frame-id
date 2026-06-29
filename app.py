#!/usr/bin/env python3
"""
Photo Frame Identification System
Hosted version: Dropbox photo sync + face recognition + mobile-ready UI.
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
from flask import (Flask, request, jsonify, send_file,
                   render_template_string, session, redirect, url_for)

app = Flask(__name__)

# ── Config (set via environment variables on Railway) ────────────────────────
BASE_DIR       = Path(__file__).parent
PHOTOS_DIR     = BASE_DIR / 'photos'          # local cache synced from Dropbox
ENCODINGS_FILE = BASE_DIR / 'face_index.pkl'
SYNC_STATUS_FILE  = BASE_DIR / 'sync_status.json'
INDEX_STATUS_FILE = BASE_DIR / 'index_status.json'
IMG_EXTS       = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}

DROPBOX_TOKEN  = os.environ.get('DROPBOX_TOKEN', '')
DROPBOX_FOLDER = os.environ.get('DROPBOX_FOLDER', '/Student Photos')
APP_PASSWORD   = os.environ.get('APP_PASSWORD', '')   # blank = no password required

# Secret key: signs sessions + photo URL tokens.
_SECRET = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.secret_key = _SECRET

PHOTOS_DIR.mkdir(exist_ok=True)

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if APP_PASSWORD and not session.get('authenticated'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def sign_photo_url(rel_path: str) -> str:
    expiry  = int(time.time()) + 3600
    message = f"{rel_path}:{expiry}".encode()
    sig     = hmac.new(_SECRET.encode(), message, hashlib.sha256).hexdigest()
    return f"{expiry}:{sig}"


def verify_photo_token(rel_path: str, token: str) -> bool:
    try:
        expiry_str, sig = token.split(':', 1)
        expiry = int(expiry_str)
        if time.time() > expiry:
            return False
        message  = f"{rel_path}:{expiry}".encode()
        expected = hmac.new(_SECRET.encode(), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

# ── Lazy imports ─────────────────────────────────────────────────────────────

_fr_error = None

def _patch_pkg_resources():
    """
    Python 3.12 venvs don't include pkg_resources by default.
    face_recognition_models uses it to locate model files.
    We inject a minimal shim so the import succeeds.
    """
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


_patch_pkg_resources()   # run once at startup


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
        return None, 'dropbox package not installed — run: pip install dropbox'


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


# ── Dropbox sync ──────────────────────────────────────────────────────────────

def sync_dropbox(progress_cb=None):
    """Download student photos from Dropbox to local PHOTOS_DIR."""
    dbx, err = require_dbx()
    if err:
        return {'success': False, 'error': err}

    import dropbox as dbx_module

    try:
        if progress_cb: progress_cb('Listing files in Dropbox…')
        result   = dbx.files_list_folder(DROPBOX_FOLDER, recursive=True)
        entries  = list(result.entries)
        while result.has_more:
            result  = dbx.files_list_folder_continue(result.cursor)
            entries += result.entries
    except Exception as e:
        return {'success': False, 'error': f'Cannot access Dropbox folder "{DROPBOX_FOLDER}": {e}'}

    downloaded = 0
    skipped    = 0
    errors     = []
    total_imgs = sum(
        1 for e in entries
        if hasattr(e, 'name') and Path(e.name).suffix.lower() in IMG_EXTS
    )

    for entry in entries:
        rel   = entry.path_display[len(DROPBOX_FOLDER):].lstrip('/')
        local = PHOTOS_DIR / rel

        if isinstance(entry, dbx_module.files.FolderMetadata):
            local.mkdir(parents=True, exist_ok=True)

        elif isinstance(entry, dbx_module.files.FileMetadata):
            if Path(entry.name).suffix.lower() not in IMG_EXTS:
                continue
            if local.exists() and local.stat().st_size == entry.size:
                skipped += 1
                continue
            try:
                local.parent.mkdir(parents=True, exist_ok=True)
                _, response = dbx.files_download(entry.path_display)
                local.write_bytes(response.content)
                downloaded += 1
                if progress_cb:
                    progress_cb(
                        f'Downloaded {downloaded} of {total_imgs - skipped} files…'
                    )
            except Exception as e:
                errors.append(f'{entry.name}: {e}')

    return {
        'success':    True,
        'downloaded': downloaded,
        'skipped':    skipped,
        'errors':     errors[:10],
    }


def _start_sync_background():
    """Start sync in a background thread. Returns False if already running."""
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
            _write_status(SYNC_STATUS_FILE, {'state': 'running', 'message': 'Starting sync…'})
            result = sync_dropbox(progress_cb=cb)
            if result.get('success'):
                _write_status(SYNC_STATUS_FILE, {
                    'state':      'done',
                    'success':    True,
                    'downloaded': result['downloaded'],
                    'skipped':    result['skipped'],
                    'errors':     result.get('errors', []),
                })
            else:
                _write_status(SYNC_STATUS_FILE, {
                    'state':   'done',
                    'success': False,
                    'error':   result['error'],
                })
        except Exception as e:
            _write_status(SYNC_STATUS_FILE, {'state': 'done', 'success': False, 'error': str(e)})
        finally:
            _sync_running = False

    threading.Thread(target=run, daemon=True).start()
    return True


# ── Photo scanning & indexing ─────────────────────────────────────────────────

def scan_photos():
    photos = []
    if not PHOTOS_DIR.exists():
        return photos
    for folder in sorted(PHOTOS_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):
            continue
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                photos.append({
                    'path':   str(f),
                    'folder': folder.name,
                    'name':   f.stem,
                })
    return photos


def build_index(progress_cb=None):
    fr, np = require_fr()
    if fr is None:
        return {'success': False, 'error': 'face_recognition not installed.'}

    photos = scan_photos()
    if not photos:
        return {'success': False, 'error': 'No photos found. Sync from Dropbox first.'}

    indexed = []
    skipped = []
    total   = len(photos)

    for i, p in enumerate(photos):
        if progress_cb and i % 5 == 0:
            progress_cb(f'Processing photo {i+1} of {total}…')
        try:
            img  = fr.load_image_file(p['path'])
            encs = fr.face_encodings(img, model='large')
            if encs:
                indexed.append({
                    'encoding': encs[0],
                    'folder':   p['folder'],
                    'name':     p['name'],
                    'path':     p['path'],
                })
            else:
                skipped.append(p['name'])
        except Exception as e:
            skipped.append(f"{p['name']} ({e})")

    with open(ENCODINGS_FILE, 'wb') as fh:
        pickle.dump(indexed, fh)

    return {
        'success':       True,
        'indexed':       len(indexed),
        'skipped':       len(skipped),
        'skipped_names': skipped[:20],
    }


def _start_index_background():
    """Start index build in a background thread. Returns False if already running."""
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
                    'state':   'done',
                    'success': False,
                    'error':   result['error'],
                })
        except Exception as e:
            _write_status(INDEX_STATUS_FILE, {'state': 'done', 'success': False, 'error': str(e)})
        finally:
            _index_running = False

    threading.Thread(target=run, daemon=True).start()
    return True


def search_photo(image_path: str):
    fr, np = require_fr()
    if fr is None:
        return None, 'face_recognition not installed.'

    if not ENCODINGS_FILE.exists():
        return None, 'No index yet — sync photos from Dropbox, then click "Build Index".'

    with open(ENCODINGS_FILE, 'rb') as fh:
        data = pickle.load(fh)

    if not data:
        return None, 'Index is empty. Make sure photos were synced and indexed.'

    img  = fr.load_image_file(image_path)
    encs = fr.face_encodings(img, model='large')

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
                'folder':     data[idx]['folder'],
                'name':       data[idx]['name'],
                'path':       data[idx]['path'],
                'confidence': round((1 - d) * 100, 1),
                'distance':   round(d, 4),
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
  // Resume polling if a background task was already running
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
      txt.innerHTML='<strong>face_recognition error:</strong> ' + (d.fr_error||'unknown — check Railway Deploy Logs');
      return;
    }
    if(!d.dropbox_configured){
      dot.className='dot y';
      txt.textContent='Dropbox token not configured — set DROPBOX_TOKEN in Railway variables.';
      return;
    }
    const synced  = d.local_photos;
    const indexed = d.indexed;
    if(synced===0){
      dot.className='dot y';
      txt.textContent='No photos synced yet — click "Sync from Dropbox" to start.';
    } else if(indexed===0){
      dot.className='dot y';
      txt.textContent=synced+' photos synced · Not indexed yet — click "Build Index".';
    } else {
      dot.className='dot g';
      txt.textContent='✓ '+indexed+' students indexed · '+synced+' photos on server';
    }
  }catch(e){
    document.getElementById('stxt').textContent='Cannot reach server.';
  }
}

/* ── Sync ── */
async function doSync(){
  setLoading('sync-btn','⬇','Starting…');
  document.getElementById('dot').className='dot y';
  document.getElementById('stxt').textContent='Starting sync…';
  try{
    const d = await fetch('/api/sync',{method:'POST'}).then(r=>r.json());
    if(d.already_running){
      document.getElementById('stxt').textContent='Sync already in progress…';
    } else {
      document.getElementById('stxt').textContent='Syncing photos from Dropbox…';
    }
    scheduleSyncPoll();
  }catch(e){
    document.getElementById('stxt').textContent='Could not start sync.';
    resetBtn('sync-btn','⬇ Sync from Dropbox');
  }
}

function scheduleSyncPoll(){
  clearTimeout(_syncPoll);
  _syncPoll = setTimeout(pollSync, 2000);
}

async function pollSync(){
  try{
    const d = await fetch('/api/sync_status').then(r=>r.json());
    if(d.state==='running'){
      document.getElementById('stxt').textContent=d.message||'Syncing…';
      scheduleSyncPoll();
    } else if(d.state==='done'){
      if(d.success){
        document.getElementById('stxt').textContent=
          '✓ Sync complete — '+d.downloaded+' new, '+d.skipped+' unchanged. Now click "Build Index".';
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
    document.getElementById('stxt').textContent='Lost connection — retrying…';
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
    if(d.already_running){
      document.getElementById('stxt').textContent='Index build already in progress…';
    } else {
      document.getElementById('stxt').textContent='Building face index — this may take several minutes…';
    }
    scheduleIndexPoll();
  }catch(e){
    document.getElementById('stxt').textContent='Could not start index build.';
    resetBtn('idx-btn','⚙ Build Index');
  }
}

function scheduleIndexPoll(){
  clearTimeout(_indexPoll);
  _indexPoll = setTimeout(pollIndex, 2000);
}

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
    document.getElementById('stxt').textContent='Lost connection — retrying…';
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
      const imgUrl='/api/photo?rel='+encodeURIComponent(best.rel)+'&token='+encodeURIComponent(best.token);
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


# ── Login page HTML ───────────────────────────────────────────────────────────

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
    fr, _        = require_fr()
    local_photos = len(scan_photos())
    indexed      = 0
    if ENCODINGS_FILE.exists():
        with open(ENCODINGS_FILE, 'rb') as fh:
            indexed = len(pickle.load(fh))
    return jsonify({
        'fr_installed':       fr is not None,
        'fr_error':           _fr_error,
        'dropbox_configured': bool(DROPBOX_TOKEN),
        'local_photos':       local_photos,
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
        rel = os.path.relpath(m['path'], str(PHOTOS_DIR))
        m['token'] = sign_photo_url(rel)
        m['rel']   = rel
    return jsonify({'matches': matches})


@app.route('/api/photo')
@login_required
def api_photo():
    rel   = request.args.get('rel', '')
    token = request.args.get('token', '')

    if not rel or not token:
        return 'Forbidden', 403
    if not verify_photo_token(rel, token):
        return 'Link expired or invalid', 403

    abs_path    = (PHOTOS_DIR / rel).resolve()
    photos_root = PHOTOS_DIR.resolve()
    if not str(abs_path).startswith(str(photos_root)):
        return 'Forbidden', 403
    if not abs_path.exists():
        return 'Not found', 404

    return send_file(str(abs_path))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Photo Frame ID System → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
