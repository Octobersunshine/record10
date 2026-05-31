import os
import uuid
import hashlib
import json
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Set

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
UPLOAD_TMP_DIR = BASE_DIR / "uploads" / "tmp"
UPLOAD_FILES_DIR = BASE_DIR / "uploads" / "files"
META_FILE = BASE_DIR / "uploads" / "meta.json"
FILE_INDEX = BASE_DIR / "uploads" / "file_index.json"

UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FILES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="大文件分片上传API")

_meta_lock = threading.Lock()
_ws_connections: Dict[str, Set[WebSocket]] = {}


def _load_meta() -> dict:
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_file_index() -> dict:
    if FILE_INDEX.exists():
        with open(FILE_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_file_index(index: dict):
    with open(FILE_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _calc_file_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


async def _broadcast_progress(upload_id: str):
    if upload_id not in _ws_connections or not _ws_connections[upload_id]:
        return

    meta = _load_meta()
    if upload_id not in meta:
        return

    session = meta[upload_id]
    uploaded = session["uploaded_chunks"]
    total = session["total_chunks"]
    message = json.dumps({
        "upload_id": upload_id,
        "filename": session["filename"],
        "total_chunks": total,
        "uploaded_chunks": sorted(uploaded),
        "progress": round(len(uploaded) / total, 4) if total > 0 else 0,
        "is_complete": len(uploaded) == total,
    })

    for ws in list(_ws_connections[upload_id]):
        try:
            await ws.send_text(message)
        except Exception:
            pass


async def _broadcast_merge_result(upload_id: str, result: dict):
    if upload_id not in _ws_connections or not _ws_connections[upload_id]:
        return

    message = json.dumps({
        "upload_id": upload_id,
        "event": "merge_complete",
        **result,
    })

    for ws in list(_ws_connections[upload_id]):
        try:
            await ws.send_text(message)
        except Exception:
            pass


class InitRequest(BaseModel):
    filename: str
    file_size: int
    chunk_size: int
    total_chunks: int
    total_md5: Optional[str] = None


class MergeRequest(BaseModel):
    upload_id: str


@app.post("/upload/init")
async def init_upload(req: InitRequest):
    if req.total_md5:
        file_index = _load_file_index()
        if req.total_md5 in file_index:
            entry = file_index[req.total_md5]
            return {
                "fast_upload": True,
                "filename": entry["filename"],
                "file_size": entry["file_size"],
                "md5": req.total_md5,
                "output_path": entry["output_path"],
                "message": "秒传成功，文件已存在",
            }

    upload_id = uuid.uuid4().hex
    chunk_dir = UPLOAD_TMP_DIR / upload_id
    chunk_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta()
    meta[upload_id] = {
        "filename": req.filename,
        "file_size": req.file_size,
        "chunk_size": req.chunk_size,
        "total_chunks": req.total_chunks,
        "total_md5": req.total_md5,
        "uploaded_chunks": [],
    }
    _save_meta(meta)

    return {
        "fast_upload": False,
        "upload_id": upload_id,
        "chunk_dir": str(chunk_dir),
    }


@app.post("/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk_md5: str = Form(...),
    file: UploadFile = File(...),
):
    meta = _load_meta()
    if upload_id not in meta:
        raise HTTPException(status_code=404, detail="上传会话不存在")

    session = meta[upload_id]
    if chunk_index < 0 or chunk_index >= session["total_chunks"]:
        raise HTTPException(status_code=400, detail="分片索引越界")

    chunk_dir = UPLOAD_TMP_DIR / upload_id
    chunk_path = chunk_dir / str(chunk_index)

    if chunk_index in session["uploaded_chunks"] and chunk_path.exists():
        progress = len(session["uploaded_chunks"]) / session["total_chunks"]
        return {
            "upload_id": upload_id,
            "chunk_index": chunk_index,
            "uploaded_chunks": sorted(session["uploaded_chunks"]),
            "progress": round(progress, 4),
            "skipped": True,
        }

    content = await file.read()
    actual_md5 = hashlib.md5(content).hexdigest()
    if actual_md5 != chunk_md5:
        raise HTTPException(status_code=400, detail=f"分片MD5校验失败: 期望={chunk_md5}, 实际={actual_md5}")

    with open(chunk_path, "wb") as f:
        f.write(content)

    with _meta_lock:
        meta = _load_meta()
        session = meta.get(upload_id)
        if session and chunk_index not in session["uploaded_chunks"]:
            session["uploaded_chunks"].append(chunk_index)
            _save_meta(meta)
            asyncio.create_task(_broadcast_progress(upload_id))

    meta = _load_meta()
    session = meta[upload_id]
    progress = len(session["uploaded_chunks"]) / session["total_chunks"]

    return {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "uploaded_chunks": sorted(session["uploaded_chunks"]),
        "progress": round(progress, 4),
        "skipped": False,
    }


@app.get("/upload/progress")
async def get_progress(upload_id: str):
    meta = _load_meta()
    if upload_id not in meta:
        raise HTTPException(status_code=404, detail="上传会话不存在")

    session = meta[upload_id]
    uploaded = session["uploaded_chunks"]
    total = session["total_chunks"]

    return {
        "upload_id": upload_id,
        "filename": session["filename"],
        "total_chunks": total,
        "uploaded_chunks": sorted(uploaded),
        "progress": round(len(uploaded) / total, 4) if total > 0 else 0,
        "is_complete": len(uploaded) == total,
    }


@app.post("/upload/merge")
async def merge_chunks(req: MergeRequest):
    meta = _load_meta()
    upload_id = req.upload_id

    if upload_id not in meta:
        raise HTTPException(status_code=404, detail="上传会话不存在")

    session = meta[upload_id]
    uploaded = set(session["uploaded_chunks"])
    total = session["total_chunks"]
    expected = set(range(total))

    if uploaded != expected:
        missing = sorted(expected - uploaded)
        raise HTTPException(
            status_code=400,
            detail=f"分片未全部上传完毕: 缺失分片={missing}",
        )

    sorted_indices = sorted(uploaded)

    chunk_dir = UPLOAD_TMP_DIR / upload_id
    output_path = UPLOAD_FILES_DIR / session["filename"]

    md5_hasher = hashlib.md5()
    with open(output_path, "wb") as out_f:
        for i in sorted_indices:
            chunk_path = chunk_dir / str(i)
            if not chunk_path.exists():
                raise HTTPException(status_code=400, detail=f"分片 {i} 文件缺失")
            with open(chunk_path, "rb") as chunk_f:
                while True:
                    data = chunk_f.read(8192)
                    if not data:
                        break
                    out_f.write(data)
                    md5_hasher.update(data)

    merged_md5 = md5_hasher.hexdigest()
    expected_md5 = session.get("total_md5")

    md5_match = True
    if expected_md5 and expected_md5.lower() != merged_md5.lower():
        md5_match = False
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"文件总MD5校验失败: 期望={expected_md5}, 实际={merged_md5}",
        )

    shutil.rmtree(chunk_dir, ignore_errors=True)

    with _meta_lock:
        meta = _load_meta()
        if upload_id in meta:
            del meta[upload_id]
            _save_meta(meta)

    file_index = _load_file_index()
    file_index[merged_md5] = {
        "filename": session["filename"],
        "file_size": session["file_size"],
        "output_path": str(output_path),
    }
    _save_file_index(file_index)

    merge_result = {
        "upload_id": upload_id,
        "filename": session["filename"],
        "file_size": session["file_size"],
        "merged_md5": merged_md5,
        "md5_match": md5_match,
        "output_path": str(output_path),
    }

    asyncio.create_task(_broadcast_merge_result(upload_id, merge_result))

    return merge_result


@app.websocket("/ws/upload/{upload_id}")
async def ws_upload_progress(websocket: WebSocket, upload_id: str):
    await websocket.accept()
    if upload_id not in _ws_connections:
        _ws_connections[upload_id] = set()
    _ws_connections[upload_id].add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if upload_id in _ws_connections:
            _ws_connections[upload_id].discard(websocket)
            if not _ws_connections[upload_id]:
                del _ws_connections[upload_id]
    except Exception:
        if upload_id in _ws_connections:
            _ws_connections[upload_id].discard(websocket)
