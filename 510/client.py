import os
import sys
import json
import hashlib
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import websocket

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024
DEFAULT_CONCURRENCY = 5

_last_progress = {"pct": 0.0, "done": 0, "total": 0, "lock": threading.Lock()}
_ws_done = threading.Event()
_merge_result = None


def calc_file_md5(filepath: str, block_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _update_progress_bar():
    with _last_progress["lock"]:
        pct = _last_progress["pct"]
        done = _last_progress["done"]
        total = _last_progress["total"]
    bar_len = 40
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct*100:6.2f}% ({done}/{total})", end="", flush=True)


def _on_ws_message(ws, message):
    global _merge_result
    try:
        data = json.loads(message)
        event = data.get("event")
        if event == "merge_complete":
            _merge_result = data
            _ws_done.set()
            return
        pct = data.get("progress", 0.0)
        done = len(data.get("uploaded_chunks", []))
        total = data.get("total_chunks", 0)
        with _last_progress["lock"]:
            _last_progress["pct"] = pct
            _last_progress["done"] = done
            _last_progress["total"] = total
        _update_progress_bar()
    except Exception as e:
        print(f"  [WS] 消息解析失败: {e}")


def _on_ws_error(ws, error):
    print(f"\n  [WS] 错误: {error}")


def _on_ws_close(ws, close_status_code, close_msg):
    pass


def _start_ws_listener(upload_id: str):
    ws_url = f"{WS_URL}/ws/upload/{upload_id}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=_on_ws_message,
        on_error=_on_ws_error,
        on_close=_on_ws_close,
    )
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return ws


def _upload_single_chunk(filepath: str, filename: str, upload_id: str,
                         chunk_index: int, chunk_size: int, uploaded_chunks: set):
    if chunk_index in uploaded_chunks:
        return chunk_index, True

    with open(filepath, "rb") as f:
        f.seek(chunk_index * chunk_size)
        chunk_data = f.read(chunk_size)

    chunk_md5 = hashlib.md5(chunk_data).hexdigest()

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE_URL}/upload/chunk",
                data={
                    "upload_id": upload_id,
                    "chunk_index": str(chunk_index),
                    "chunk_md5": chunk_md5,
                },
                files={"file": (filename, chunk_data)},
                timeout=60,
            )
            if resp.status_code == 400:
                return chunk_index, False
            resp.raise_for_status()
            return chunk_index, True
        except Exception as e:
            if attempt == 2:
                print(f"\n  分片 {chunk_index} 上传失败: {e}")
                return chunk_index, False
            time.sleep(0.5 * (attempt + 1))


def upload_file(filepath: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                concurrency: int = DEFAULT_CONCURRENCY):
    global _merge_result
    filepath = str(Path(filepath).resolve())
    if not os.path.isfile(filepath):
        print(f"文件不存在: {filepath}")
        return

    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)
    total_chunks = (file_size + chunk_size - 1) // chunk_size

    print(f"文件: {filename}")
    print(f"大小: {file_size} bytes")
    print(f"分片: {total_chunks} 块 (每块 {chunk_size} bytes)")
    print(f"并发: {concurrency} 线程")

    print("\n计算文件MD5...")
    total_md5 = calc_file_md5(filepath)
    print(f"文件MD5: {total_md5}")

    print("\n[1/4] 初始化上传会话...")
    resp = requests.post(f"{BASE_URL}/upload/init", json={
        "filename": filename,
        "file_size": file_size,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "total_md5": total_md5,
    })
    resp.raise_for_status()
    data = resp.json()

    if data.get("fast_upload"):
        print(f"\n✓ 秒传成功！文件已存在")
        print(f"  文件名:   {data['filename']}")
        print(f"  文件大小: {data['file_size']} bytes")
        print(f"  MD5:      {data['md5']}")
        print(f"  存储路径: {data['output_path']}")
        print(f"  提示:     {data['message']}")
        return

    upload_id = data["upload_id"]
    print(f"上传会话已创建: upload_id={upload_id}")

    print("\n[2/4] 启动WebSocket进度监听...")
    _ws = _start_ws_listener(upload_id)
    time.sleep(0.3)

    print("\n[3/4] 查询已上传分片（断点续传）...")
    resp = requests.get(f"{BASE_URL}/upload/progress", params={"upload_id": upload_id})
    resp.raise_for_status()
    progress_data = resp.json()
    uploaded_chunks = set(progress_data["uploaded_chunks"])

    with _last_progress["lock"]:
        _last_progress["total"] = total_chunks
        _last_progress["done"] = len(uploaded_chunks)
        _last_progress["pct"] = len(uploaded_chunks) / total_chunks if total_chunks > 0 else 0

    if uploaded_chunks:
        print(f"发现已上传 {len(uploaded_chunks)} 个分片，跳过已上传部分")
    else:
        print("无已上传分片，从头开始上传")

    pending = [i for i in range(total_chunks) if i not in uploaded_chunks]
    print(f"\n[4/4] 并发上传剩余 {len(pending)} 个分片...")
    _update_progress_bar()

    success_count = len(uploaded_chunks)
    failed_indices = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _upload_single_chunk,
                filepath, filename, upload_id, idx, chunk_size, uploaded_chunks
            ): idx for idx in pending
        }
        for future in as_completed(futures):
            idx, ok = future.result()
            if ok:
                success_count += 1
                with _last_progress["lock"]:
                    _last_progress["done"] = success_count
                    _last_progress["pct"] = success_count / total_chunks if total_chunks > 0 else 0
                _update_progress_bar()
            else:
                failed_indices.append(idx)

    if failed_indices:
        print(f"\n\n以下分片上传失败: {failed_indices}")
        return

    print("\n\n所有分片上传完毕，请求合并...")
    resp = requests.post(f"{BASE_URL}/upload/merge", json={"upload_id": upload_id})
    resp.raise_for_status()

    timeout_start = time.time()
    while not _ws_done.is_set() and (time.time() - timeout_start) < 10:
        time.sleep(0.1)

    if _merge_result:
        print(f"\n[WS 推送] 合并完成!")
        merge_result = _merge_result
    else:
        merge_result = resp.json()
        print(f"\n合并完成!")

    print(f"  文件名:   {merge_result['filename']}")
    print(f"  文件大小: {merge_result['file_size']} bytes")
    print(f"  合并MD5:  {merge_result['merged_md5']}")
    print(f"  MD5校验:  {'✓ 通过' if merge_result['md5_match'] else '✗ 不匹配'}")
    print(f"  存储路径: {merge_result['output_path']}")

    _ws.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python client.py <文件路径> [分片大小(字节)] [并发数]")
        print(f"  默认分片大小: {DEFAULT_CHUNK_SIZE} bytes (5MB)")
        print(f"  默认并发数:   {DEFAULT_CONCURRENCY}")
        sys.exit(1)

    fp = sys.argv[1]
    cs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNK_SIZE
    cc = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CONCURRENCY
    upload_file(fp, cs, cc)
