import asyncio
import base64
import io
import math
import re
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Batch Image Resize API")

ALLOWED_FORMATS = {"jpeg", "jpg", "png", "webp"}
FORMAT_TO_PIL = {
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "png": "PNG",
    "webp": "WebP",
}
FORMAT_TO_MIME = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


class OutputFormat(str, Enum):
    jpeg = "jpeg"
    jpg = "jpg"
    png = "png"
    webp = "webp"


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ImageInput(BaseModel):
    base64: Optional[str] = Field(None, description="Base64 encoded image data")
    url: Optional[str] = Field(None, description="Image URL to download")

    @field_validator("base64", "url")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if v else v

    def get_source(self) -> str:
        if self.base64 and self.url:
            raise ValueError("Provide either base64 or url, not both")
        if not self.base64 and not self.url:
            raise ValueError("Provide either base64 or url")
        return "base64" if self.base64 else "url"


class ResizeRequest(BaseModel):
    images: List[ImageInput] = Field(..., min_length=1, description="List of images to resize")
    width: int = Field(..., ge=16, description="Target width in pixels (min 16)")
    height: int = Field(..., ge=16, description="Target height in pixels (min 16)")
    output_format: OutputFormat = Field(OutputFormat.png, description="Output image format")
    quality: Union[int, str] = Field(
        "auto",
        description="Output quality 1-100, or 'auto' for adaptive compression",
    )

    @field_validator("quality", mode="before")
    @classmethod
    def validate_quality(cls, v):
        if isinstance(v, str):
            if v.lower() == "auto":
                return "auto"
            try:
                v = int(v)
            except ValueError:
                raise ValueError("quality must be 'auto' or an integer 1-100")
        if isinstance(v, int) and (v < 1 or v > 100):
            raise ValueError("quality must be between 1 and 100")
        return v


class ResizedImage(BaseModel):
    base64: str
    format: str
    mime_type: str
    original_size: str
    resized_size: str
    quality_used: Optional[int] = None


class ResizeResponse(BaseModel):
    images: List[ResizedImage]
    total: int


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: TaskStatus
    total_images: int


class TaskProgress(BaseModel):
    completed: int
    total: int
    percent: float


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[TaskProgress] = None
    results: Optional[List[ResizedImage]] = None
    error: Optional[str] = None
    created_at: float
    finished_at: Optional[float] = None


class TaskInfo:
    def __init__(self, task_id: str, total: int, created_at: float):
        self.task_id = task_id
        self.status = TaskStatus.pending
        self.total = total
        self.completed = 0
        self.results: List[ResizedImage] = []
        self.error: Optional[str] = None
        self.created_at = created_at
        self.finished_at: Optional[float] = None

    def to_response(self) -> TaskStatusResponse:
        progress = TaskProgress(
            completed=self.completed,
            total=self.total,
            percent=round(self.completed / self.total * 100, 1) if self.total > 0 else 0,
        )
        return TaskStatusResponse(
            task_id=self.task_id,
            status=self.status,
            progress=progress,
            results=self.results if self.status == TaskStatus.completed else None,
            error=self.error,
            created_at=self.created_at,
            finished_at=self.finished_at,
        )


tasks_store: Dict[str, TaskInfo] = {}

MIN_DIMENSION = 16


def _strip_base64_header(data: str) -> bytes:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", data, re.DOTALL)
    raw = match.group(1) if match else data
    missing_padding = len(raw) % 4
    if missing_padding:
        raw += "=" * (4 - missing_padding)
    return base64.b64decode(raw)


def _estimate_original_quality(img: Image.Image, raw_bytes: int) -> int:
    pixel_count = img.width * img.height
    if pixel_count == 0:
        return 75
    bytes_per_pixel = raw_bytes / pixel_count
    if bytes_per_pixel >= 3.0:
        return 95
    elif bytes_per_pixel >= 1.5:
        return 85
    elif bytes_per_pixel >= 0.8:
        return 70
    elif bytes_per_pixel >= 0.3:
        return 55
    else:
        return 40


def _compute_adaptive_quality(
    img: Image.Image,
    raw_bytes: int,
    target_w: int,
    target_h: int,
    output_format: OutputFormat,
) -> int:
    if output_format.value == "png":
        return 0

    estimated = _estimate_original_quality(img, raw_bytes)

    scale_ratio = min(target_w / img.width, target_h / img.height)
    scale_factor = 0.65 + 0.35 * scale_ratio

    pixel_ratio = (target_w * target_h) / (img.width * img.height) if img.width * img.height > 0 else 1.0
    if pixel_ratio < 0.1:
        size_factor = 0.8
    elif pixel_ratio < 0.25:
        size_factor = 0.9
    elif pixel_ratio < 0.5:
        size_factor = 0.95
    else:
        size_factor = 1.0

    quality = round(estimated * scale_factor * size_factor)
    return max(30, min(95, quality))


def _calculate_new_size(orig_w: int, orig_h: int, target_w: int, target_h: int) -> tuple:
    ratio_w = target_w / orig_w
    ratio_h = target_h / orig_h
    ratio = min(ratio_w, ratio_h)
    new_w = round(orig_w * ratio)
    new_h = round(orig_h * ratio)
    if new_w < MIN_DIMENSION:
        new_w = MIN_DIMENSION
        new_h = max(MIN_DIMENSION, round(orig_h * (MIN_DIMENSION / orig_w)))
    if new_h < MIN_DIMENSION:
        new_h = MIN_DIMENSION
        new_w = max(MIN_DIMENSION, round(orig_w * (MIN_DIMENSION / orig_h)))
    return new_w, new_h


def _multi_step_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    cur_w, cur_h = img.width, img.height
    while cur_w > target_w * 2 and cur_h > target_h * 2:
        cur_w = max(target_w, cur_w // 2)
        cur_h = max(target_h, cur_h // 2)
        img = img.resize((cur_w, cur_h), Image.LANCZOS)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


async def _load_image(img_input: ImageInput) -> tuple:
    source = img_input.get_source()
    try:
        if source == "base64":
            data = _strip_base64_header(img_input.base64)
            img = Image.open(io.BytesIO(data))
            return img, len(data)
        else:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(img_input.url)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                return img, len(resp.content)
    except Exception as e:
        raise ValueError(f"Failed to load image from {source}: {e}")


def _resize_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        if img.mode == "P":
            img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    new_w, new_h = _calculate_new_size(img.width, img.height, target_w, target_h)
    resized = _multi_step_resize(img, new_w, new_h)

    scale_ratio = min(new_w / img.width, new_h / img.height)
    if scale_ratio < 0.5:
        sharpness_factor = 1.0 + (0.5 - scale_ratio) * 1.0
        enhancer = ImageEnhance.Sharpness(resized)
        resized = enhancer.enhance(min(sharpness_factor, 1.5))

    return resized


def _encode_image(img: Image.Image, output_format: OutputFormat, quality: int) -> str:
    buf = io.BytesIO()
    pil_format = FORMAT_TO_PIL[output_format.value]

    save_kwargs = {"format": pil_format}
    if pil_format in ("JPEG", "WebP"):
        save_kwargs["quality"] = quality
    if pil_format == "JPEG" and img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
        img = bg

    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resolve_quality(
    quality: Union[int, str],
    img: Image.Image,
    raw_bytes: int,
    target_w: int,
    target_h: int,
    output_format: OutputFormat,
) -> int:
    if quality == "auto":
        return _compute_adaptive_quality(img, raw_bytes, target_w, target_h, output_format)
    return int(quality)


async def _process_single_image(
    img_input: ImageInput,
    width: int,
    height: int,
    output_format: OutputFormat,
    quality: Union[int, str],
) -> ResizedImage:
    img, raw_bytes = await _load_image(img_input)
    original_size = f"{img.width}x{img.height}"
    resized = _resize_image(img, width, height)
    resized_size = f"{resized.width}x{resized.height}"

    resolved_q = _resolve_quality(quality, img, raw_bytes, resized.width, resized.height, output_format)

    try:
        b64 = _encode_image(resized, output_format, resolved_q)
    except Exception as e:
        raise ValueError(f"Encoding failed: {e}")

    return ResizedImage(
        base64=b64,
        format=output_format.value,
        mime_type=FORMAT_TO_MIME[output_format.value],
        original_size=original_size,
        resized_size=resized_size,
        quality_used=resolved_q if output_format.value != "png" else None,
    )


@app.post("/resize", response_model=ResizeResponse)
async def resize_images(req: ResizeRequest):
    results: List[ResizedImage] = []

    for idx, img_input in enumerate(req.images):
        try:
            result = await _process_single_image(
                img_input, req.width, req.height, req.output_format, req.quality
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Image #{idx + 1}: {e}")
        results.append(result)

    return ResizeResponse(images=results, total=len(results))


async def _run_background_task(task_id: str, req: ResizeRequest):
    task = tasks_store[task_id]
    task.status = TaskStatus.processing

    for idx, img_input in enumerate(req.images):
        try:
            result = await _process_single_image(
                img_input, req.width, req.height, req.output_format, req.quality
            )
            task.results.append(result)
        except Exception as e:
            task.error = f"Image #{idx + 1}: {e}"
            task.status = TaskStatus.failed
            task.finished_at = time.time()
            return

        task.completed = idx + 1

    task.status = TaskStatus.completed
    task.finished_at = time.time()


@app.post("/resize/async", response_model=TaskSubmitResponse)
async def submit_resize_task(req: ResizeRequest):
    task_id = uuid.uuid4().hex
    task = TaskInfo(task_id=task_id, total=len(req.images), created_at=time.time())
    tasks_store[task_id] = task

    asyncio.create_task(_run_background_task(task_id, req))

    return TaskSubmitResponse(
        task_id=task_id,
        status=task.status,
        total_images=len(req.images),
    )


@app.get("/resize/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_response()


@app.get("/health")
async def health():
    return {"status": "ok"}
