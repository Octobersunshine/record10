from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from trie_filter import SensitiveWordFilter

app = FastAPI(title="敏感词过滤API", version="1.0.0")

_filter = SensitiveWordFilter()

DEFAULT_WORDS_FILE = Path(__file__).parent / "sensitive_words.txt"


class FilterRequest(BaseModel):
    text: str = Field(..., description="待过滤的文本")
    replace_char: str = Field(default="*", description="替换字符，默认为*")


class HitPosition(BaseModel):
    word: str = Field(..., description="匹配到的敏感词（原始词）")
    start: int = Field(..., description="在原文中的起始索引（包含）")
    end: int = Field(..., description="在原文中的结束索引（不包含）")
    match_type: str = Field(..., description="匹配类型：exact/pinyin_first/pinyin_full")
    matched_text: str = Field(..., description="原文中实际匹配到的文本片段")


class FilterResponse(BaseModel):
    original_text: str = Field(..., description="原始文本")
    filtered_text: str = Field(..., description="过滤后文本")
    hit_words: list[str] = Field(..., description="命中的敏感词列表")
    hit_count: int = Field(..., description="命中敏感词数量")
    positions: list[HitPosition] = Field(..., description="敏感词命中位置详情")


class LoadResponse(BaseModel):
    loaded_count: int = Field(..., description="加载的敏感词数量")
    total_count: int = Field(..., description="当前总敏感词数量")


class AddWordRequest(BaseModel):
    word: str = Field(..., description="要添加的敏感词")


class LoadRequest(BaseModel):
    filepath: str = Field(..., description="敏感词文件路径")


class StatusResponse(BaseModel):
    word_count: int = Field(..., description="当前已加载的敏感词总数")
    status: str = Field(default="ok")


@app.on_event("startup")
def _load_default_words() -> None:
    if DEFAULT_WORDS_FILE.exists():
        _filter.load_from_file(str(DEFAULT_WORDS_FILE))


@app.get("/", response_model=StatusResponse)
def get_status():
    return StatusResponse(word_count=_filter.word_count)


@app.post("/filter", response_model=FilterResponse)
def filter_text(req: FilterRequest) -> Any:
    filtered, hit_words, positions = _filter.filter_text(
        req.text, replace_char=req.replace_char
    )
    return FilterResponse(
        original_text=req.text,
        filtered_text=filtered,
        hit_words=hit_words,
        hit_count=len(hit_words),
        positions=[HitPosition(**p.to_dict()) for p in positions],
    )


@app.post("/load", response_model=LoadResponse)
def load_words(req: LoadRequest):
    p = Path(req.filepath)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.filepath}")
    before = _filter.word_count
    _filter.load_from_file(str(p))
    loaded = _filter.word_count - before
    return LoadResponse(loaded_count=loaded, total_count=_filter.word_count)


@app.post("/add-word", response_model=StatusResponse)
def add_word(req: AddWordRequest):
    _filter.add_word(req.word)
    return StatusResponse(word_count=_filter.word_count)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
