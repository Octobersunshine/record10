from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from scheduler.api import init_router, router
from scheduler.callback import close_client, execute_callback
from scheduler.engine import SchedulerEngine
from scheduler.store import TaskStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STORE_PATH = "sqlite:///tasks.db"

store = TaskStore(db_url=STORE_PATH)
engine = SchedulerEngine(store, callback_fn=execute_callback)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    logger.info("Application started, loaded %d tasks from store", len(store.list_all()))
    yield
    await engine.stop()
    await close_client()
    logger.info("Application shut down")


app = FastAPI(title="Task Scheduler", version="1.0.0", lifespan=lifespan)
init_router(store, engine)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
