import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from environment import get_env
from environment.models import Action, ResetRequest, StepResponse, ResetResponse

app = FastAPI(
    title="Customer Support Triage - OpenEnv",
    description="Real-world customer support ticket triage and resolution environment for evaluating LLM agents.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "name": "customer-support-triage",
        "status": "ok",
        "version": "1.0.0",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    env = get_env()
    result = env.reset(ResetRequest(**body))
    return result.model_dump()


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    env = get_env()
    result = env.step(Action(**body))
    return result.model_dump()


@app.get("/state")
async def state():
    env = get_env()
    obs = env.state()
    return obs.model_dump()


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
