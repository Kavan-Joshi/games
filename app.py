import json
from typing import Any, Dict

from fastapi import FastAPI
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
async def reset(request: ResetRequest):
    env = get_env()
    result = env.reset(request)
    return result.model_dump()


@app.post("/step")
async def step(action: Action):
    env = get_env()
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
async def state():
    env = get_env()
    obs = env.state()
    return obs.model_dump()


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
