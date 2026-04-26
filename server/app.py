from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from environment import get_env
from environment.models import InspectorAction, ResetRequest

app = FastAPI(
    title="FleetAI - Scalable Oversight Environment",
    description="Multi-agent environment for training and evaluating AI oversight agents that monitor customer support workers.",
    version="2.0.0",
)


@app.get("/")
async def root():
    return {
        "name": "fleet-ai",
        "status": "ok",
        "version": "2.0.0",
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
    try:
        env = get_env()
        result = env.reset(ResetRequest(**body))
        return result.model_dump()
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "Invalid request", "details": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        if "confidence" in body:
            try:
                body["confidence"] = max(0.0, min(1.0, float(body["confidence"])))
            except (ValueError, TypeError):
                body["confidence"] = 0.5
        if "flagged_fields" not in body:
            body["flagged_fields"] = []
        elif isinstance(body["flagged_fields"], str):
            body["flagged_fields"] = [f.strip() for f in body["flagged_fields"].split(",") if f.strip()]
        elif not isinstance(body["flagged_fields"], list):
            body["flagged_fields"] = []
        if "issues" not in body:
            body["issues"] = []
        elif not isinstance(body["issues"], list):
            body["issues"] = []
        if "suggested_corrections" not in body:
            body["suggested_corrections"] = {}
        elif not isinstance(body["suggested_corrections"], dict):
            body["suggested_corrections"] = {}
        if "flagged" not in body:
            body["flagged"] = False
        elif not isinstance(body["flagged"], bool):
            body["flagged"] = bool(body["flagged"])
        env = get_env()
        result = env.step(InspectorAction(**body))
        return result.model_dump()
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "Invalid request", "details": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


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
