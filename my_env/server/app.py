# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the My Env Environment.

This module creates an HTTP server that exposes the TrainSchedulingEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from my_env.models import TrainObservation, TrainAction
    from my_env.server.my_env_environment import TrainSchedulingEnv
    MyAction = TrainAction
    MyObservation = TrainObservation
except ModuleNotFoundError:
    from models import MyAction, MyObservation
    from server.my_env_environment import TrainSchedulingEnv


# Create the app with web interface and README integration
app = create_app(
    TrainSchedulingEnv,
    MyAction,
    MyObservation,
    env_name="indian-railway-scheduling",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@app.get("/")
async def health_check():
    return {"status": "ok"}

def main():
    """
    Standard entry point for OpenEnv multi-mode deployment.
    """
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()