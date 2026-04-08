import time
import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# 1. Start timer
init_start = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Create FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Import Environment
try:
    from my_env.server.my_env_environment import TrainSchedulingEnv
    from my_env.models import TrainAction, TrainObservation
except ImportError as e:
    logger.error(f"Import Error: {e}")
    raise

# 4. POST /reset API (for OpenEnv validation)
@app.post("/reset")
async def reset_env():
    logger.info("Environment reset triggered.")
    return JSONResponse(content={"status": "ok"}, status_code=200)

# 5. UI Dashboard (Fixes the 404 error)
def get_info():
    return {"status": "Active", "env": "Indian Railway Scheduling", "port": 7860}

demo = gr.Interface(
    fn=get_info,
    inputs=[],
    outputs=gr.JSON(),
    title="🚄 My_ENV - Railway Dashboard"
)

# 6. Mount UI to root "/"
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    
    # THE LOG YOU REQUESTED
    duration = time.time() - init_start
    logger.info(f"--- STARTING: My_ENV initialization took {duration:.2f} seconds ---")
    
    # Run on 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)