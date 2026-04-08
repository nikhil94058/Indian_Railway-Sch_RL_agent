import gradio as gr
import subprocess
import os
import sys
import time
import requests

# Set path for local imports
sys.path.append(os.getcwd())

# 1. Start Uvicorn Server in the background
def start_server():
    print("Starting Uvicorn Server on port 8000...")
    return subprocess.Popen([
        "uvicorn", "my_env.server.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

# Initialize the server
server_process = start_server()
time.sleep(5) # Wait for it to boot

# 2. Reset Function (Calls the /reset endpoint on your FastAPI server)
def reset_env():
    try:
        response = requests.post("http://127.0.0.1:8000/reset")
        if response.status_code == 200:
            return "✅ Environment reset successfully."
        else:
            return f"❌ Reset failed (Status: {response.status_code})"
    except Exception as e:
        return f"⚠️ Error connecting to server: {str(e)}"

# 3. Agent Execution Function
def run_agent(debug_mode):
    env = os.environ.copy()
    env["DEBUG"] = "True" if debug_mode else "False"

    process = subprocess.Popen(
        ["python", "inference.py"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    output_log = ""
    for line in iter(process.stdout.readline, ''):
        output_log += line
        yield output_log 
    process.wait()

# 4. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🚂 Indian Railway AI Agent")
    
    with gr.Row():
        debug_toggle = gr.Checkbox(label="🐛 Enable Debug Mode", value=True)
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
    
    run_btn = gr.Button("▶️ Run Agent Benchmark", variant="primary")
    status_output = gr.Textbox(label="System Status", interactive=False)
    logs = gr.Textbox(label="Terminal Logs", lines=20)
    
    # Logic connections
    reset_btn.click(reset_env, outputs=status_output)
    run_btn.click(run_agent, inputs=[debug_toggle], outputs=logs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)