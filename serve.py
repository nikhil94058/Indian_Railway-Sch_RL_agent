#!/usr/bin/env python3
"""
serve.py — Live Dashboard Server
=================================
Run this from the project root to serve the dashboard.

Usage:
    python serve.py

Then open: http://localhost:3000/frontend/index.html
The dashboard auto-polls simulation_state.json every 2 seconds.
"""
import http.server
import socketserver
import webbrowser
import threading
import os
import sys

PORT = 3000
ROOT = os.path.dirname(os.path.abspath(__file__))


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    """Serves from project root. Suppresses noisy access logs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, fmt, *args):
        # Only log errors, not every GET request
        if args and len(args) >= 2 and not str(args[1]).startswith("2"):
            sys.stderr.write(f"[serve] {fmt % args}\n")


def open_browser():
    import time
    time.sleep(0.6)
    url = f"http://localhost:{PORT}/frontend/index.html"
    webbrowser.open(url)


threading.Thread(target=open_browser, daemon=True).start()

print("=" * 55)
print("  IR Dispatch — Live Dashboard Server")
print("=" * 55)
print(f"  Dashboard : http://localhost:{PORT}/frontend/index.html")
print(f"  State     : http://localhost:{PORT}/my_env/data/simulation_state.json")
print(f"  Network   : http://localhost:{PORT}/my_env/data/network.json")
print(f"  Serving   : {ROOT}")
print()
print("  Start the simulation in another terminal:")
print("    1. uvicorn my_env.server.app:app --port 8000")
print("    2. python inference.py")
print()
print("  Dashboard auto-refreshes every 2 seconds. Ctrl+C to stop.")
print("=" * 55)

with socketserver.TCPServer(("", PORT), SilentHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
