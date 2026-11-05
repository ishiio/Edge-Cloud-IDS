"""Launcher.  python -m service.run            -> single process (simulated WAN)
             python -m service.run --split    -> cloud tier on :8001 + gateway on :8000 (real HTTP hop)
"""
import argparse
import os
import subprocess
import sys
import time

import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--split", action="store_true")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
if a.split:
    cloud = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.cloud:app", "--port", "8001"])
    os.environ["CLOUD_URL"] = "http://127.0.0.1:8001"
    time.sleep(2)
    try:
        uvicorn.run("service.app:app", host=a.host, port=a.port)
    finally:
        cloud.terminate()
else:
    uvicorn.run("service.app:app", host=a.host, port=a.port)
