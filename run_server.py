"""
KIND Decision Engine Server Entrypoint.
Starts the FastAPI System One server with Uvicorn.
"""

import uvicorn
import os
import sys

# Hugging Face authentication
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))

# Ensure workspace is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"============================================================")
    print(f" KIND System One Decision Server")
    print(f" Web Studio:   http://localhost:{port}/")
    print(f" API Endpoint: POST http://localhost:{port}/v1/systemone")
    print(f" Health Check: GET  http://localhost:{port}/health")
    print(f" OpenAPI Docs: GET  http://localhost:{port}/docs")
    print(f"============================================================")
    uvicorn.run("kind.server:app", host=host, port=port, reload=False)
