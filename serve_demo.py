"""Part 14 demo: start the server, then use it the way a client app would: over HTTP.

    python serve_demo.py
Needs the ONNX star model once: python -m reviewsense.serving.onnx_model export
While it runs, open http://localhost:8000/ui (web page) and http://localhost:8000/docs (API).
"""
import threading
import time

import httpx
import uvicorn

BASE = "http://localhost:8000"

# 1. Start the server in the background (in production: the container's CMD, see Dockerfile).
server = uvicorn.Server(uvicorn.Config("reviewsense.serving.api:app", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

# 2. Wait until it is ready: /health answers at once, /ready only after the models are loaded.
start = time.time()
while True:
    try:
        if httpx.get(f"{BASE}/ready", timeout=2).status_code == 200:
            break
    except httpx.TransportError:
        pass                                              # not listening yet
    time.sleep(1)
print(f"Ready after {time.time() - start:.0f}s\n")

# 3. Call it like any client: JSON in, JSON out.
reply = httpx.post(f"{BASE}/stars", json={"texts": ["Best tacos in town, friendly staff!",
                                                     "Cold pizza and a rude manager."]})
print("POST /stars ", reply.json(), " request id:", reply.headers["X-Request-ID"])
reply = httpx.post(f"{BASE}/search", json={"query": "slwo servce", "k": 2})
print("POST /search", [hit["text"][:60] for hit in reply.json()["results"]])
for message in ["Book a table at Golden Dragon for 2 tomorrow at 7 pm", "yes"]:
    reply = httpx.post(f"{BASE}/chat", json={"session_id": "demo", "message": message}, timeout=120)
    print("POST /chat  ", reply.json()["answer"])
reply = httpx.post(f"{BASE}/stars", json={"texts": []})
print("POST /stars with no texts ->", reply.status_code, reply.json()["detail"][0]["msg"])

# 4. Keep serving so you can try the web page and the API docs.
input(f"\nOpen {BASE}/ui or {BASE}/docs - press Enter to stop the server. ")
server.should_exit = True
