from fastapi import FastAPI, UploadFile, File
import shutil
import os

from api.stream import get_stream_response

app = FastAPI()

VIDEO_PATH = "inputs/input.mp4"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)

    with open(VIDEO_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": "uploaded", "stream_url": "/stream"}


@app.get("/stream")
def stream():
    return get_stream_response()