from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from inference.video_runner import process_video

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):

    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = f"inputs/{file.filename}"
    output_path = f"outputs/out_{file.filename}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    process_video(input_path, output_path, "models/best.pt")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="output.mp4"
    )