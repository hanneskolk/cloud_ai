from fastapi import APIRouter, UploadFile
import shutil
from inference.batch_infer import process_video

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile):
    input_path = f"inputs/{file.filename}"
    output_path = f"outputs/output_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_video(input_path, output_path)

    return {"output": output_path}