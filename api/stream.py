import cv2
from fastapi.responses import StreamingResponse
from inference.pipeline import process_video_stream

VIDEO_PATH = "inputs/input.mp4"


def mjpeg_generator():
    """
    Converts inference frames into MJPEG stream.
    """

    for frame in process_video_stream(VIDEO_PATH):
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


def get_stream_response():
    """
    Returns FastAPI StreamingResponse for browser playback.
    """

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )