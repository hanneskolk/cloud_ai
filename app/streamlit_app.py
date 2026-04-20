import streamlit as st
from stream_utils import frame_generator
import time

st.title("Drone Real-Time Detection System")

video = st.file_uploader("Upload video (or replace with RTSP)")

if video:
    temp_path = "temp.mp4"

    with open(temp_path, "wb") as f:
        f.write(video.read())

    stframe = st.empty()

    for frame_bytes in frame_generator(temp_path, "models/best.pt"):
        stframe.image(frame_bytes)
        time.sleep(0.01)  # controls playback speed