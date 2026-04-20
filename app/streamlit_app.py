import streamlit as st
import requests

st.title("Drone Detection System")

file = st.file_uploader("Upload MP4 Video")

if file:
    st.write("Processing...")

    res = requests.post(
        "http://localhost:8000/upload",
        files={"file": file}
    )

    if res.status_code != 200:
        st.error(res.text)
    else:
        st.write("Looking for response")
        st.success("Done")
        st.video(res.content)