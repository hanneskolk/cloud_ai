import streamlit as st
import requests

st.title("Drone Detection System")

mode = st.selectbox("Mode", ["Upload Video"])

if mode == "Upload Video":
    file = st.file_uploader("Upload MP4")

    if file:
        st.write("Processing...")
        res = requests.post(
            "http://84.50.156.4:20010/",
            files={"file": file}
        )

        output_path = res.json()["output"]
        st.video(output_path)