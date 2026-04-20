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
        try:
            data = res.json()
            st.write(data)
        except Exception as e:
            st.error("Response is not JSON")
            st.write(res.text[:1000])
        st.write(res.headers)
        st.write(res.text)
        st.write(data["output"])
        st.success("Done")
        st.video(res.content)