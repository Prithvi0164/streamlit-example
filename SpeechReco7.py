# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:07:13 2024

@author: prith
"""

import streamlit as st
import tempfile
import os
import datetime
from pydub import AudioSegment

def get_audio_file_info(uploaded_file):
    """
    Get various information about an uploaded audio file including its length, size, 
    modification date, and creation date.
    """
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        # Get audio length
        audio_length = len(AudioSegment.from_file(file_path)) / 1000  # Length in seconds

        # Get file size
        file_size = os.path.getsize(file_path)  # Size in bytes

        # Get file modification date
        modification_time = os.path.getmtime(file_path)
        modification_date = datetime.datetime.fromtimestamp(modification_time).strftime("%Y-%m-%d %H:%M:%S")

        # Get file creation date
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")

        # Delete the temporary file
        os.unlink(file_path)

        return {
            "audio_length_seconds": audio_length,
            "file_size_bytes": file_size,
            "file_modification_date": modification_date,
            "file_creation_date": creation_date
        }
    except Exception as e:
        return f"Error processing file: {e}"

# Streamlit App
st.title("Audio File Information")
uploaded_file = st.file_uploader("Upload your audio file", type=['wav', 'mp3', 'aac'])

if uploaded_file is not None:
    audio_info = get_audio_file_info(uploaded_file)
    if isinstance(audio_info, dict):
        st.write(audio_info)
    else:
        st.error(audio_info)
