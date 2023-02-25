# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:35:31 2023

@author: prith
"""

import streamlit as st
import speech_recognition as sr

# Create a SpeechRecognition recognizer instance
r = sr.Recognizer()

# Set up the microphone as a source
mic = sr.Microphone()

# Get speech input from the user
st.write("Say something...")
with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

# Perform speech recognition
st.write("Transcribing...")
try:
    text = r.recognize_google(audio)
    st.write(f"You said: {text}")
except sr.UnknownValueError:
    st.write("Sorry, I could not understand what you said.")
except sr.RequestError as e:
    st.write(f"Sorry, there was an error with the speech recognition service: {e}")