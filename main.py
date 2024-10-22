import numpy as np #type:ignore
import pandas as pd #type: ignore
import requests #type: ignore
import streamlit as st #type:ignore
import moviepy.editor as mp #type:ignore
import openai #type:ignore
import speech_recognition as sr #type:ignore
import pyttsx3 #type:ignore
from pydub import AudioSegment #type:ignore
from io import BytesIO
import os

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Streamlit app layout
st.title("Video Audio Replacement with AI Generated Voice")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    video = mp.VideoFileClip("uploaded_video.mp4")  # Process the complete video
    audio = video.audio
    audio.write_audiofile("extracted_audio.wav")

    # Convert audio to a format suitable for speech recognition
    sound = AudioSegment.from_wav("extracted_audio.wav")
    sound.export("extracted_audio_converted.wav", format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile("extracted_audio_converted.wav") as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
            transcription = None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            transcription = None
    
    if transcription:
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_type = "azure"
        openai.api_version = "2024-08-01-preview"

        if not openai.api_key or not openai.api_base:
            st.error("Azure OpenAI API key or endpoint not found. Please set the AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
        else:
            try:
                gpt_response = openai.ChatCompletion.create(
                    engine="gpt-4o",  # Ensure this matches your deployment name in Azure
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Correct the following transcription: {transcription}"}
                    ]
                )
                corrected_transcription = gpt_response.choices[0].message['content'].strip()
                
                engine.save_to_file(corrected_transcription, "new_audio.mp3")
                engine.runAndWait()
                
                new_audio = mp.AudioFileClip("new_audio.mp3")
                final_video = video.set_audio(new_audio)
                final_video.write_videofile("final_video.mp4")
                
                with open("final_video.mp4", "rb") as f:
                    st.download_button("Download Video with New Audio", f, file_name="final_video.mp4")
            except openai.error.InvalidRequestError as e:
                st.error(f"Error: {e}")
            except openai.error.OpenAIError as e:
                st.error(f"OpenAI API error: {e}")