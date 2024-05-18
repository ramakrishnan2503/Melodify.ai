import os
import streamlit as st
from generater import generate

st.title("Melodify - AI Powered Music Generator")

def generate_music():
    st.write("Generating music...")
    music = generate()
    midi_file_path = "Melody_Generated.mid"
    music.write('midi', fp=midi_file_path)
    st.write("Music generated successfully!")
    return midi_file_path

generate_button = st.button("Generate Music")

if generate_button:
    midi_file_path = generate_music()
    st.write("Download your generated music:")
    st.download_button(label="Download MIDI", data=open(midi_file_path, "rb"), file_name="Melody_Generated.mid", mime="audio/midi")
