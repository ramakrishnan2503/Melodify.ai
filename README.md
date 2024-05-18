# Melodify.ai - AI Powered Music Generator 🎶

Welcome to **Melodify**, your personal AI composer! Melodify uses advanced deep learning models to create unique, captivating musical compositions with just a click. Currently, our model is trained exclusively on Chopin's classical pieces, ensuring a touch of timeless elegance in every generated melody. However, you can expand its creativity by training the model on a broader dataset. Easily generate, download, and convert MIDI files to WAV, all within our user-friendly interface.


## How It Works

### Model and Training Process

Melodify's music generation capability is built on a Sequential model with LSTM (Long Short-Term Memory) layers, designed to handle sequence prediction tasks like music composition. Our training process involves:

1. **Dataset:** We obtained a diverse dataset of classical MIDI files, which you can access [here](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi).
2. **Preprocessing:** Extracting notes and chords, while filtering out rare elements to streamline learning.
3. **Model Training:** The model learns to predict subsequent notes based on previous sequences, ensuring coherent and melodically pleasing outputs.
4. **Music Generation:** The trained model generates new musical pieces by sampling from learned sequences, producing unique compositions every time.

## Getting Started

To start creating your own music with Melodify, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ramakrishnan2503/Melodify.ai.git
   cd Melodify.ai

2. **Install Dependencies:**
    ```bash
   pip install tensorflow numpy pandas music21 pydub mido streamlit

3. **Run the application:**
    ```bash
    streamlit run app.py

Kindly ensure that you have changed all the path variables according to you
