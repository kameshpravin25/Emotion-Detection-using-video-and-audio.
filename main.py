import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (replace 'emotion_model.h5' with your model's filename)
model = load_model('emotion_model.h5')

# Constants for audio recording
CHUNK = 1024              # Number of audio frames per buffer
RATE = 22050              # Sampling rate in Hz
DURATION = 2              # Duration of the audio sample in seconds
FORMAT = pyaudio.paInt16  # Format for audio capture

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream to record audio
stream = audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

def predict_emotion(audio_data):
    # Convert raw audio data to numpy array
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Extract MFCC features from audio sample
    mfccs = librosa.feature.mfcc(y=samples, sr=RATE, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)  # Get the mean of MFCCs for each coefficient

    # Prepare input for the model
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=2)  # Add channel dimension (for CNN)

    # Predict emotion
    prediction = model.predict(mfccs)
    predicted_emotion = np.argmax(prediction)  # Get the predicted class index

    # Define emotion labels (replace with your model's emotion classes)
    emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    return emotion_labels[predicted_emotion]

print("Recording...")

try:
    while True:
        frames = []

        # Record audio for specified duration
        for _ in range(int(RATE / CHUNK * DURATION)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Concatenate audio frames
        audio_data = b''.join(frames)

        # Predict emotion from audio data
        emotion = predict_emotion(audio_data)
        print(f"Detected Emotion: {emotion}")

except KeyboardInterrupt:
    print("Stopped recording")

finally:
    # Close audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
