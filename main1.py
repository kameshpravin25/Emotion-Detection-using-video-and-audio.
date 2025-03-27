from flask import Flask, render_template, jsonify
import threading
import sounddevice as sd
import numpy as np
import wave
import queue
import tempfile
import os
import time
import google.generativeai as genai
from datetime import datetime

app = Flask(__name__)
detector = None

class AudioEmotionDetector:
    def __init__(self, api_key, duration=5):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.duration = duration
        self.sample_rate = 44100
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.temp_dir = tempfile.mkdtemp()
        self.latest_analysis = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())

    def save_audio(self, audio_data, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def analyze_emotion(self, audio_file):
        try:
            prompt = f"""
            Analyze the emotional content of this audio segment captured at {datetime.now()}.
            Consider these aspects:
            - Volume variations
            - Pitch patterns
            - Speech rhythm
            - Voice quality
            
            Provide a brief emotion analysis with confidence levels for:
            - Primary emotion
            - Secondary emotion
            - Overall mood
            """
            response = self.model.generate_content(prompt)
            self.latest_analysis = response.text
            return response.text
        except Exception as e:
            return f"Error analyzing emotion: {str(e)}"

    def process_audio(self):
        while self.is_recording:
            audio_data = []
            start_time = time.time()
            while time.time() - start_time < self.duration:
                try:
                    data = self.audio_queue.get(timeout=1)
                    audio_data.append(data)
                except queue.Empty:
                    continue
            if audio_data:
                audio_chunk = np.concatenate(audio_data)
                temp_file = os.path.join(self.temp_dir, f"audio_{time.time()}.wav")
                self.save_audio(audio_chunk, temp_file)
                emotion_result = self.analyze_emotion(temp_file)
                print("\nEmotion Analysis:")
                print(emotion_result)
                print("\n" + "="*50 + "\n")
                os.remove(temp_file)

    def start(self):
        try:
            self.is_recording = True
            with sd.InputStream(callback=self.audio_callback,
                                channels=self.channels,
                                samplerate=self.sample_rate):
                print("Started recording... Press Ctrl+C to stop")
                process_thread = threading.Thread(target=self.process_audio)
                process_thread.start()
                while self.is_recording:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"Error: {str(e)}")
            self.stop()

    def stop(self):
        self.is_recording = False
        print("\nStopping emotion detection...")
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global detector
    if detector is None:
        GEMINI_API_KEY = "AIzaSyDu-u4TKO92aM8yUSjCoiXM-WJV6v0ODYY"
        detector = AudioEmotionDetector(api_key=GEMINI_API_KEY)
        threading.Thread(target=detector.start).start()
        return jsonify({"status": "Recording started"})
    return jsonify({"status": "Already recording"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global detector
    if detector is not None:
        detector.stop()
        detector = None
        return jsonify({"status": "Recording stopped"})
    return jsonify({"status": "Not recording"})

@app.route('/get_result', methods=['GET'])
def get_result():
    global detector
    if detector:
        return jsonify({"result": detector.latest_analysis or "No analysis available"})
    return jsonify({"result": "No analysis available"})

if __name__ == "__main__":
    app.run(debug=True)
