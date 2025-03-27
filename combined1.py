from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace
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

# Initialize global variables
detector = None
video_active = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = None

# Helper function to log responses
def log_response(response, endpoint):
    with open("api_responses.txt", "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {endpoint}: {response}\n")

class AudioEmotionDetector:
    def __init__(self, api_key, duration=5):
        # Configure with proper error handling
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            print(f"Error configuring Gemini AI: {str(e)}")
            raise
            
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
            self.latest_analysis = f"Error analyzing emotion: {str(e)}"
            return self.latest_analysis

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
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")

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
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {str(e)}")

def generate_frames():
    global video_capture, video_active
    
    while video_active:
        try:
            success, frame = video_capture.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                except Exception as e:
                    print(f"Error during emotion analysis: {e}")
                    emotion = "Unknown"

                # Draw rectangle and emotion text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (212, 175, 55), 2)  # Gold color
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (212, 175, 55), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break

    if video_capture:
        video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global video_active
    if video_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(status=204)  # No content response if video is not active

@app.route('/start_video', methods=['POST'])
def start_video():
    global video_capture, video_active
    try:
        if not video_active:
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                response = {"status": "Error: Could not open camera"}
                log_response(response, '/start_video')
                return jsonify(response), 500
            video_active = True
            response = {"status": "Video started successfully"}
            log_response(response, '/start_video')
            return jsonify(response)
        response = {"status": "Video already running"}
        log_response(response, '/start_video')
        return jsonify(response)
    except Exception as e:
        response = {"status": f"Error starting video: {str(e)}"}
        log_response(response, '/start_video')
        return jsonify(response), 500

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_capture, video_active
    try:
        if video_active:
            video_active = False
            if video_capture:
                video_capture.release()
            response = {"status": "Video stopped successfully"}
            log_response(response, '/stop_video')
            return jsonify(response)
        response = {"status": "Video not running"}
        log_response(response, '/stop_video')
        return jsonify(response)
    except Exception as e:
        response = {"status": f"Error stopping video: {str(e)}"}
        log_response(response, '/stop_video')
        return jsonify(response), 500

@app.route('/start_audio', methods=['POST'])
def start_audio():
    global detector
    try:
        if detector is None:
            # Replace this with your actual Gemini API key
            GEMINI_API_KEY = "AIzaSyClT0TmFqH16hmiIq_dDTdY4xn55-xosLs"
            detector = AudioEmotionDetector(api_key=GEMINI_API_KEY)
            threading.Thread(target=detector.start).start()
            response = {"status": "Audio recording started successfully"}
            log_response(response, '/start_audio')
            return jsonify(response)
        response = {"status": "Already recording audio"}
        log_response(response, '/start_audio')
        return jsonify(response)
    except Exception as e:
        response = {"status": f"Error starting audio: {str(e)}"}
        log_response(response, '/start_audio')
        return jsonify(response), 500

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    global detector
    try:
        if detector is not None:
            detector.stop()
            detector = None
            response = {"status": "Audio recording stopped successfully"}
            log_response(response, '/stop_audio')
            return jsonify(response)
        response = {"status": "Not recording audio"}
        log_response(response, '/stop_audio')
        return jsonify(response)
    except Exception as e:
        response = {"status": f"Error stopping audio: {str(e)}"}
        log_response(response, '/stop_audio')
        return jsonify(response), 500

@app.route('/get_audio_result', methods=['GET'])
def get_audio_result():
    global detector
    try:
        if detector:
            return jsonify({"result": detector.latest_analysis or "No analysis available yet"})
        return jsonify({"result": "Audio detection not running"})
    except Exception as e:
        return jsonify({"result": f"Error getting audio result: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)