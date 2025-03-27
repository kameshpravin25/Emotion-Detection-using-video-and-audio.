import sounddevice as sd
import numpy as np
import wave
import threading
import queue
import tempfile
import os
import time
import google.generativeai as genai
from scipy.io import wavfile
from google.cloud import speech_v1
from datetime import datetime

class AudioEmotionDetector:
    def __init__(self, api_key, duration=5):
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Audio parameters
        self.duration = duration
        self.sample_rate = 44100
        self.channels = 1
        self.audio_queue = queue.Queue()
        
        # Recording state
        self.is_recording = False
        self.temp_dir = tempfile.mkdtemp()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def save_audio(self, audio_data, filename):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for 'int16'
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
    
    def analyze_emotion(self, audio_file):
        """Analyze emotion using Gemini API"""
        try:
            # Convert audio to text first (you might want to use a speech-to-text service here)
            # For this example, we'll send a prompt about the audio characteristics
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
            return response.text
            
        except Exception as e:
            return f"Error analyzing emotion: {str(e)}"
    
    def process_audio(self):
        """Process audio chunks and analyze emotions"""
        while self.is_recording:
            # Collect audio for specified duration
            audio_data = []
            start_time = time.time()
            
            while time.time() - start_time < self.duration:
                try:
                    data = self.audio_queue.get(timeout=1)
                    audio_data.append(data)
                except queue.Empty:
                    continue
                
            if audio_data:
                # Combine audio chunks
                audio_chunk = np.concatenate(audio_data)
                
                # Save to temporary file
                temp_file = os.path.join(self.temp_dir, f"audio_{time.time()}.wav")
                self.save_audio(audio_chunk, temp_file)
                
                # Analyze emotion
                emotion_result = self.analyze_emotion(temp_file)
                print("\nEmotion Analysis:")
                print(emotion_result)
                print("\n" + "="*50 + "\n")
                
                # Clean up
                os.remove(temp_file)
    
    def start(self):
        """Start real-time audio emotion detection"""
        try:
            self.is_recording = True
            
            # Start audio stream
            with sd.InputStream(callback=self.audio_callback,
                              channels=self.channels,
                              samplerate=self.sample_rate):
                print("Started recording... Press Ctrl+C to stop")
                
                # Start processing thread
                process_thread = threading.Thread(target=self.process_audio)
                process_thread.start()
                
                # Keep main thread alive
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.stop()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop recording and processing"""
        self.is_recording = False
        print("\nStopping emotion detection...")
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

# Example usage
if __name__ == "__main__":
    # Replace with your Gemini API key
    GEMINI_API_KEY ="AIzaSyDu-u4TKO92aM8yUSjCoiXM-WJV6v0ODYY"
    
    detector = AudioEmotionDetector(api_key=GEMINI_API_KEY)
    detector.start()