# Audio & Video Emotion Analysis with Flask

This project is a Flask web application that performs real-time emotion analysis using both video and audio inputs. It uses computer vision and deep learning techniques to analyze facial expressions and vocal cues, and it integrates with a generative AI model for audio emotion analysis.

## Features

- **Video Emotion Analysis:**  
  Uses OpenCV and DeepFace to capture video from a connected camera, detect faces, and analyze emotions.

- **Audio Emotion Analysis:**  
  Records audio using the `sounddevice` module and processes audio segments. It leverages Google Generative AI (Gemini) to analyze the emotional content of the audio.

- **Web API Endpoints:**  
  Provides endpoints to start/stop video and audio capture, as well as to retrieve analysis results.

- **Logging:**  
  Logs API responses with timestamps to a local file (`api_responses.txt`).

## Prerequisites

- Python 3.7 or higher
- Flask
- OpenCV (`opencv-python`)
- DeepFace (`deepface`)
- SoundDevice (`sounddevice`)
- NumPy (`numpy`)
- Wave (standard library)
- Queue (standard library)
- Tempfile (standard library)
- OS, Time, and DateTime (standard libraries)
- Google Generative AI SDK (`google-generativeai`)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/audio-video-emotion-analysis.git
   cd audio-video-emotion-analysis
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can install dependencies manually:

   ```bash
   pip install flask opencv-python deepface sounddevice numpy google-generativeai
   ```

4. **Configure API Keys:**

   In the code, replace the placeholder Gemini API key with your actual API key:

   ```python
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   ```

## Usage

1. **Run the Flask Application:**

   ```bash
   python combined1.py
   ```

   The app will start in debug mode and be accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000).

2. **Access the Web Interface:**

   Open your browser and navigate to the root URL to access the main page (make sure you have an `index.html` template in your templates directory).

3. **API Endpoints:**

   - **Start Video:**
     - URL: `/start_video`
     - Method: `POST`
     - Description: Opens the camera stream and starts video emotion analysis.

   - **Stop Video:**
     - URL: `/stop_video`
     - Method: `POST`
     - Description: Stops the video stream.

   - **Video Feed:**
     - URL: `/video_feed`
     - Method: `GET`
     - Description: Streams the processed video frames.

   - **Start Audio:**
     - URL: `/start_audio`
     - Method: `POST`
     - Description: Begins recording audio and starts audio emotion analysis.

   - **Stop Audio:**
     - URL: `/stop_audio`
     - Method: `POST`
     - Description: Stops audio recording and cleans up temporary files.

   - **Get Audio Analysis:**
     - URL: `/get_audio_result`
     - Method: `GET`
     - Description: Retrieves the latest audio emotion analysis result.

## Project Structure

```
├── combined1.py          # Main Flask application source code
├── templates/
│   └── index.html        # HTML template for the web interface
├── api_responses.txt     # Log file for API responses (created at runtime)
└── README.md             # Project documentation (this file)
```

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [DeepFace GitHub Repository](https://github.com/serengil/deepface)
- [Google Generative AI](https://cloud.google.com/generative-ai)

