import cv2
from deepface import DeepFace

# Initialize video capture for webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Analyze the frame using DeepFace
    try:
        result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        
        # Check if result is a list or a dictionary and extract the dominant emotion accordingly
        if isinstance(result, list):
            predicted_emotion = result[0]['dominant_emotion']
        else:
            predicted_emotion = result['dominant_emotion']

        print(predicted_emotion)  # Print the predicted emotion to the console

        # Display the frame with the predicted emotion
        cv2.putText(frame, predicted_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Show the real-time video feed
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
