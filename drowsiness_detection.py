# drowsiness_detection.py

from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import cv2 as cv
import imutils
import dlib
import argparse
import os

# --- FIXED: Models and Constants with better error handling ---
script_dir = os.path.dirname(os.path.abspath(__file__))
haar_cascade_face_detector = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
dlib_facial_landmark_predictor = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

# Check if required files exist
if not os.path.exists(haar_cascade_face_detector):
    print(f"Warning: Face detector file not found at {haar_cascade_face_detector}")
    # Try to use OpenCV's built-in cascade
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    face_detector = cv.CascadeClassifier(haar_cascade_face_detector)

if not os.path.exists(dlib_facial_landmark_predictor):
    print(f"Error: Dlib predictor file not found at {dlib_facial_landmark_predictor}")
    print("Please download shape_predictor_68_face_landmarks.dat from dlib's website")
    landmark_predictor = None
else:
    landmark_predictor = dlib.shape_predictor(dlib_facial_landmark_predictor)

font = cv.FONT_HERSHEY_SIMPLEX
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_CLOSED_THRESHOLD = 20
MOUTH_ASPECT_RATIO_THRESHOLD = 0.5
MOUTH_OPEN_THRESHOLD = 15
FACE_LOST_THRESHOLD = 25

# --- GLOBAL STATE VARIABLES ---
EYE_THRESH_COUNTER = 0
DROWSY_COUNTER = 0
drowsy_alert = False
YAWN_THRESH_COUNTER = 0
YAWN_COUNTER = 0
yawn_alert = False
FACE_LOST_COUNTER = 0
HEAD_DOWN_COUNTER = 0
head_down_alert = False

# --- FIXED: Audio handling for cloud deployment ---
_audio_initialized = False
_audio_available = False

def _initialize_audio():
    """Initializes audio only if available (for local deployment)."""
    global _audio_initialized, _audio_available
    if _audio_initialized:
        return
    
    try:
        # Check if we're in a cloud environment
        if os.getenv("SPACE_ID") or os.getenv("HUGGINGFACE_SPACE"):
            print("Cloud environment detected - audio disabled")
            _audio_available = False
        else:
            import pygame
            pygame.mixer.init()
            _audio_available = True
            print("Audio initialized successfully.")
    except Exception as e:
        print(f"Audio not available: {e}")
        _audio_available = False
    
    _audio_initialized = True

def play_alarm(sound_file=None):
    """Plays an alarm sound if audio is available."""
    _initialize_audio()
    if not _audio_available:
        return
    
    try:
        import pygame
        if sound_file and os.path.exists(sound_file) and not pygame.mixer.get_busy():
            sound = pygame.mixer.Sound(sound_file)
            sound.play()
    except Exception as e:
        print(f"Could not play sound: {e}")

def generate_alert(final_eye_ratio, final_mouth_ratio):
    global EYE_THRESH_COUNTER, YAWN_THRESH_COUNTER, drowsy_alert, yawn_alert, DROWSY_COUNTER, YAWN_COUNTER
    
    if final_eye_ratio < EYE_ASPECT_RATIO_THRESHOLD:
        EYE_THRESH_COUNTER += 1
        if EYE_THRESH_COUNTER >= EYE_CLOSED_THRESHOLD and not drowsy_alert:
            DROWSY_COUNTER += 1
            drowsy_alert = True
            # Try to play sound if available
            drowsiness_sound = os.path.join(script_dir, "drowsiness-detected.mp3")
            Thread(target=play_alarm, args=(drowsiness_sound,)).start()
    else:
        EYE_THRESH_COUNTER = 0
        drowsy_alert = False

    if final_mouth_ratio > MOUTH_ASPECT_RATIO_THRESHOLD:
        YAWN_THRESH_COUNTER += 1
        if YAWN_THRESH_COUNTER >= MOUTH_OPEN_THRESHOLD and not yawn_alert:
            YAWN_COUNTER += 1
            yawn_alert = True
            # Try to play sound if available
            yawn_sound = os.path.join(script_dir, "yawning-detected.mp3")
            Thread(target=play_alarm, args=(yawn_sound,)).start()
    else:
        YAWN_THRESH_COUNTER = 0
        yawn_alert = False

def detect_facial_landmarks(x, y, w, h, gray_frame):
    """Detect facial landmarks using dlib predictor."""
    if landmark_predictor is None:
        return None
    
    face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    face_landmarks = landmark_predictor(gray_frame, face)
    return face_utils.shape_to_np(face_landmarks)

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_eye_aspect_ratio(shape):
    """Calculate final eye aspect ratio from both eyes."""
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    left_ear = eye_aspect_ratio(shape[lStart:lEnd])
    right_ear = eye_aspect_ratio(shape[rStart:rEnd])
    return (left_ear + right_ear) / 2.0

def mouth_aspect_ratio(mouth):
    """Calculate mouth aspect ratio."""
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def final_mouth_aspect_ratio(shape):
    """Calculate final mouth aspect ratio."""
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    return mouth_aspect_ratio(shape[mStart:mEnd])

def reset_counters():
    """Resets all global counters and alerts for a new processing session."""
    global EYE_THRESH_COUNTER, YAWN_THRESH_COUNTER, FACE_LOST_COUNTER
    global DROWSY_COUNTER, YAWN_COUNTER, HEAD_DOWN_COUNTER
    global drowsy_alert, yawn_alert, head_down_alert
    EYE_THRESH_COUNTER, YAWN_THRESH_COUNTER, FACE_LOST_COUNTER = 0, 0, 0
    DROWSY_COUNTER, YAWN_COUNTER, HEAD_DOWN_COUNTER = 0, 0, 0
    drowsy_alert, yawn_alert, head_down_alert = False, False, False

def process_frame(frame):
    """Processes a single frame to detect drowsiness, yawns, and head position."""
    global FACE_LOST_COUNTER, head_down_alert, HEAD_DOWN_COUNTER
    
    # The output frame will have a fixed width of 640px
    frame = imutils.resize(frame, width=640)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        FACE_LOST_COUNTER = 0
        head_down_alert = False
        (x, y, w, h) = faces[0]
        
        # Draw rectangle around face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect landmarks if predictor is available
        face_landmarks = detect_facial_landmarks(x, y, w, h, gray_frame)
        
        if face_landmarks is not None:
            final_ear = final_eye_aspect_ratio(face_landmarks)
            final_mar = final_mouth_aspect_ratio(face_landmarks)
            generate_alert(final_ear, final_mar)
            
            # Display ratios
            cv.putText(frame, f"EAR: {final_ear:.2f}", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv.putText(frame, f"MAR: {final_mar:.2f}", (10, 60), font, 0.7, (0, 0, 255), 2)
        else:
            # If no landmarks detected, show warning
            cv.putText(frame, "Landmarks not available", (10, 30), font, 0.7, (0, 0, 255), 2)
    else:
        FACE_LOST_COUNTER += 1
        if FACE_LOST_COUNTER >= FACE_LOST_THRESHOLD and not head_down_alert:
            HEAD_DOWN_COUNTER += 1
            head_down_alert = True
            
    # Draw status information
    cv.putText(frame, f"Drowsy: {DROWSY_COUNTER}", (480, 30), font, 0.7, (255, 255, 0), 2)
    cv.putText(frame, f"Yawn: {YAWN_COUNTER}", (480, 60), font, 0.7, (255, 255, 0), 2)
    cv.putText(frame, f"Head Down: {HEAD_DOWN_COUNTER}", (480, 90), font, 0.7, (255, 255, 0), 2)
    
    # Draw alerts
    if drowsy_alert:
        cv.putText(frame, "DROWSINESS ALERT!", (150, 30), font, 0.9, (0, 0, 255), 2)
    if yawn_alert:
        cv.putText(frame, "YAWN ALERT!", (200, 60), font, 0.9, (0, 0, 255), 2)
    if head_down_alert:
        cv.putText(frame, "HEAD NOT VISIBLE!", (180, 90), font, 0.9, (0, 0, 255), 2)
    
    return frame

# --- Command-line execution for local testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drowsiness Detection System (Local Runner)')
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam', help='Mode of operation')
    parser.add_argument('--input', type=str, help='Input video file path for video mode')
    args = parser.parse_args()
    
    # Check if landmark predictor is available
    if landmark_predictor is None:
        print("Error: Dlib facial landmark predictor not found!")
        print("Please download shape_predictor_68_face_landmarks.dat")
        exit(1)
    
    if args.mode == 'webcam':
        print("Starting webcam detection... Press 'q' to quit.")
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
        else:
            reset_counters()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame)
                cv.imshow("Live Drowsiness Detection", processed_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()
            
    elif args.mode == 'video':
        if not args.input or not os.path.exists(args.input):
            print("Error: Please provide a valid --input video file path.")
        else:
            from video_processor import process_video_with_progress
            output_file = args.input.replace('.mp4', '_processed.mp4')
            print(f"Processing video {args.input}, output will be {output_file}")
            
            def cli_progress(current, total):
                percent = int((current / total) * 100)
                print(f"\rProcessing: {percent}%", end="")

            process_video_with_progress(args.input, output_file, progress_callback=cli_progress)
            print("\nDone.")