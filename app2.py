import asyncio
import sys

# --- Boilerplate for compatibility ---
if sys.platform.startswith('linux') and sys.version_info >= (3, 8):
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception:
        pass

import streamlit as st
from PIL import Image
import numpy as np
import subprocess
import time
import tempfile
import os
from ultralytics import YOLO
import cv2 as cv

# --- NEW: Import your refactored video processing logic ---
from video_processor import process_video_with_progress

# --- FIXED: Model path handling ---
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it's included in your deployment.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Driver Distraction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
st.sidebar.title("🚗 Driver Distraction System")
st.sidebar.write("Choose an option below:")

# --- FIXED: Disable webcam feature for cloud deployment ---
if os.getenv("SPACE_ID"):  # Running on Hugging Face Spaces
    available_features = [
        "Distraction System",
        "Video Drowsiness Detection"
    ]
    st.sidebar.info("💡 Note: Real-time webcam detection is not available in cloud deployment.")
else:
    available_features = [
        "Distraction System",
        "Video Drowsiness Detection"
    ]

# --- Sidebar navigation ---
page = st.sidebar.radio("Select Feature", available_features)

# --- Class Labels (for YOLO model) ---
st.sidebar.subheader("Class Names")
class_names = ['drinking', 'hair and makeup', 'operating the radio', 'reaching behind',
               'safe driving', 'talking on the phone', 'talking to passenger', 'texting']
for idx, class_name in enumerate(class_names):
    st.sidebar.write(f"{idx}: {class_name}")

# --- Feature: YOLO Distraction Detection ---
if page == "Distraction System":
    st.title("Driver Distraction System")
    st.write("Upload an image or video to detect distractions using YOLO model.")

    # File type selection
    file_type = st.radio("Select file type:", ["Image", "Video"])

    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                image_np = np.array(image)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.subheader("Detection Results")
                    
                    # Load model with error handling
                    try:
                        model = YOLO(model_path)
                        start_time = time.time()
                        results = model(image_np)
                        end_time = time.time()
                        prediction_time = end_time - start_time
                        
                        result = results[0]
                        if len(result.boxes) > 0:
                            boxes = result.boxes
                            confidences = boxes.conf.cpu().numpy()
                            classes = boxes.cls.cpu().numpy()
                            class_names_dict = result.names
                            max_conf_idx = confidences.argmax()
                            predicted_class = class_names_dict[int(classes[max_conf_idx])]
                            confidence_score = confidences[max_conf_idx]
                            st.markdown(f"### Predicted Class: **{predicted_class}**")
                            st.markdown(f"### Confidence Score: **{confidence_score:.4f}**  ({confidence_score*100:.1f}%)")
                            st.markdown(f"Inference Time: {prediction_time:.2f} seconds")
                        else:
                            st.warning("No distractions detected.")
                    except Exception as e:
                        st.error(f"Error loading or running model: {str(e)}")
                        st.info("Please ensure the model file 'best.pt' is present and valid.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif file_type == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if uploaded_file is not None:
            try:
                # Create a temporary file to hold the uploaded video
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                temp_input_path = tfile.name
                temp_output_path = tempfile.mktemp(suffix="_processed.mp4")

                st.subheader("Original Video Preview")
                st.video(uploaded_file)

                if st.button("Process Video for Distraction Detection"):
                    progress_bar = st.progress(0, text="Preparing to process video...")
                    
                    try:
                        model = YOLO(model_path)
                        cap = cv.VideoCapture(temp_input_path)
                        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv.CAP_PROP_FPS)
                        
                        # Get video properties
                        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                        
                        # Setup video writer
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        out = cv.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                        
                        frame_count = 0
                        detections = []
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            
                            # Process frame with YOLO
                            results = model(frame)
                            result = results[0]
                            
                            # Draw detections on frame
                            annotated_frame = result.plot()
                            out.write(annotated_frame)
                            
                            # Store detection info
                            if len(result.boxes) > 0:
                                boxes = result.boxes
                                for i in range(len(boxes)):
                                    conf = boxes.conf[i].cpu().numpy()
                                    cls = int(boxes.cls[i].cpu().numpy())
                                    class_name = result.names[cls]
                                    detections.append({
                                        'frame': frame_count,
                                        'class': class_name,
                                        'confidence': conf
                                    })
                            
                            # Update progress
                            progress = int((frame_count / total_frames) * 100)
                            progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}")
                        
                        cap.release()
                        out.release()
                        
                        st.success("Video processed successfully!")
                        
                        # Show results
                        st.subheader("Detection Results")
                        if detections:
                            # Count detections by class
                            class_counts = {}
                            for det in detections:
                                class_name = det['class']
                                if class_name not in class_counts:
                                    class_counts[class_name] = 0
                                class_counts[class_name] += 1
                            
                            # Display metrics
                            cols = st.columns(len(class_counts))
                            for i, (class_name, count) in enumerate(class_counts.items()):
                                cols[i].metric(class_name.title(), count)
                        else:
                            st.info("No distractions detected in the video.")
                        
                        # Offer processed video for download
                        if os.path.exists(temp_output_path):
                            with open(temp_output_path, "rb") as file:
                                video_bytes = file.read()
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name=f"distraction_detected_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                    finally:
                        # Cleanup
                        try:
                            if os.path.exists(temp_input_path):
                                os.unlink(temp_input_path)
                            if os.path.exists(temp_output_path):
                                os.unlink(temp_output_path)
                        except Exception as e:
                            st.warning(f"Failed to clean up temporary files: {e}")
                            
            except Exception as e:
                st.error(f"Error handling video upload: {str(e)}")

# --- Feature: Video Drowsiness Detection ---
elif page == "Video Drowsiness Detection":
    st.title("📹 Video Drowsiness Detection")
    st.write("Upload a video file to detect drowsiness and generate a report.")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"])

    if uploaded_video is not None:
        try:
            # Create a temporary file to hold the uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            temp_input_path = tfile.name
            temp_output_path = tempfile.mktemp(suffix="_processed.mp4")

            st.subheader("Original Video Preview")
            st.video(uploaded_video)

            if st.button("Process Video for Drowsiness Detection"):
                progress_bar = st.progress(0, text="Preparing to process video...")
                
                # --- Define a callback function for the progress bar ---
                def streamlit_progress_callback(current, total):
                    if total > 0:
                        percent_complete = int((current / total) * 100)
                        progress_bar.progress(percent_complete, text=f"Analyzing frame {current}/{total}...")

                try:
                    with st.spinner("Processing video... This may take a while."):
                        # Call your robust video processing function
                        stats = process_video_with_progress(
                            input_path=temp_input_path,
                            output_path=temp_output_path,
                            progress_callback=streamlit_progress_callback
                        )
                    
                    progress_bar.progress(100, text="Video processing completed!")
                    st.success("Video processed successfully!")


                    # Offer the processed video for download
                    if os.path.exists(temp_output_path):
                        with open(temp_output_path, "rb") as file:
                            video_bytes = file.read()
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name=f"drowsiness_detected_{uploaded_video.name}",
                            mime="video/mp4"
                        )
                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                    st.info("Please ensure all required model files are present and the video format is supported.")
                finally:
                    # Cleanup temporary files
                    try:
                        if os.path.exists(temp_input_path):
                            os.unlink(temp_input_path)
                        if os.path.exists(temp_output_path):
                            os.unlink(temp_output_path)
                    except Exception as e_clean:
                        st.warning(f"Failed to clean up temporary files: {e_clean}")
                        
        except Exception as e:
            st.error(f"Error handling video upload: {str(e)}")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Notes")
st.sidebar.markdown("""
- **Image Detection**: Upload JPG, PNG images
- **Video Detection**: Upload MP4, AVI, MOV videos
- **Cloud Limitations**: Webcam access not available in cloud deployment
- **Model**: Uses YOLO for distraction detection
""")