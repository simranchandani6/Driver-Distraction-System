import streamlit as st
from PIL import Image
import numpy as np
import subprocess
import time
import tempfile
import os
from ultralytics import YOLO
import cv2 as cv
from video_processor import process_video_with_progress

# --- Page Configuration ---
st.set_page_config(
    page_title="Driver Distraction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Paths ---
model_det_path = "raw17.pt"
model_cls_path = "final_classification.pt"

# --- Model Loading ---
@st.cache_resource
def load_models():
    if not os.path.exists(model_det_path):
        st.error(f"Detection model not found at: {model_det_path}")
        return None, None
    if not os.path.exists(model_cls_path):
        st.error(f"Classification model not found at: {model_cls_path}")
        st.info("Please ensure you have a trained YOLOv8 classification model at the specified path.")
        return None, None
        
    detection_model = YOLO(model_det_path)
    classification_model = YOLO(model_cls_path)
    return detection_model, classification_model

model_det, model_cls = load_models()
# ### NEW & IMPROVED: Decision Fusion Logic ###

# --- Constants for the decision logic (easy to tune) ---
HIGH_CONF_THRESHOLD = 0.95
MID_CONF_LOWER_BOUND = 0.72
CLASSIFIER_GRACE_MARGIN = 0.07 # 7% margin

def make_final_decision(det_class, det_conf, cls_class, cls_conf, file_type):
    """
    Combines predictions using a more robust, multi-layered logic.
    Returns: (final_class, final_confidence, reason_string)
    """
    # --- RULE 1: HIGH-CONFIDENCE PRIORITY ---
    if det_conf >= HIGH_CONF_THRESHOLD or cls_conf >= HIGH_CONF_THRESHOLD:
        if cls_conf > det_conf:
            reason = f"🏆 High-Confidence Priority: Classifier is dominant ({cls_conf:.1%})."
            return cls_class, cls_conf, reason
        else:
            reason = f"🏆 High-Confidence Priority: Detector is dominant ({det_conf:.1%})."
            return det_class, det_conf, reason

    # --- RULE 2: LOW-CONFIDENCE HANDLING ---
    if det_conf < MID_CONF_LOWER_BOUND and cls_conf < MID_CONF_LOWER_BOUND:
        if file_type == "Image":
            # Choose the best available class even if confidence is low
            if cls_conf > det_conf:
                reason = f"⚠️ Low Confidence (Image): Choosing Classifier by default ({cls_conf:.1%})."
                return cls_class, cls_conf, reason
            else:
                reason = f"⚠️ Low Confidence (Image): Choosing Detector by default ({det_conf:.1%})."
                return det_class, det_conf, reason
        else:
            # For video: signal to use previous frame
            reason = f"⚠️ Low Confidence (Video): Both models below {MID_CONF_LOWER_BOUND:.0%}. Reusing last known status."
            return None, None, reason

    # --- RULE 3: MID-CONFIDENCE DECISION LOGIC ---
    if cls_conf > det_conf:
        reason = f"🧠 Classifier Priority: Classifier is stronger in the mid-range ({cls_conf:.1%}) vs Detector ({det_conf:.1%})."
        return cls_class, cls_conf, reason
    else:
        difference = det_conf - cls_conf
        if difference <= CLASSIFIER_GRACE_MARGIN:
            reason = f"🧠 Classifier Priority (Grace Margin): Choosing Classifier ({cls_conf:.1%})."
            return cls_class, cls_conf, reason
        else:
            reason = f"🎯 Detector Priority: Detector is significantly stronger ({det_conf:.1%}) than Classifier ({cls_conf:.1%})."
            return det_class, det_conf, reason

# --- Sidebar ---
st.sidebar.title("🚗 Driver Distraction System")
st.sidebar.write("Choose an option below:")
page = st.sidebar.radio("Select Feature", ["Distraction System","Video Drowsiness Detection"])
class_names = ['drinking', 'hair and makeup', 'operating the radio', 'reaching behind', 'safe driving', 
               'talking on the phone', 'talking to passenger', 'texting']
st.sidebar.subheader("Class Names")
for idx, class_name in enumerate(class_names):
    st.sidebar.write(f"{idx}: {class_name}")

# --- Feature: YOLO Distraction Detection ---
if page == "Distraction System":
    st.title("Driver Distraction System")
    if model_det is None or model_cls is None:
        st.stop() 

    file_type = st.radio("Select file type:", ["Image", "Video"])

    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                st.subheader("Detection & Classification Results")
                
                # Image processing with the new decision logic
                start_time = time.time()
                results_det = model_det(image_np, verbose=False)
                result_det = results_det[0]

                if len(result_det.boxes) > 0:
                    boxes = result_det.boxes
                    best_box = boxes[boxes.conf.argmax()]
                    
                    det_class = result_det.names[int(best_box.cls)]
                    det_conf = best_box.conf.item()

                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                    cropped_image = image_np[y1:y2, x1:x2]

                    results_cls = model_cls(cropped_image, verbose=False)
                    cls_class = results_cls[0].names[results_cls[0].probs.top1]
                    cls_conf = results_cls[0].probs.top1conf.item()
                    
                    processing_time = time.time() - start_time

                    final_class, final_conf, reason = make_final_decision(det_class, det_conf, cls_class, cls_conf,file_type="image")

                    st.markdown("##### Result")
                    # st.success(reason) # Using success to highlight the decision reason

                    st.metric(label="Predicted Class", value=final_class.replace('_', ' ').title())
                    st.metric(label="Confidence Score", value=f"{final_conf:.4f}")
                    st.metric(label="Total Inference Time", value=f"{processing_time:.2f} seconds")

                else:
                    st.warning("No driver/distraction detected in the image.")


    else:  # Video processing
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            temp_input_path = tfile.name
            temp_output_path = tempfile.mktemp(suffix="_distraction_detected.mp4")

            st.subheader("Video Information")
            cap = cv.VideoCapture(temp_input_path)
            fps = cap.get(cv.CAP_PROP_FPS)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            col1, col2 = st.columns(2)
            with col1: st.metric("Duration", f"{duration:.2f} seconds"); st.metric("Original FPS", f"{fps:.2f}")
            with col2: st.metric("Resolution", f"{width}x{height}"); st.metric("Total Frames", total_frames)

            st.subheader("Original Video Preview")
            st.video(uploaded_video)

            if st.button("Process Video for Distraction Detection"):
                TARGET_PROCESSING_FPS = 10
                
                progress_bar = st.progress(0, text="Starting video processing...")
                
                with st.spinner(f"Processing video... This may take a while."):
                    cap = cv.VideoCapture(temp_input_path)
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    out = cv.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                    frame_skip_interval = max(1, round(fps / TARGET_PROCESSING_FPS))
                    
                    frame_count = 0
                    last_best_box_coords = None
                    last_best_box_label = ""
                    last_status_text = "Status: Initializing..."
                    last_status_color = (128, 128, 128)
                    last_final_class = "safe driving"
                    last_final_conf = 0.80
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        frame_count += 1
                        progress = int((frame_count / total_frames) * 100)
                        progress_bar.progress(progress, text=f"Analyzing frame {frame_count}/{total_frames}")
                        
                        annotated_frame = frame.copy()

                        if frame_count % frame_skip_interval == 0:
                            results_det = model_det(annotated_frame, verbose=False)
                            result_det = results_det[0]
                            
                            last_best_box_coords = None

                            if len(result_det.boxes) > 0:
                                best_box = result_det.boxes[result_det.boxes.conf.argmax()]
                                det_class = result_det.names[int(best_box.cls)]
                                det_conf = best_box.conf.item()
                                
                                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                                x1, y1 = max(0, x1), max(0, y1)
                                cropped_frame = annotated_frame[y1:y2, x1:x2]
                                
                                cls_class, cls_conf = 'safe driving', 0.0
                                if cropped_frame.size > 0:
                                    results_cls = model_cls(cropped_frame, verbose=False)
                                    cls_class = results_cls[0].names[results_cls[0].probs.top1]
                                    cls_conf = results_cls[0].probs.top1conf.item()

                                final_class, final_conf, _ = make_final_decision(det_class, det_conf, cls_class, cls_conf,file_type="video")

                                # Handle fallback if decision is None
                                if final_class is None or final_conf is None:
                                    print(f"Frame {frame_count}: Detected {det_class} ({det_conf:.1%}), Classified {cls_class} ({cls_conf:.1%}), Final Decision: None — Reusing last known state.")
                                    final_class = last_final_class
                                    final_conf = last_final_conf
                                    print(f"→ Reused: {final_class} ({final_conf:.1%})")
                                else:
                                    last_final_class = final_class
                                    last_final_conf = final_conf
                                    print(f"Frame {frame_count}: Detected {det_class} ({det_conf:.1%}), Classified {cls_class} ({cls_conf:.1%}), Final Decision: {final_class} ({final_conf:.1%})")

                                last_best_box_coords = (x1, y1, x2, y2)
                                last_best_box_label = f"FINAL: {final_class} ({final_conf:.1%})"
                                
                                if final_class != 'safe driving':
                                    last_status_text = f"Status: {final_class.replace('_', ' ').title()}"
                                    last_status_color = (0, 0, 255)
                                else:
                                    last_status_text = "Status: Safe Driving"
                                    last_status_color = (0, 128, 0)
                            else:
                                last_status_text = "Status: Safe Driving"
                                last_status_color = (0, 128, 0)
                        
                        # Drawing logic
                        if last_best_box_coords:
                            cv.rectangle(annotated_frame, (last_best_box_coords[0], last_best_box_coords[1]), 
                                         (last_best_box_coords[2], last_best_box_coords[3]), (0, 255, 0), 2)
                            cv.putText(annotated_frame, last_best_box_label, 
                                       (last_best_box_coords[0], last_best_box_coords[1] - 10), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

                        # Status Text drawing
                        font_scale, font_thickness = 1.0, 2
                        (text_w, text_h), _ = cv.getTextSize(last_status_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        padding = 10
                        rect_start = (padding, padding)
                        rect_end = (padding + text_w + padding, padding + text_h + padding)
                        cv.rectangle(annotated_frame, rect_start, rect_end, last_status_color, -1)
                        text_pos = (padding + 5, padding + text_h + 5)
                        cv.putText(annotated_frame, last_status_text, text_pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)
                        
                        out.write(annotated_frame)
                    
                    cap.release()
                    out.release()
                    progress_bar.progress(100, text="Video processing completed!")
                    
                    st.success("Video processed successfully!")
                    
                    if os.path.exists(temp_output_path):
                        with open(temp_output_path, "rb") as file: video_bytes = file.read()
                        st.download_button(label="📥 Download Processed Video", data=video_bytes, file_name=f"distraction_detected_{uploaded_video.name}", mime="video/mp4")
                        st.subheader("Sample Frame from Processed Video")
                        cap_out = cv.VideoCapture(temp_output_path)
                        ret, frame = cap_out.read()
                        if ret: st.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), caption="Sample frame with distraction detection", use_container_width=True)
                        cap_out.release()
                
                try:
                    os.unlink(temp_input_path)
                    if os.path.exists(temp_output_path): os.unlink(temp_output_path)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary files: {e}")


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
