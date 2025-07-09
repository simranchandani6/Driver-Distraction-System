# video_processor.py

import cv2 as cv
# We import the necessary functions and state from drowsiness_detection
from drowsiness_detection import process_frame, reset_counters, DROWSY_COUNTER, YAWN_COUNTER, HEAD_DOWN_COUNTER

def process_video_with_progress(input_path, output_path, progress_callback=None):
    """
    Processes a video file to detect drowsiness, providing progress updates.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.
        progress_callback (function, optional): A function to call with progress updates. 
                                                It receives (current_frame, total_frames).
    
    Returns:
        dict: A dictionary containing the final detection statistics.
    """
    reset_counters()  # Ensure all counters are zeroed before starting
    
    video_stream = cv.VideoCapture(input_path)
    if not video_stream.isOpened():
        raise ValueError(f"Could not open video file {input_path}")
    
    # Get video properties for the writer
    fps = video_stream.get(cv.CAP_PROP_FPS)
    original_width = int(video_stream.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv.CAP_PROP_FRAME_COUNT))
    
    # --- FIX: Calculate correct output dimensions to prevent distortion ---
    # The process_frame function resizes frames to a fixed width of 640.
    output_width = 640
    aspect_ratio = original_height / original_width
    output_height = int(output_width * aspect_ratio)
    output_dims = (output_width, output_height)
    
    # Setup video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_path, fourcc, fps, output_dims)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame = process_frame(frame)
            
            # The processed frame is already resized to 640px width.
            # We must ensure it fits the calculated output_dims.
            # If aspect ratios differ slightly, resizing is a safe fallback.
            if processed_frame.shape[1] != output_dims[0] or processed_frame.shape[0] != output_dims[1]:
                processed_frame = cv.resize(processed_frame, output_dims)
                
            video_writer.write(processed_frame)
            
            if progress_callback:
                progress_callback(frame_count, total_frames)
                
        stats = {
            'total_frames_processed': frame_count,
            'drowsy_events': DROWSY_COUNTER,
            'yawn_events': YAWN_COUNTER,
            'head_down_events': HEAD_DOWN_COUNTER
        }
        return stats
        
    finally:
        video_stream.release()
        if video_writer:
            video_writer.release()