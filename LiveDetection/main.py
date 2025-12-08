from ultralytics import YOLO
import cv2
import torch

# Load trained model
model = YOLO("best.pt")

# Optimize model for inference
model.fuse()  # Fuse model layers for faster inference

# Set device (prioritize MPS on Mac, then CUDA, else CPU)
if torch.backends.mps.is_available():
    device = 'mps'
    print("🚀 Using Apple Metal (MPS) acceleration")
elif torch.cuda.is_available():
    device = 'cuda'
    print("🚀 Using CUDA acceleration")
else:
    device = 'cpu'
    print("⚠️  Using CPU (no GPU acceleration available)")
print(f"Device: {device}")

# Connect to default webcam (0)
# Try AVFoundation first (Mac), then fallback to default backend
print("Attempting to open camera...")
cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# If AVFoundation fails, try default backend
if not cam.isOpened():
    print("⚠️  AVFoundation failed, trying default backend...")
    cam = cv2.VideoCapture(0)
    
    # If that also fails, try camera index 1
    if not cam.isOpened():
        print("⚠️  Camera 0 failed, trying camera 1...")
        cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("❌ Camera failed to open. Please check:")
    print("   - Camera permissions in System Settings > Privacy & Security")
    print("   - Camera is not being used by another application")
    exit(1)
else:
    print("✅ Camera ready")
    
    # Reduce camera resolution for faster processing
    # Common resolutions: 640x480, 320x240 (faster but lower quality)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get actual resolution (might differ from requested)
    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Camera resolution: {actual_width}x{actual_height}")

# Frame skipping for faster processing (process every N frames)
frame_skip = 1  # Set to 2 to process every other frame, 3 for every third, etc.
frame_count = 0
last_results = None  # Store last results for skipped frames

print("Starting video capture loop...")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️  Failed to read frame from camera")
            continue  # Continue instead of breaking to keep trying
        
        frame_count += 1
        
        # Process frame or reuse last results
        if frame_count % frame_skip == 0:
            try:
                # Run inference with optimized settings
                # imgsz: smaller image size = faster inference (416, 320, or 640)
                # half: use FP16 precision if GPU available (faster)
                # verbose: disable verbose output (less overhead)
                results = model.predict(
                    frame, 
                    conf=0.4,
                    imgsz=416,  # Reduced from default (usually 640) for faster processing
                    half=device in ['cuda', 'mps'],  # Use FP16 on GPU (CUDA or MPS) for faster inference
                    verbose=False,  # Disable verbose output
                    device=device
                )
                last_results = results
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                results = last_results  # Use last valid results
        
        # Always show the frame, with or without annotations
        display_frame = frame.copy()  # Default to showing raw frame
        
        # Draw predictions on the frame if we have results
        if last_results and len(last_results) > 0:
            try:
                display_frame = last_results[0].plot()
            except Exception as e:
                print(f"⚠️  Error plotting results: {e}")
                # Continue with raw frame
        
        # Display live output
        cv2.imshow("Live Waste Detection", display_frame)
        
        # Stop if 'q' pressed or window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        # Check if window was closed
        if cv2.getWindowProperty("Live Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window was closed")
            break

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Cleaning up...")

cam.release()
cv2.destroyAllWindows()
