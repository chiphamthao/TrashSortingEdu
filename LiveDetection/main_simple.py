from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best.pt")

# Connect to default webcam
print("Opening camera...")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Camera failed to open")
    exit(1)

print("✅ Camera ready")
print("Starting video capture loop...")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️  Failed to read frame from camera")
            break
        
        # Run inference on every frame (no optimizations)
        results = model.predict(frame, conf=0.4)
        
        # Draw predictions on the frame
        display_frame = results[0].plot()
        
        # Display live output
        cv2.imshow("Live Waste Detection", display_frame)
        
        # Stop if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    print("Cleaning up...")
    cam.release()
    cv2.destroyAllWindows()

