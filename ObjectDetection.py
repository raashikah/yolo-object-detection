import cv2
import torch
from ultralytics import YOLO

def start_pro_detection():
    # 1. SMART DEVICE SELECTION
    # This checks if a GPU is actually available and ready
    if torch.cuda.is_available():
        device = 0
        print(f"[INFO] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[WARNING] GPU not found or PyTorch +cpu installed. Using CPU.")

    print("[INFO] Initializing YOLOv8 Large...")
    model = YOLO('yolov8l.pt') 

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. USE THE DYNAMIC DEVICE
        # We pass the 'device' variable we created above
        results = model.predict(frame, conf=0.25, imgsz=640, verbose=False, device=device)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Pro (GPU Accelerated)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_pro_detection()