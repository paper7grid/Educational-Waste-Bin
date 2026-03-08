import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os
from picamera2 import Picamera2

class VisionSystem:
    def __init__(self, model_path, label_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.labels = self._load_labels(label_path)
        self.interpreter = self._load_model(model_path)
        self.camera = Picamera2()
        self.blue_reduction = 0.6
        
        # Setup TFLite details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_labels(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_model(self, path):
        # Using the standard tf.lite for now; ignore the LiteRT warning
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()  # <--- FIXED: Added the 's'
        return interpreter

    def start_camera(self):
        # XBGR8888 is efficient for Pi, but we'll convert to BGR for OpenCV
        config = self.camera.create_preview_configuration(
            main={"size": (1280, 720), "format": "XBGR8888"}
        )
        self.camera.configure(config)
        self.camera.start()
        
        # Hardware-level color tuning
        self.camera.set_controls({
            "AwbEnable": False, 
            "ColourGains": (2.8, 1.0),
            "AeEnable": True,
            "Contrast": 1.3
        })

    def apply_color_correction(self, frame):
        """Standardizes the BGR channels to remove blue/purple tint."""
        frame_float = frame.astype(np.float32)
        # Red is index 2, Green is 1, Blue is 0 in BGR
        frame_float[:, :, 2] *= 1.5              
        frame_float[:, :, 1] *= 1.1              
        frame_float[:, :, 0] *= self.blue_reduction  
        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def classify_frame(self, frame):
        """Runs the TFLite model on the provided frame."""
        # Convert BGR (OpenCV) to RGB (Model Requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame).resize((224, 224))
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        idx = np.argmax(preds)
        return self.labels[idx], preds[idx] * 100

    def close(self):
        self.camera.stop()
        cv2.destroyAllWindows()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure your files are in the same folder as this script!
    try:
        vision = VisionSystem('model_unquant.tflite', 'labels.txt')
        vision.start_camera()
        
        frame_count = 0
        label, confidence = "Initializing...", 0

        while True:
            # Picamera2 capture_array is very fast
            raw_frame = vision.camera.capture_array()
            
            # Convert from the camera's XBGR format to standard BGR
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
            frame = vision.apply_color_correction(frame)

            # Only run AI every 30 frames to keep the Pi cool
            if frame_count % 30 == 0:
                label, confidence = vision.classify_frame(frame)

            # Display logic
            display = cv2.resize(frame, (800, 600))
            color = (0, 255, 0) if confidence > 70 else (0, 255, 255)
            
            cv2.putText(display, f"{label} ({confidence:.1f}%)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Waste Bin Vision System", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('+'): 
                vision.blue_reduction = max(0.1, vision.blue_reduction - 0.05)
                print(f"Blue Multiplier: {vision.blue_reduction:.2f}")
            elif key == ord('-'): 
                vision.blue_reduction = min(1.5, vision.blue_reduction + 0.05)
                print(f"Blue Multiplier: {vision.blue_reduction:.2f}")
            
            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'vision' in locals():
            vision.close()
        print("System shutdown complete.")
